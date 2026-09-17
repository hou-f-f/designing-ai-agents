# =============================================================================
# argus/perception.py — Argus 感知管道（Listings 3.5a-e）
#
# 【学习目标】
#   理解如何把本章 4 个感知模式组合起来，应用到真实的 Argus Agent 中。
#   这是"模式 → 实战"的桥梁文件。
#
# 【与 patterns/ 的关系】
#   patterns/ 里是独立、抽象的模式（可复用于任何场景）
#   这个文件是模式在 Argus 场景下的具体应用（代码审查）
#
# 【这个文件做什么】
#   给定一段 diff 和代码仓库根目录，自动收集审查所需的上下文：
#     Step 1: 找到被修改的文件（直接相关）
#     Step 2: 追踪这些文件的 import（间接相关）
#     Step 3: 找到对应的测试文件（验证影响范围）
#     Step 4: 加载项目配置文件（了解代码规范）
#     Step 5: 按优先级排序，在 token 预算内贪心选取
#
# 【可以直接导入，无需 API Key】
# =============================================================================

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

# 模块级日志器：logging 是 Python 标准库的日志模块
# 使用层级命名（argus.perception），方便统一配置日志级别
logger = logging.getLogger("argus.perception")


# =============================================================================
# 文件上下文条目
# =============================================================================
@dataclass
class FileContext:
    path: str       # 相对于 repo_root 的文件路径
    content: str    # 文件内容
    # #A relevance（相关性类型）：决定了这个文件的优先级
    #   "modified"  → 被 diff 直接修改的文件，最重要
    #   "imported"  → 被修改文件 import 的文件，间接相关
    #   "test"      → 修改文件对应的测试，验证影响范围
    #   "config"    → 项目配置文件，了解代码规范
    relevance: str
    tokens: int = 0

    def __post_init__(self):
        # #B 自动计算 token 估算值（字符数 ÷ 4）
        if self.tokens == 0:
            self.tokens = len(self.content) // 4  # #B


# =============================================================================
# 感知追踪：记录感知过程的统计数据（可观测性）
# 【说明】
#   Agent 的决策过程应该是可解释的，PerceptionTrace 记录：
#   发现了多少文件、选了多少、丢了多少、token 使用率等。
#   这些数据对调试 Agent 行为非常有用。
# =============================================================================
@dataclass
class PerceptionTrace:
    """感知过程的可观测记录"""
    files_discovered: int = 0   # 总共发现了多少相关文件
    files_selected: int = 0     # 最终选入上下文的文件数
    files_dropped: int = 0      # 因超出预算被丢弃的文件数
    tokens_considered: int = 0  # 所有候选文件的总 token
    tokens_selected: int = 0    # 选中文件的总 token
    dropped_files: list = field(default_factory=list)  # 被丢弃的文件列表

    @property
    def selectivity(self) -> float:
        """
        选择率：选中的 token / 考虑的总 token
        #C 值越低说明过滤越激进（大量信息被丢弃）
        """
        if self.tokens_considered == 0:
            return 0
        return self.tokens_selected / self.tokens_considered  # #C


# =============================================================================
# 核心函数：收集代码审查所需的上下文
# =============================================================================
def gather_review_context(
    diff: str,
    repo_root: str,
    budget: int = 50_000,   # #A 默认 token 预算：50,000
) -> tuple[list[FileContext], PerceptionTrace]:
    """
    【函数说明】
    分析 diff，从代码库里收集最相关的文件上下文。

    参数：
        diff      - git diff 格式的代码变更
        repo_root - 代码仓库根目录路径
        budget    - token 预算上限（默认 50,000）

    返回：
        (selected, trace)
        selected - 选中的 FileContext 列表（已排序，在预算内）
        trace    - 感知过程的统计信息
    """
    contexts: list[FileContext] = []
    trace = PerceptionTrace()

    # =========================================================================
    # Step 1: 提取 diff 修改的文件（最高优先级）
    # #B 【说明】修改的文件是代码审查最直接的对象，必须包含。
    # =========================================================================
    modified = _extract_modified_files(diff)  # #B
    for fpath in modified:
        full = os.path.join(repo_root, fpath)
        if os.path.exists(full):
            content = Path(full).read_text(errors="ignore")
            contexts.append(FileContext(fpath, content, "modified"))

    # =========================================================================
    # Step 2: 追踪 import 的依赖文件（理解修改的上下游影响）
    # #C 【说明】
    #   如果修改了 auth.py，而 auth.py import 了 db.py，
    #   那么 db.py 对理解这次修改可能很关键（比如数据库连接方式）。
    # =========================================================================
    for ctx in list(contexts):  # list() 拷贝一份，避免迭代时修改
        if ctx.path.endswith(".py"):
            for imp in _find_imports(ctx.content):
                imp_path = os.path.join(repo_root, imp)
                if os.path.exists(imp_path) and imp not in modified:
                    content = Path(imp_path).read_text(errors="ignore")
                    contexts.append(FileContext(imp, content, "imported"))  # #C

    # =========================================================================
    # Step 3: 找到修改文件对应的测试文件
    # #A 【说明】
    #   测试文件帮助理解"这个函数预期的行为是什么"，
    #   审查时可以验证修改是否破坏了现有测试逻辑。
    # =========================================================================
    for test in _find_test_files(modified, repo_root):
        if test not in [c.path for c in contexts]:
            full = os.path.join(repo_root, test)
            if os.path.exists(full):
                content = Path(full).read_text(errors="ignore")
                contexts.append(FileContext(test, content, "test"))    # #A

    # =========================================================================
    # Step 4: 加载项目配置文件
    # #B 【说明】
    #   项目配置（如 pyproject.toml）包含代码风格规范、lint 规则等，
    #   有助于判断修改是否符合项目规范。
    # =========================================================================
    for cfg in ["pyproject.toml", "setup.cfg", ".flake8"]:            # #B
        cfg_path = os.path.join(repo_root, cfg)
        if os.path.exists(cfg_path):
            content = Path(cfg_path).read_text(errors="ignore")
            contexts.append(FileContext(cfg, content, "config"))       # #C

    # =========================================================================
    # Step 5: 按优先级排序，在 token 预算内贪心选取
    # #A 【说明】
    #   优先级顺序：modified(0) > imported/test(1) > config(2)
    #   数字越小优先级越高（sort 默认升序）
    # =========================================================================
    priority_order = {"modified": 0, "imported": 1, "test": 1, "config": 2}  # #A
    contexts.sort(key=lambda c: priority_order.get(c.relevance, 3))

    # 统计：发现了多少、总共多少 token
    trace.files_discovered = len(contexts)
    trace.tokens_considered = sum(c.tokens for c in contexts)

    # 贪心填充：从高优先级开始，装满预算为止
    selected, used = [], 0
    for ctx in contexts:
        if used + ctx.tokens <= budget:
            selected.append(ctx)
            used += ctx.tokens
        else:
            # #B 记录被丢弃的文件（用于调试）
            trace.dropped_files.append((ctx.path, ctx.relevance))     # #B

    # 更新追踪统计
    trace.files_selected = len(selected)
    trace.files_dropped = trace.files_discovered - trace.files_selected
    trace.tokens_selected = used

    # 输出日志（info 级别正常信息，warning 级别记录丢弃情况）
    logger.info(
        "Perception: %d/%d files selected (%.0f%% selectivity), "
        "%d tokens used of %d budget",
        trace.files_selected, trace.files_discovered,
        trace.selectivity * 100, used, budget,
    )
    if trace.dropped_files:
        # #C 有文件被丢弃时发出警告，提示可能遗漏了相关信息
        logger.warning("Dropped: %s", trace.dropped_files)            # #C

    return selected, trace


# =============================================================================
# 辅助函数
# =============================================================================

def _extract_modified_files(diff: str) -> list[str]:
    """
    从 diff 文本中提取被修改的文件路径列表。

    【diff 格式说明】
    git diff 的文件头长这样：
      --- a/src/auth.py    ← 修改前
      +++ b/src/auth.py    ← 修改后
    用正则提取 +++ 和 --- 后面的文件路径（去掉 a/ 或 b/ 前缀）。

    #A dict.fromkeys() 保持顺序去重（Python 3.7+ 字典保持插入顺序）
    """
    matches = re.findall(r'^(?:\+\+\+|---) [ab]/(.+)$',
                         diff, re.MULTILINE)  # #A
    return list(dict.fromkeys(matches))  # 去重保序


def _find_imports(source: str) -> list[str]:
    """
    从 Python 源代码中提取所有 import 的模块，转成文件路径。

    示例：
      from utils.helpers import xxx  →  utils/helpers.py
      import config                  →  config.py

    #B 把模块名里的 . 替换成 /，加上 .py 后缀
    """
    imports = []
    for m in re.finditer(
            r'^(?:from|import)\s+([\w.]+)',
            source, re.MULTILINE):
        path = m.group(1).replace(".", "/") + ".py"  # #B
        imports.append(path)
    return imports


def _find_test_files(modified: list[str], repo_root: str) -> list[str]:
    """
    为每个修改的文件找到对应的测试文件。

    #C 按照 Python 项目的测试文件命名惯例，尝试三个候选路径：
      tests/test_模块名.py
      test/test_模块名.py
      同目录/test_模块名.py
    """
    tests = []
    for fpath in modified:
        name = Path(fpath).stem  # 文件名去掉扩展名，如 auth.py → auth
        candidates = [
            f"tests/test_{name}.py",
            f"test/test_{name}.py",
            f"{Path(fpath).parent}/test_{name}.py",  # #C 同目录下
        ]
        for c in candidates:
            if os.path.exists(os.path.join(repo_root, c)):
                tests.append(c)
    return tests


# =============================================================================
# 【学习小结】
#
# 整个感知管道的逻辑链：
#
#   diff 输入
#     ↓
#   [Step 1] 提取直接修改的文件        ← relevance="modified"，优先级最高
#     ↓
#   [Step 2] 追踪 import 依赖          ← relevance="imported"，理解上下游
#     ↓
#   [Step 3] 找测试文件                ← relevance="test"，验证影响范围
#     ↓
#   [Step 4] 加载项目配置              ← relevance="config"，了解规范
#     ↓
#   [Step 5] 排序 + 预算内贪心选取     ← 应用 Context Triage 模式
#     ↓
#   输出：FileContext 列表 + PerceptionTrace
#
# PerceptionTrace 的价值：
#   - 调试：为什么某个文件没被选中？（查 dropped_files）
#   - 监控：selectivity 低说明信息过滤激进，可能需要增大 budget
#   - 可解释性：让 Agent 的"看了什么"变得透明
# =============================================================================
