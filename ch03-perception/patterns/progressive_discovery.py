# =============================================================================
# patterns/progressive_discovery.py — 渐进发现模式（Listing 3.3）
#
# 【学习目标】
#   理解 Agent 在不知道去哪里找信息时，如何从宽到窄逐步缩小搜索范围。
#
# 【解决的问题】
#   Agent 接到任务时，往往不知道相关文件在哪里。
#   如果一次把所有文件都读进来：代价太高，可能超出上下文。
#   如果什么都不读：LLM 没有足够上下文，只能凭空猜测。
#   需要一个"先粗后细"的策略，高效找到相关内容。
#
# 【类比：侦探办案】
#   第一阶段（广撒网）：去案发现场周围问所有人，记下可疑线索
#   第二阶段（精读线索）：仔细阅读最关键的几条证词
#   第三阶段（顺藤摸瓜）：根据证词里提到的人，再去找那些人问
#
# 【三个阶段】
#   Phase 1 Forage（觅食）: 用廉价工具广撒网，找到候选文件列表
#   Phase 2 Focus（聚焦）:  完整读取最高价值的几个文件
#   Phase 3 Deepen（深入）: 追踪读到的文件里的 import，递归扩展
# =============================================================================

from typing import List, Protocol


# =============================================================================
# 工具接口定义（Protocol）
# 【说明】
#   Protocol 是 Python 的"鸭子类型接口"，不需要继承，
#   只要一个类有这些方法就可以传入 ProgressiveDiscovery 使用。
#   这样设计让模式与具体工具实现解耦。
# =============================================================================
class FileTools(Protocol):
    def exists(self, path: str) -> bool: ...   # 文件是否存在
    def read(self, path: str) -> str: ...      # 读取文件内容
    def glob(self, pattern: str) -> List[str]: ...  # 按通配符匹配文件名
    def grep(self, query: str) -> List[str]: ...    # 在文件内容中搜索


# =============================================================================
# 渐进发现器
# =============================================================================
class ProgressiveDiscovery:

    def __init__(self,
                 tools: FileTools,
                 max_forage: int = 15,   # #A Phase 1 最多发现多少个候选文件
                 max_focus: int = 5,     # Phase 2 最多完整读取几个文件
                 max_deepen: int = 3):   # Phase 3 最多追踪几个 import
        self.tools = tools
        self.limits = (max_forage, max_focus, max_deepen)

    def explore(self, task: str) -> List[str]:
        """
        【方法说明】
        对任务描述执行三阶段探索，返回收集到的文件内容列表。

        参数：task - 用户的任务描述，如"重构 auth 模块的登录函数"
        返回：文件内容字符串列表，每项格式为 "### 文件路径\n文件内容"
        """
        targets  = self._forage(task)    # #B Phase 1: 找到候选文件路径列表
        contents = self._focus(targets)  # Phase 2: 完整读取这些文件
        return   self._deepen(contents)  # Phase 3: 追踪 import 扩展

    # =========================================================================
    # Phase 1: Forage（觅食）— 用廉价操作找候选文件
    # 【说明】
    #   "廉价"指不需要完整读取文件内容，只是通过文件名、关键词快速定位。
    #   三个策略并行：读项目说明文件 + 文件名匹配 + 内容关键词搜索
    # =========================================================================
    def _forage(self, task: str) -> List[str]:
        """Phase 1: 广撒网，找到候选文件路径列表"""
        targets = []

        # 策略1: 读项目说明文件（最容易找到项目结构信息）
        for m in ["CLAUDE.md", "README.md", "package.json", "pyproject.toml"]:
            if self.tools.exists(m):
                targets.append(m)

        # 策略2: 从任务描述提取关键词，按文件名匹配
        # #C 提取长度 > 4 的单词作为关键词（过滤 the/is/in 等短词），最多取 3 个
        keywords = [w for w in task.lower().split() if len(w) > 4][:3]
        for kw in keywords:
            # glob("**/*auth*") 匹配所有路径中含有 auth 的文件
            targets.extend(self.tools.glob(f"**/*{kw}*")[:5])  # 每个关键词最多5个结果

        # 策略3: 在文件内容中搜索关键词（比文件名匹配更深入）
        for kw in keywords:
            targets.extend(self.tools.grep(kw)[:5])

        # 去重并限制总数
        # 【说明】
        #   这个去重技巧利用了 set 的特性：
        #   seen.add(t) 返回 None（falsy），所以 `not (t in seen or seen.add(t))`
        #   当 t 不在 seen 中时：t in seen=False，执行 seen.add(t) 返回 None，
        #   not(False or None) = not None = True → 保留
        #   当 t 已在 seen 中时：t in seen=True，不执行 add，
        #   not(True or ...) = not True = False → 过滤
        seen = set()
        return [t for t in targets
                if not (t in seen or seen.add(t))][:self.limits[0]]

    # =========================================================================
    # Phase 2: Focus（聚焦）— 完整读取高价值文件
    # =========================================================================
    def _focus(self, targets: List[str]) -> List[str]:
        """Phase 2: 完整读取候选文件，限制数量避免过多"""
        contents = []
        # #D 只读取前 max_focus 个（默认5个），控制读取代价
        for path in targets[:self.limits[1]]:
            content = self.tools.read(path)
            # 每项格式化为 "### 文件名\n内容"，方便后续 LLM 识别文件边界
            contents.append(f"### {path}\n{content}")
        return contents

    # =========================================================================
    # Phase 3: Deepen（深入）— 追踪 import，扩展相关文件
    # =========================================================================
    def _deepen(self, contents: List[str]) -> List[str]:
        """
        Phase 3: 分析读到的文件里的 import 语句，追踪并读取被导入的文件。

        【为什么要追踪 import？】
        代码审查时，如果修改了 utils.py，但 utils.py 里 import 了 config.py，
        config.py 的内容可能对理解修改很重要，应该一起纳入上下文。
        """
        import re
        deepened = 0

        for content in list(contents):  # list() 避免迭代时修改列表
            # 用正则匹配 Python 的 import 语句
            # 匹配 "from xxx import" 或 "import xxx" 格式
            # #E re.MULTILINE 让 ^ 匹配每行开头（而不只是字符串开头）
            for match in re.finditer(
                r'^(?:from|import)\s+([\w.]+)', content, re.MULTILINE
            ):
                if deepened >= self.limits[2]:
                    return contents  # 达到追踪上限，停止

                # 把模块名转成文件路径：utils.helpers → utils/helpers.py
                path = match.group(1).replace(".", "/") + ".py"

                if self.tools.exists(path):
                    contents.append(
                        f"### {path} (import)\n{self.tools.read(path)}"
                    )
                    deepened += 1

        return contents


# =============================================================================
# 【学习小结】
#
# 三个阶段的代价和收益：
#
#   Phase 1 Forage   代价低（文件名/关键词搜索），覆盖广（最多15个候选）
#       ↓
#   Phase 2 Focus    代价中（完整读取），精准（最多5个文件）
#       ↓
#   Phase 3 Deepen   代价低（只追踪已读文件的 import），扩展关联（最多3个）
#
# 渐进式的优势：
#   避免"一次性读所有文件"（成本太高）
#   避免"什么都不读"（LLM 没有上下文）
#   在成本和信息质量之间找到平衡
#
# 参数调节：
#   max_forage/focus/deepen 可以根据 token 预算和任务复杂度调整
# =============================================================================
