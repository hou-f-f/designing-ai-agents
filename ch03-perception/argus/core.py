# =============================================================================
# argus/core.py — Argus Ch3 快照：在 Ch2 基础上加入感知层
#
# 【学习目标】
#   理解 Ch3 相比 Ch2 的核心进化：
#   Ch2 的 review_diff 接收 diff + 可选 context 字符串，但 context 是空的，
#   调用者需要自己拼好内容传进来。
#   Ch3 在 LLM 调用之前加了一个感知管道，自动从代码库里收集上下文。
#
# 【Ch2 → Ch3 的变化】
#   Ch2: review_diff(diff, context="")  ← context 是调用者手动传的字符串
#   Ch3: review_diff(diff, repo_root)   ← 自动从 repo 里发现并筛选上下文
#
# 【这个文件的位置】
#   argus/core.py 是 Argus 这一章的"总装"文件，
#   把 perception.py 里的感知管道和 LLM 调用组合在一起。
# =============================================================================

import json
import os
from pathlib import Path

# 从同包的 perception 模块导入感知管道
from .perception import gather_review_context, PerceptionTrace, FileContext

# 自动加载项目根目录的 .env 配置
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.split("#")[0].strip()
            os.environ.setdefault(_k.strip(), _v)


def _format_context(selected: list[FileContext]) -> str:
    """
    【函数说明】
    把感知层选出的 FileContext 列表，格式化成 LLM 能读的 Prompt 片段。

    每个文件格式化为：
      ### 文件路径  [relevance类型, token数]
      ```
      文件内容
      ```

    【为什么要加 relevance 和 tokens 标注？】
    让 LLM 知道每个文件的"角色"（是被修改的？还是依赖？还是测试？），
    有助于 LLM 做出更准确的审查判断。
    """
    sections = []
    for ctx in selected:
        sections.append(
            f"### {ctx.path}  [{ctx.relevance}, {ctx.tokens} tokens]\n"
            f"```\n{ctx.content}\n```"
        )
    return "\n\n".join(sections)


def review_diff(
    diff: str,
    repo_root: str = ".",
    budget: int = 50_000,
) -> tuple[dict, PerceptionTrace]:
    """
    【函数说明】
    Ch3 版 Argus：带感知层的代码审查，完整 PRA 循环。

    参数：
        diff      - git diff 格式的代码变更
        repo_root - 代码仓库根目录（默认当前目录）
        budget    - 感知层的 token 预算（默认 50,000）

    返回：
        (review_json, perception_trace)
        review_json      - 审查结果字典（summary + comments）
        perception_trace - 感知过程的统计记录（可观测性）

    【返回两个值的原因】
    感知追踪（trace）对调试非常重要：
    你可以通过 trace 知道 Agent 看了哪些文件、丢了哪些、用了多少 token，
    而不是把 Agent 当成黑盒。
    """
    import anthropic  # 懒加载：让 perception 模块在没有 LLM SDK 时也能导入
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    model    = os.environ.get("ARGUS_MODEL", "claude-sonnet-4-6@default")
    client   = anthropic.Anthropic(**({"base_url": base_url} if base_url else {}))

    # =========================================================================
    # P — PERCEPTION（感知）
    # 【Ch3 新增的核心步骤】
    # 调用感知管道，自动发现并筛选与 diff 相关的文件
    # gather_review_context 内部会：
    #   1. 提取 diff 修改的文件
    #   2. 追踪 import 依赖
    #   3. 找测试文件
    #   4. 加载项目配置
    #   5. 按优先级在 budget 内贪心选取
    # =========================================================================
    selected, trace = gather_review_context(diff, repo_root, budget)
    context = _format_context(selected) if selected else ""

    # =========================================================================
    # R — REASONING（推理）
    # 和 Ch2 相同的 LLM 调用，但现在 user_msg 里包含了真实的项目上下文
    # =========================================================================
    system = """You are Argus, an expert code reviewer.
Respond with JSON: {"summary": "...", "comments": [
  {"file": "...", "line": ..., "severity": "...", "message": "..."}
]}"""

    user_msg = f"## Diff:\n```diff\n{diff}\n```"
    if context:
        # Ch3 新增：把感知层选出的上下文追加进 prompt
        user_msg += f"\n\n## Project context:\n{context}"

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    # =========================================================================
    # A — ACTION（行动）
    # 解析 LLM 返回的 JSON，和 Ch2 相同
    # =========================================================================
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    review = json.loads(text)

    # 同时返回审查结果和感知追踪，调用方可以检查 Agent 的感知决策
    return review, trace
