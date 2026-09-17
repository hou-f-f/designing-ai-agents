# =============================================================================
# patterns/semantic_compaction.py — 语义压缩模式（Listing 3.2）
#
# 【学习目标】
#   理解当对话历史太长时，如何智能压缩旧内容，同时保留关键信息。
#
# 【解决的问题】
#   Agent 执行长任务时，对话历史会越来越长，最终超出上下文窗口。
#   简单截断会丢失重要信息；全量保留又装不下。
#   需要"有选择地压缩"：压缩不重要的，保护重要的。
#
# 【类比】
#   会议纪要：开完3小时会，不需要逐字记录，
#   但关键决策、待办事项、负责人必须保留，
#   冗长讨论过程可以压缩成一句"讨论了X方案，决定采用Y"。
#
# 【三级压缩策略】
#   Level 0: 不需要压缩（总量 ≤ 目标）
#   Level 1: 清除冗长的工具输出（错误信息保留）
#   Level 2: 用 LLM 总结旧内容（错误信息仍然保留原文）
# =============================================================================

from dataclasses import dataclass
from typing import List


# =============================================================================
# 对话轮次：表示一条历史消息
# =============================================================================
@dataclass
class Turn:
    role: str        # 消息角色："user"（用户）/ "assistant"（AI）/ "tool_result"（工具返回）
    content: str     # 消息内容
    tokens: int      # 这条消息占用的 token 数
    is_error: bool = False  # 是否是错误信息（错误要特殊保护，不能随意压缩）


# =============================================================================
# 语义压缩器：核心逻辑
# =============================================================================
class SemanticCompactor:

    def __init__(self, llm, preserve_recent: int = 5):
        self.llm = llm
        # #A 保护最近 N 轮：这些绝对不压缩
        # 【说明】
        #   最近的对话是当前任务最相关的上下文，不能压缩。
        #   只压缩"旧的"部分，保留"新的"部分原样。
        self.preserve_recent = preserve_recent  # #A

    def compact(self, turns: List[Turn], target: int) -> List[Turn]:
        """
        【方法说明】
        压缩对话历史到 target token 以内。

        参数：
            turns  - 完整的对话历史列表
            target - 目标 token 上限
        返回：
            压缩后的对话历史列表
        """
        total = sum(t.tokens for t in turns)

        # #B 已经在预算内：不需要压缩，直接返回
        if total <= target:
            return turns  # #B

        # 把历史分成两段：
        #   old（旧的）= 可以压缩的部分
        #   recent（最近的）= 绝对保护，不动
        boundary = len(turns) - self.preserve_recent
        old    = turns[:boundary]   # 旧内容，可以压缩
        recent = turns[boundary:]   # 最近 N 轮，原样保留

        # ---------------------------------------------------------------------
        # Level 1：清除冗长的工具输出
        # 【说明】
        #   工具调用的返回结果往往很长（如搜索结果、文件内容），
        #   但大部分是冗余的。可以把超过 500 token 的工具结果替换成占位符，
        #   告诉 LLM"这里有结果，需要时重新运行工具"。
        #   错误信息不处理（错误对调试很重要）。
        # ---------------------------------------------------------------------
        cleared = self._clear_tools(old)
        if self._fits(cleared + recent, target):
            return cleared + recent  # Level 1 够用了，不需要 Level 2

        # ---------------------------------------------------------------------
        # Level 2：用 LLM 总结旧内容
        # 【说明】
        #   Level 1 还不够，需要更激进的压缩。
        #   把所有非错误的旧内容交给 LLM 总结成一段摘要。
        #   错误信息（is_error=True）保留原文，因为错误堆栈丢失了无法恢复。
        # ---------------------------------------------------------------------
        non_errors = [t for t in old if not t.is_error]
        errors     = [t for t in old if t.is_error]   # #C 错误信息单独保留
        summary    = self._summarize(non_errors)       # #D LLM 总结非错误内容

        # 最终结构：[总结] + [原始错误] + [最近N轮]
        return [summary] + errors + recent

    def _clear_tools(self, turns: List[Turn]) -> List[Turn]:
        """
        【方法说明】
        Level 1 压缩：把冗长的工具返回替换成占位符。
        只处理超过 500 token 且不是错误的工具结果。
        """
        result = []
        for t in turns:
            if (t.role == "tool_result"
                    and t.tokens > 500       # 超过 500 token 才压缩
                    and not t.is_error):     # 错误信息不动
                # 替换成简短占位符，告知 LLM 原始数据在哪里可以重取
                result.append(Turn(
                    role="tool_result",
                    content=f"[Cleared: {t.tokens} tokens. Re-run to retrieve.]",
                    tokens=25,       # 占位符很短
                    is_error=False,
                ))
            else:
                result.append(t)
        return result

    def _summarize(self, turns: List[Turn]) -> Turn:
        """
        【方法说明】
        Level 2 压缩：用 LLM 把多轮对话压缩成一段摘要。

        【关键：总结时要求保留什么】
        不是让 LLM 随意总结，而是明确要求：
          - 保留：所有决策、文件路径、函数名、当前进度、剩余工作
          - 丢弃：冗余工具输出、已放弃的方案
        这样压缩后的摘要对 Agent 继续工作依然有效。
        """
        # 把所有轮次拼成文本，格式：[角色]: 内容
        text = "\n".join(f"[{t.role}]: {t.content}" for t in turns)

        # #E 调用 LLM 生成摘要，明确指定保留什么、丢弃什么
        summary = self.llm.generate(
            f"Summarize, preserving:\n"
            f"- All decisions made\n"
            f"- All file paths and function names\n"
            f"- Current progress and remaining work\n"
            f"Discard: redundant tool outputs, abandoned approaches.\n\n"
            f"{text}"
        )  # #E

        # 返回一个 system 角色的摘要 Turn
        return Turn(
            role="system",
            content=f"[Summary]\n{summary}",
            tokens=len(summary) // 4,
        )

    def _fits(self, turns: List[Turn], target: int) -> bool:
        """检查 turns 的总 token 数是否在 target 以内"""
        return sum(t.tokens for t in turns) <= target


# =============================================================================
# 【学习小结】
#
# 压缩的三个级别（逐步升级）：
#
#   Level 0  ──→  总量已在预算内，原样返回
#      ↓ 还不够
#   Level 1  ──→  清除冗长工具输出（替换为占位符），保留错误
#      ↓ 还不够
#   Level 2  ──→  LLM 总结旧对话（保留错误原文），最近N轮不动
#
# 两个"永远保护"的原则：
#   1. 最近 N 轮对话：当前任务最相关
#   2. 错误信息（is_error=True）：错误堆栈一旦丢失很难恢复
#
# 为什么要让 LLM 来总结，而不是简单截断？
#   截断 = 按时间丢弃，可能丢掉早期的关键决策
#   LLM 总结 = 按语义保留，能抓住真正重要的信息
# =============================================================================
