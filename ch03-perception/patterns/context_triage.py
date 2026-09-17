# =============================================================================
# patterns/context_triage.py — 上下文分诊模式（Listing 3.1）
#
# 【学习目标】
#   理解当信息量超过 LLM 上下文窗口时，如何按优先级智能筛选内容。
#
# 【解决的问题】
#   LLM 有 token 上限（如 180,000 token），但 Agent 收集到的信息可能远超这个限制。
#   不能随机丢弃，要保留最重要的，丢掉不那么重要的。
#
# 【类比】
#   急诊室分诊：病人（信息）涌入时，护士按病情严重程度排序，
#   重症优先处理，轻症等待。不是先到先得，而是优先级决定顺序。
#
# 【可以直接运行】
#   python patterns/context_triage.py   （无需 API Key）
# =============================================================================

from dataclasses import dataclass
from enum import IntEnum
from typing import List


# =============================================================================
# 优先级枚举：定义4个重要程度等级
# 【说明】
#   IntEnum 让枚举值可以直接用数字比较大小（4 > 3 > 2 > 1）。
#   每个等级对应不同类型的信息：
# =============================================================================
class Priority(IntEnum):
    CRITICAL   = 4   # 系统提示、安全规则、当前任务描述 — 绝对不能丢
    IMPORTANT  = 3   # 当前正在操作的文件、最近的结果、错误堆栈 — 尽量保留
    SUPPORTING = 2   # 背景文档、较早的对话历史 — 空间够就放
    DEFERRABLE = 1   # 可以通过工具随时取回的内容 — 先不放，需要时再加载


# =============================================================================
# 上下文条目：描述一块信息
# 【说明】
#   @dataclass 自动生成 __init__、__repr__ 等方法，不用手写。
#   每个 ContextItem 代表一块要放进上下文的内容（如一个文件、一段历史）。
# =============================================================================
@dataclass
class ContextItem:
    name: str            # 标识符，如文件名或描述
    content: str         # 实际内容
    priority: Priority   # 优先级
    token_estimate: int = 0    # 预估占用的 token 数量
    is_error: bool = False     # 是否是错误信息（错误信息有特殊保护）

    def __post_init__(self):
        # #A 自动估算 token 数
        # 【说明】
        #   __post_init__ 在 @dataclass 的 __init__ 执行完后自动调用。
        #   这里用"字符数 ÷ 4"来粗略估算 token 数，
        #   因为英文平均每个 token 约 4 个字符（中文约 1-2 字符/token）。
        #   如果调用时已经传入了 token_estimate，就不覆盖。
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4  # #A


# =============================================================================
# 上下文分诊器：核心逻辑
# =============================================================================
class ContextTriage:

    def __init__(self, budget: int = 180_000):
        # #B Token 预算：最多能放多少 token
        # 【说明】
        #   180_000 是 Python 数字字面量的下划线写法，等于 180000，
        #   只是为了可读性（类似 180,000）。
        #   这个值对应很多主流模型的上下文窗口大小。
        self.budget = budget  # #B

    def triage(self, items: List[ContextItem]):
        """
        【方法说明】
        对所有条目排序，贪心填充：从高优先级开始，装满为止。

        参数：items - 所有候选信息条目
        返回：(selected, deferred)
          selected - 选中放入上下文的条目列表
          deferred - 被推迟的条目列表（DEFERRABLE 级别，需要时用工具取）
        """

        # #C 排序：三个排序键，依次比较
        # 【说明】
        #   sorted() 的 key 返回一个元组，Python 按元组元素依次比较。
        #   排序键1: priority.value → 优先级高的排前面
        #   排序键2: 2.0 if is_error else 0.0 → 同优先级下，错误信息排前面
        #   排序键3: len(content) → 同等条件下，内容越长越靠前
        #            （先放长内容，这样如果空间不够，丢掉的是小内容）
        #   reverse=True → 从大到小排（最重要的在最前面）
        sorted_items = sorted(
            items,
            key=lambda x: (
                x.priority.value,           # 主排序：优先级
                2.0 if x.is_error else 0.0, # 次排序：错误优先
                len(x.content),             # 三排序：长内容优先
            ),
            reverse=True,
        )  # #C

        selected, deferred, tokens_used = [], [], 0

        for item in sorted_items:
            # #D DEFERRABLE 级别：直接放到 deferred，不参与上下文填充
            # 【说明】
            #   DEFERRABLE 的内容（如工具可以取到的文档）不浪费上下文空间，
            #   需要时 Agent 主动调用工具获取。
            if item.priority == Priority.DEFERRABLE:
                deferred.append(item)  # #D
                continue

            # 贪心填充：还有空间就放进去
            if tokens_used + item.token_estimate <= self.budget:
                selected.append(item)
                tokens_used += item.token_estimate
            # 空间不足：这个条目被丢弃（优先级低的已经排在后面了）

        return selected, deferred


# =============================================================================
# 演示：不需要 API Key，直接运行看效果
# =============================================================================
if __name__ == "__main__":
    # 模拟 Agent 收集到的各类信息
    items = [
        ContextItem("系统提示",     "You are Argus...",         Priority.CRITICAL,   tokens=500),
        ContextItem("当前任务",     "Review this diff...",      Priority.CRITICAL,   tokens=300),
        ContextItem("修改的文件",   "def get_user()..." * 100,  Priority.IMPORTANT,  tokens=5000),
        ContextItem("错误堆栈",     "Traceback: ...",           Priority.IMPORTANT,  tokens=800, is_error=True),
        ContextItem("背景文档",     "Project overview..." * 50, Priority.SUPPORTING, tokens=2000),
        ContextItem("旧对话历史",   "Earlier: ..." * 200,       Priority.SUPPORTING, tokens=8000),
        ContextItem("工具可取文档", "API Reference..." * 500,   Priority.DEFERRABLE, tokens=20000),
    ]

    triage = ContextTriage(budget=10_000)  # 限制 10000 token 演示效果
    selected, deferred = triage.triage(items)

    print("=== 选中放入上下文 ===")
    for item in selected:
        print(f"  [{item.priority.name:10}] {item.name} ({item.token_estimate} tokens)")

    print(f"\n=== 推迟（DEFERRABLE，需要时用工具取）===")
    for item in deferred:
        print(f"  {item.name} ({item.token_estimate} tokens)")

# =============================================================================
# 【学习小结】
#
# 分诊的两个核心决策：
#   1. 按优先级排序（不是先来先得）
#   2. 贪心填充（从最重要的开始，装满为止）
#
# DEFERRABLE 级别的意义：
#   不是"不重要"，而是"工具可以取到，不需要预加载"。
#   这体现了 Agent 设计的重要原则：
#   上下文窗口 ≠ 知识的全部，工具可以弥补。
# =============================================================================
