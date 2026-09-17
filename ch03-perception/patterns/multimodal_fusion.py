# =============================================================================
# patterns/multimodal_fusion.py — 多模态融合模式（Listing 3.4）
#
# 【学习目标】
#   理解 Agent 如何统一处理不同格式的输入（文字、图片、结构化数据）。
#
# 【解决的问题】
#   现实中 Agent 收到的输入五花八门：
#     - 用户发来一段文字说明
#     - 附上了一张 UI 截图
#     - 还有一份 CSV 数据
#   LLM 需要的是统一格式的上下文，不能直接"混合"处理。
#   融合层负责把这些不同格式转换成 LLM 能理解的统一表示。
#
# 【类比：翻译官】
#   多国代表发言（中文、英文、法文），翻译官统一译成一种语言，
#   让主席（LLM）只需要处理一种格式的输入。
#
# 【三种模态的处理策略】
#   TEXT（文本）       → 直接透传，不做处理
#   STRUCTURED（结构化）→ 转成紧凑的 Markdown 表格
#   IMAGE（图片）      → 判断是否需要空间理解：
#                          需要 → 保留原始图片字节（视觉模型处理）
#                          不需要 → 用文字描述替代（节省 token）
# =============================================================================

from dataclasses import dataclass
from enum import Enum
from typing import List, Union


# =============================================================================
# 模态枚举：三种输入类型
# =============================================================================
class Modality(Enum):
    TEXT       = "text"        # 普通文本
    IMAGE      = "image"       # 图片（bytes）
    STRUCTURED = "structured"  # 结构化数据（JSON、CSV 等）


# =============================================================================
# 单个输入项
# =============================================================================
@dataclass
class ModalInput:
    modality: Modality           # 这个输入是什么类型
    content: Union[str, bytes]   # 内容（文本用 str，图片用 bytes）
    description: str             # 描述（如"登录页截图"、"用户数据表"）
    token_estimate: int          # 预估 token 占用


# =============================================================================
# 融合结果
# =============================================================================
@dataclass
class FusedContext:
    text_parts: List[str]    # 所有文本内容（文本 + 结构化转换后的表格）
    image_refs: List[bytes]  # 需要保留的图片（只有空间相关的图片）
    total_tokens: int        # 总 token 估算
    # #A 决策日志：记录每个输入的处理方式（方便调试和理解）
    decisions: List[str]


# =============================================================================
# 多模态融合器
# =============================================================================
class MultiModalFusion:

    def fuse(self, inputs: List[ModalInput]) -> FusedContext:
        """
        【方法说明】
        遍历所有输入，按模态分别处理，汇总成统一的 FusedContext。

        参数：inputs - 各种模态的输入列表
        返回：FusedContext - 统一处理后的上下文
        """
        text_parts, image_refs, decisions = [], [], []
        total = 0

        for inp in inputs:

            # -----------------------------------------------------------------
            # 文本：直接透传
            # #B 【说明】文本是 LLM 的原生格式，不需要任何转换。
            # -----------------------------------------------------------------
            if inp.modality == Modality.TEXT:
                text_parts.append(inp.content)
                total += inp.token_estimate
                decisions.append(f"{inp.description}: text passthrough")  # #B

            # -----------------------------------------------------------------
            # 结构化数据：转成紧凑 Markdown 表格
            # #C 【说明】
            #   JSON/CSV 直接放进上下文会很占空间（大量花括号、引号）。
            #   转成 Markdown 表格更紧凑，LLM 也更容易理解。
            # -----------------------------------------------------------------
            elif inp.modality == Modality.STRUCTURED:
                compact = self._to_compact_table(inp.content)
                text_parts.append(compact)
                total += len(compact) // 4
                decisions.append(f"{inp.description}: structured to table")  # #C

            # -----------------------------------------------------------------
            # 图片：根据是否需要空间理解来决定处理方式
            # -----------------------------------------------------------------
            elif inp.modality == Modality.IMAGE:
                if self._needs_spatial(inp.description):
                    # #D 需要空间理解：保留原始图片，交给视觉模型处理
                    # 【说明】
                    #   "空间理解"指需要分析图片里元素的位置、布局关系，
                    #   比如：UI 截图（按钮在哪里）、流程图（连接关系）、
                    #   架构图（组件位置）等。这类必须保留图片本身。
                    image_refs.append(inp.content)
                    total += inp.token_estimate
                    decisions.append(f"{inp.description}: spatial, keep image")  # #D
                else:
                    # 不需要空间理解：用文字描述替代，节省大量 token
                    # 【说明】
                    #   比如一张产品 Logo 图，只需要知道"这是一张 Logo"，
                    #   不需要分析它的布局，用文字描述 [Logo图片] 就够了。
                    text_parts.append(f"[{inp.description}]")
                    total += 50  # 占位符很短，约 50 token
                    decisions.append(f"{inp.description}: no spatial, text")

        return FusedContext(text_parts, image_refs, total, decisions)

    def _needs_spatial(self, desc: str) -> bool:
        """
        【方法说明】
        判断这张图片是否需要空间理解（即必须看图片本身，而不是文字描述）。

        【判断依据】
        描述里是否包含空间相关关键词：
          layout（布局）、screenshot（截图）、diagram（图表）、
          UI（界面）、wireframe（线框图）、position（位置）、alignment（对齐）

        #E 【说明】any() 短路求值：找到第一个匹配就返回 True，不会继续检查。
        """
        spatial_keywords = [
            "layout", "screenshot", "diagram", "UI",
            "wireframe", "position", "alignment"
        ]
        return any(kw in desc.lower() for kw in spatial_keywords)  # #E

    def _to_compact_table(self, data) -> str:
        """
        【方法说明】
        把 JSON/CSV 等结构化数据转成紧凑的 Markdown 表格。

        示例：
          输入：[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
          输出：
            | name  | age |
            |-------|-----|
            | Alice | 30  |
            | Bob   | 25  |

        注：此处用 ... 占位，实际实现需根据数据格式编写解析逻辑。
        """
        ...  # 实际使用时实现具体的转换逻辑


# =============================================================================
# 【学习小结】
#
# 三种模态的处理决策树：
#
#   输入
#    ├── TEXT       → 直接放入文本区域（0 转换成本）
#    ├── STRUCTURED → 转成 Markdown 表格（节省 token，提升可读性）
#    └── IMAGE
#         ├── 需要空间理解？ YES → 保留图片字节（高 token 成本，但必要）
#         └── 需要空间理解？ NO  → 用 [描述] 替代（极低 token 成本）
#
# 核心设计思想：
#   不是所有信息都需要"原样"传给 LLM，
#   根据 LLM 真正需要的信息类型选择最合适的表示方式，
#   在信息完整性和 token 效率之间取得平衡。
#
# decisions 字段的意义：
#   记录每个输入是怎么被处理的，方便调试，
#   也让整个感知过程"可解释"（知道 Agent 看到了什么）。
# =============================================================================
