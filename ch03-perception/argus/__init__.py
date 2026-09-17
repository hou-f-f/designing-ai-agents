"""
argus/__init__.py — Ch3 Argus 包的对外接口

【说明】
__init__.py 的作用是定义这个包（argus/目录）对外暴露什么。
当别人写 `from argus import review_diff` 时，就是从这里找的。

Ch3 结束时 Argus 具备的能力：
  * 读取 diff，发现被修改的文件
  * 追踪修改文件的 import 依赖
  * 找到对应的测试文件
  * 加载项目配置文件（linter、类型检查器配置）
  * 在 token 预算内按优先级筛选上下文（Context Triage）
  * 记录每个感知决策（PerceptionTrace，可观测性）
  * 对筛选后的上下文做一次 LLM 审查，输出结构化结果

使用方式：
    from argus import review_diff, gather_review_context, PerceptionTrace
    review, trace = review_diff(diff_text, repo_root="/path/to/repo")
"""

# 从子模块导入，统一暴露在包级别
# 这样使用者只需要 `from argus import xxx`，不需要知道具体在哪个子模块
from .perception import (
    gather_review_context,   # 感知管道：收集并筛选上下文
    FileContext,             # 数据类：一个文件的上下文条目
    PerceptionTrace,         # 数据类：感知过程的统计记录
)
from .core import review_diff  # 主函数：完整 PRA 循环

# __all__ 声明这个包的公开 API
# 当别人写 `from argus import *` 时，只导入这里列出的名字
__all__ = [
    "review_diff",
    "gather_review_context",
    "FileContext",
    "PerceptionTrace",
]
