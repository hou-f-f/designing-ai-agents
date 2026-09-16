# =============================================================================
# argus/core.py — Argus 代码审查 Agent 的最简实现（第2章核心文件）
#
# 【学习目标】
#   理解 PRA 循环（感知-推理-行动）的最小完整实现。
#   这个文件只有一个函数，走一次 PRA，是全书最简单的 Agent 形态。
#
# 【PRA 循环示意】
#   感知(Perception)  →  推理(Reasoning)  →  行动(Action)
#         ↑                                        |
#         └──────────── 结果反馈（下一轮）─────────┘
#
# 【运行方式】
#   配置好 .env 后直接运行：
#   python ch02-architecture/argus/core.py
# =============================================================================

import json
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 自动加载 .env 配置文件
# 【说明】
#   Python 不会自动读取 .env 文件，这里手动解析并写入环境变量。
#   Path(__file__) 是当前文件的路径，.parent.parent.parent 向上走三级，
#   到达项目根目录（designing-ai-agents/），找到 .env。
#   os.environ.setdefault 表示"如果这个变量还没设置，才设置它"，
#   不会覆盖已有的系统环境变量。
# -----------------------------------------------------------------------------
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.split("#")[0].strip()  # 去掉行尾注释（如 KEY=value # 备注）
            os.environ.setdefault(_k.strip(), _v)


def review_diff(diff: str, context: str = "") -> dict:
    """
    【函数说明】
    对一段代码变更（diff）做一次完整的 PRA 循环，返回审查结果。

    参数：
        diff    - 代码变更内容，git diff 格式的字符串
        context - 可选的额外上下文（如相关文件内容）

    返回：
        dict，包含两个字段：
          - summary:  对本次变更的整体评价
          - comments: 具体问题列表，每项包含 file/line/severity/message

    【支持自定义 endpoint】
    在 .env 里配置以下变量，可以使用 yoka 等兼容 Anthropic 格式的接口：
      ANTHROPIC_BASE_URL=https://aicoding.dobest.com
      ANTHROPIC_API_KEY=sk-...
      ARGUS_MODEL=claude-sonnet-4-6@default   （不填则使用此默认值）
    """

    # 【说明】把 import 放在函数内部（懒加载），好处是：
    # 即使没有安装 anthropic 包，也可以 import 这个模块而不报错，
    # 只有真正调用函数时才会触发导入。
    import anthropic

    # ---------------------------------------------------------------------
    # 读取配置：从环境变量获取 base_url 和模型名
    # 【说明】
    #   os.environ.get("KEY") — 读取环境变量，不存在返回 None
    #   os.environ.get("KEY", "默认值") — 不存在时返回默认值
    # ---------------------------------------------------------------------
    base_url = os.environ.get("ANTHROPIC_BASE_URL")   # 自定义接口地址，None 表示用官方
    model    = os.environ.get("ARGUS_MODEL", "claude-sonnet-4-6@default")  # 使用的模型

    # 创建 Anthropic 客户端
    # 【说明】**{...} 是 Python 的解包语法，相当于把字典里的键值对作为参数传入。
    # 如果 base_url 有值，就传入 base_url=xxx；如果没有，就不传（用官方默认）。
    client = anthropic.Anthropic(
        **({"base_url": base_url} if base_url else {})
    )

    # =========================================================================
    # P — PERCEPTION（感知）
    # 【说明】
    #   感知 = 组装 Agent 能"看到"的所有信息。
    #   这里有两部分：
    #     1. system prompt：告诉 LLM 它的角色和输出格式
    #     2. user_msg：本次任务的具体内容（diff + 可选上下文）
    #
    #   为什么要求输出 JSON？
    #   结构化输出便于后续程序解析，而不是一段自由文本。
    # =========================================================================
    system = """You are Argus, an expert code reviewer.
Respond with JSON: {"summary": "...", "comments": [
  {"file": "...", "line": ..., "severity": "...", "message": "..."}
]}"""
    # severity（严重程度）通常有：info / warning / error / critical

    # 把 diff 内容拼成 Markdown 格式的消息
    user_msg = f"## Diff:\n```diff\n{diff}\n```"
    if context:
        # 如果调用时传入了额外上下文，追加在消息末尾
        user_msg += f"\n\n## Context:\n{context}"

    # =========================================================================
    # R — REASONING（推理）
    # 【说明】
    #   把感知阶段组装好的内容发给 LLM，让它"思考"并给出判断。
    #   这里用的是 Anthropic Messages API，参数说明：
    #     model      - 使用的模型名称
    #     max_tokens - 最多生成多少 token（防止输出过长）
    #     system     - 系统提示，定义 LLM 的角色和行为
    #     messages   - 对话历史，这里只有一轮用户消息
    #
    #   注意：这是同步调用，会阻塞等待 LLM 返回。
    # =========================================================================
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    # =========================================================================
    # A — ACTION（行动）
    # 【说明】
    #   行动 = 处理 LLM 的输出，产生结果。
    #   这里的"行动"比较简单：解析 JSON 字符串，返回结构化数据。
    #
    #   为什么需要处理 ```json ... ``` 格式？
    #   LLM 有时会把 JSON 包在代码块里返回，需要手动提取。
    # =========================================================================
    text = response.content[0].text  # 取 LLM 返回的文本内容

    # 如果 LLM 把 JSON 包在了 ```json ... ``` 代码块里，提取出来
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]

    # 把 JSON 字符串解析成 Python 字典并返回
    return json.loads(text)


# =============================================================================
# 入口：直接运行此文件时执行演示
# 【说明】
#   if __name__ == "__main__" 是 Python 的惯用写法：
#   只有直接运行这个文件时才执行，作为模块被 import 时不执行。
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 演示场景：这段 diff 里藏了一个 SQL 注入漏洞
    #
    # 【什么是 SQL 注入？】
    #   原来的写法用了"参数化查询"（? 占位符），用户输入的内容会被当作
    #   纯数据处理，无法改变 SQL 语句结构，是安全的。
    #
    #   改成的写法用了 f-string 直接拼接，如果用户输入 ' OR '1'='1，
    #   SQL 就变成：SELECT * FROM users WHERE name = '' OR '1'='1'
    #   可以绕过认证，这是严重的安全漏洞。
    #
    #   Argus 应该识别出这个问题并输出 severity: "critical"。
    # -------------------------------------------------------------------------
    SAMPLE_DIFF = """--- a/app.py
+++ b/app.py
@@ -41,3 +41,3 @@ def get_user(db, username):
     cursor = db.cursor()
-    cursor.execute("SELECT * FROM users WHERE name = ?", (username,))
+    cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
     return cursor.fetchone()
"""
    # 调用审查函数，传入 diff
    review = review_diff(SAMPLE_DIFF)

    # 格式化输出 JSON，indent=2 表示缩进2个空格，方便阅读
    print(json.dumps(review, indent=2, ensure_ascii=False))
