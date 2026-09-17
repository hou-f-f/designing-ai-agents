# =============================================================================
# argus/cli.py — Ch3 Argus 的命令行入口
#
# 【学习目标】
#   理解如何把 Argus 封装成命令行工具，让用户可以直接在终端里使用。
#   同时理解 PerceptionTrace 输出的实际价值——让感知过程可见。
#
# 【运行方式】
#   cd ch03-perception
#   python -m argus.cli <diff文件路径> [--repo 仓库路径] [--budget 50000] [--verbose]
#
# 【输出两部分】
#   1. Perception trace  — 感知过程统计（看了哪些文件，丢了哪些）
#   2. Review            — 审查结果 JSON
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path

# 从当前包（argus）的 __init__.py 导入 review_diff
# "." 表示当前包，即 argus/__init__.py 里暴露的 review_diff
from . import review_diff


def main(argv: list[str] | None = None) -> int:
    """
    【函数说明】
    命令行主函数，解析参数并运行 Argus。

    参数：argv - 命令行参数列表（None 时从 sys.argv 读取）
    返回：退出码（0=成功，2=参数错误）

    【为什么返回 int 而不是直接 sys.exit？】
    返回退出码让这个函数可以被测试（直接调用检查返回值），
    而不是用 sys.exit 直接杀死进程，更灵活。
    """
    # -------------------------------------------------------------------------
    # 配置命令行参数解析器
    # argparse 是 Python 标准库，自动处理 --help、参数验证等
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus Ch3: PR review agent with perception.",
    )

    # 必填参数：diff 文件路径
    parser.add_argument("diff_file", help="path to a unified diff file")

    # 可选参数：仓库根目录（默认当前目录）
    parser.add_argument(
        "--repo", default=".",
        help="path to the repository root (default: current directory)",
    )

    # 可选参数：token 预算（控制感知层收集多少上下文）
    parser.add_argument(
        "--budget", type=int, default=50_000,
        help="token budget for perception (default: 50,000)",
    )

    # 可选参数：详细日志（会打印感知过程的 INFO 日志）
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="emit perception logger info messages",
    )

    args = parser.parse_args(argv)

    # -------------------------------------------------------------------------
    # --verbose 模式：开启 INFO 级别日志
    # 【说明】
    #   perception.py 里用 logger.info() 记录了每一步的决策，
    #   默认不显示（WARNING 级别），加 --verbose 后会打印出来，
    #   方便调试时看清楚感知层做了什么。
    # -------------------------------------------------------------------------
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # -------------------------------------------------------------------------
    # 读取 diff 文件
    # -------------------------------------------------------------------------
    diff_path = Path(args.diff_file)
    if not diff_path.exists():
        # 文件不存在：打印错误到 stderr（标准错误流），返回错误码 2
        # 【说明】错误信息写 stderr，正常输出写 stdout，是 Unix 惯例
        print(f"error: diff file not found: {diff_path}", file=sys.stderr)
        return 2
    diff = diff_path.read_text()

    # -------------------------------------------------------------------------
    # 运行 Argus：PRA 循环
    # -------------------------------------------------------------------------
    review, trace = review_diff(diff, repo_root=args.repo, budget=args.budget)

    # -------------------------------------------------------------------------
    # 输出感知追踪（让用户看到 Agent 的感知决策）
    # 【这是 Ch3 最有教育意义的输出】
    # 通过 trace 你可以看到：
    #   - Agent 发现了几个相关文件（files_discovered）
    #   - 最终选了几个（files_selected）
    #   - 因超预算丢了几个（files_dropped）
    #   - token 使用率（selectivity）
    #   - 具体丢了哪些文件（dropped_files）
    # -------------------------------------------------------------------------
    print("=== Perception trace ===")
    print(json.dumps({
        "files_discovered":  trace.files_discovered,
        "files_selected":    trace.files_selected,
        "files_dropped":     trace.files_dropped,
        "selectivity":       round(trace.selectivity, 3),
        "tokens_considered": trace.tokens_considered,
        "tokens_selected":   trace.tokens_selected,
        "dropped_files":     trace.dropped_files,
    }, indent=2))

    # 输出审查结果
    print("\n=== Review ===")
    print(json.dumps(review, indent=2))

    return 0  # 成功退出


if __name__ == "__main__":
    # raise SystemExit(main()) 是比 sys.exit(main()) 更 Pythonic 的写法，
    # 两者效果相同：把 main() 的返回值作为进程退出码
    raise SystemExit(main())
