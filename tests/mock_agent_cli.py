#!/usr/bin/env python3
"""Mock 的 agent CLI（仅用于测试）

用于端到端验证 run.py -> Plan/AskAgentExecutor -> CursorAgentClient 的 CLI 参数传递：
- 必须带 --mode plan / --mode ask
- plan/ask 模式下不允许出现 --force

输出为纯文本（与 CursorAgentConfig.output_format="text" 默认值一致）。
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-p", "--print", dest="prompt", default="")
    parser.add_argument("--mode", dest="mode", default=None)
    parser.add_argument("--model", dest="model", default=None)
    parser.add_argument("--output-format", dest="output_format", default="text")
    parser.add_argument("--stream-partial-output", action="store_true")
    parser.add_argument("--resume", dest="resume", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--help", action="store_true")

    args, _unknown = parser.parse_known_args()

    if args.help:
        print("mock agent cli --help")
        return 0

    mode = args.mode or "agent"

    # 只读约束校验：plan/ask 模式绝不允许 --force
    if mode in ("plan", "ask") and args.force:
        print("ERROR: --force is not allowed in plan/ask mode", file=sys.stderr)
        return 2

    if mode == "plan":
        # 让上层容易断言：包含固定 marker
        print("MOCK_PLAN_OUTPUT: ok")
        if args.prompt:
            print("MOCK_PLAN_PROMPT_PREFIX:", args.prompt[:80].replace("\n", "\\n"))
        return 0

    if mode == "ask":
        print("MOCK_ASK_OUTPUT: ok")
        if args.prompt:
            print("MOCK_ASK_QUESTION_PREFIX:", args.prompt[:80].replace("\n", "\\n"))
        return 0

    # 其他模式仅返回占位输出
    print("MOCK_AGENT_OUTPUT: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

