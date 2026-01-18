#!/usr/bin/env python3
"""流式处理端到端测试脚本

模拟流式事件序列，验证：
- AdvancedTerminalRenderer 的打字效果
- StreamEventLogger 的日志聚合
- 输出文件格式正确

使用方法:
  python scripts/test_streaming_e2e.py
  python scripts/test_streaming_e2e.py --mode advanced --typing-delay 0.02
  python scripts/test_streaming_e2e.py --mode logger --output-dir /tmp/stream_logs
  python scripts/test_streaming_e2e.py --mode all --verbose
"""
import argparse
import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cursor.streaming import (
    AdvancedTerminalRenderer,
    DiffInfo,
    ProgressTracker,
    StreamEvent,
    StreamEventLogger,
    StreamEventType,
    TerminalStreamRenderer,
    ToolCallInfo,
    get_diff_stats,
)


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


class StreamingE2ETest:
    """流式处理端到端测试"""

    def __init__(
        self,
        verbose: bool = False,
        output_dir: Optional[str] = None,
        typing_delay: float = 0.0,
    ) -> None:
        """初始化测试

        Args:
            verbose: 是否详细输出
            output_dir: 日志输出目录
            typing_delay: 打字延迟（秒）
        """
        self.verbose = verbose
        self.output_dir = output_dir or "logs/test_streaming_e2e/"
        self.typing_delay = typing_delay
        self.results: list[TestResult] = []

    def generate_event_sequence(self) -> Generator[StreamEvent, None, None]:
        """生成模拟的流式事件序列

        模拟一个典型的 Agent 执行流程：
        1. 系统初始化
        2. 助手消息（增量输出）
        3. 工具调用（读取文件）
        4. 更多助手消息
        5. 差异操作（编辑文件）
        6. 完成结果
        """
        # 1. 系统初始化
        yield StreamEvent(
            type=StreamEventType.SYSTEM_INIT,
            subtype="init",
            model="opus-4.5-thinking",
            data={
                "apiKeySource": "env",
                "cwd": "/test/project",
                "session_id": "test-session-001",
                "permissionMode": "default",
            },
        )

        # 2. 助手消息（模拟增量输出）
        assistant_parts = [
            "正在",
            "分析",
            "项目",
            "结构",
            "...\n",
            "发现",
            " src/ ",
            "目录",
            "包含",
            "主要",
            "代码",
            "。\n",
        ]
        for part in assistant_parts:
            yield StreamEvent(
                type=StreamEventType.ASSISTANT,
                content=part,
            )
            if self.typing_delay > 0:
                time.sleep(self.typing_delay)

        # 3. 工具调用 - 读取文件开始
        yield StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            subtype="started",
            tool_call=ToolCallInfo(
                tool_type="read",
                path="src/main.py",
                args={"path": "src/main.py"},
            ),
        )

        # 模拟读取延迟
        if self.typing_delay > 0:
            time.sleep(self.typing_delay * 5)

        # 4. 工具调用 - 读取文件完成
        yield StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            subtype="completed",
            tool_call=ToolCallInfo(
                tool_type="read",
                path="src/main.py",
                args={"path": "src/main.py"},
                success=True,
                result={"totalLines": 150, "content": "# Main module..."},
            ),
        )

        # 5. 更多助手消息
        analysis_parts = [
            "\n读取",
            "完成",
            "，",
            "开始",
            "修改",
            "代码",
            "...\n",
        ]
        for part in analysis_parts:
            yield StreamEvent(
                type=StreamEventType.ASSISTANT,
                content=part,
            )
            if self.typing_delay > 0:
                time.sleep(self.typing_delay)

        # 6. 差异操作开始
        yield StreamEvent(
            type=StreamEventType.DIFF_STARTED,
            subtype="started",
            tool_call=ToolCallInfo(
                tool_type="str_replace",
                path="src/main.py",
                old_string="def old_function():\n    pass",
                new_string="def new_function():\n    \"\"\"改进后的函数\"\"\"\n    return True",
                is_diff=True,
            ),
            diff_info=DiffInfo(
                path="src/main.py",
                old_string="def old_function():\n    pass",
                new_string="def new_function():\n    \"\"\"改进后的函数\"\"\"\n    return True",
                operation="replace",
            ),
        )

        # 模拟编辑延迟
        if self.typing_delay > 0:
            time.sleep(self.typing_delay * 3)

        # 7. 差异操作完成
        yield StreamEvent(
            type=StreamEventType.DIFF_COMPLETED,
            subtype="completed",
            tool_call=ToolCallInfo(
                tool_type="str_replace",
                path="src/main.py",
                old_string="def old_function():\n    pass",
                new_string="def new_function():\n    \"\"\"改进后的函数\"\"\"\n    return True",
                is_diff=True,
                success=True,
            ),
            diff_info=DiffInfo(
                path="src/main.py",
                old_string="def old_function():\n    pass",
                new_string="def new_function():\n    \"\"\"改进后的函数\"\"\"\n    return True",
                operation="replace",
            ),
        )

        # 8. 完成消息
        final_parts = [
            "\n代码",
            "修改",
            "完成",
            "！",
            "\n任务",
            "执行",
            "成功",
            "。\n",
        ]
        for part in final_parts:
            yield StreamEvent(
                type=StreamEventType.ASSISTANT,
                content=part,
            )
            if self.typing_delay > 0:
                time.sleep(self.typing_delay)

        # 9. 结果事件
        yield StreamEvent(
            type=StreamEventType.RESULT,
            duration_ms=1500,
            data={
                "subtype": "success",
                "is_error": False,
                "session_id": "test-session-001",
            },
        )

    def test_advanced_terminal_renderer(self) -> TestResult:
        """测试 AdvancedTerminalRenderer 的打字效果"""
        test_name = "AdvancedTerminalRenderer 打字效果"
        start_time = time.time()

        try:
            # 使用 StringIO 捕获输出
            output = io.StringIO()
            renderer = AdvancedTerminalRenderer(
                use_color=False,  # 测试时禁用颜色便于验证
                typing_delay=0,   # 测试时无延迟
                word_mode=True,
                show_status_bar=False,
                output=output,
            )

            # 处理事件序列
            renderer.start()
            event_count = 0
            for event in self.generate_event_sequence():
                renderer.render_event(event)
                event_count += 1
            renderer.finish()

            # 获取输出
            result_text = output.getvalue()

            # 验证输出
            checks = [
                ("模型信息", "opus-4.5-thinking" in result_text),
                ("助手消息", "正在分析项目结构" in result_text),
                ("工具调用", "read" in result_text or "📖" in result_text),
                ("差异操作", "str_replace" in result_text or "编辑" in result_text),
                ("完成标记", "完成" in result_text),
            ]

            failed_checks = [name for name, passed in checks if not passed]

            if failed_checks:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message=f"验证失败: {', '.join(failed_checks)}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 验证渲染器状态
            if renderer.char_count == 0:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message="字符计数为 0，渲染可能未执行",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if renderer.tool_count == 0:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message="工具计数为 0，工具事件可能未处理",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if self.verbose:
                print(f"\n[详细] 渲染器状态:")
                print(f"  - 字符数: {renderer.char_count}")
                print(f"  - 工具调用: {renderer.tool_count}")
                print(f"  - 差异操作: {renderer.diff_count}")
                print(f"  - 事件数: {event_count}")
                print(f"\n[输出预览]\n{result_text[:500]}...")

            return TestResult(
                name=test_name,
                passed=True,
                message=f"渲染 {event_count} 个事件，输出 {renderer.char_count} 字符",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"异常: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def test_stream_event_logger(self) -> TestResult:
        """测试 StreamEventLogger 的日志聚合"""
        test_name = "StreamEventLogger 日志聚合"
        start_time = time.time()

        try:
            # 创建临时目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_dir = Path(self.output_dir) / "detail" / timestamp
            raw_dir = Path(self.output_dir) / "raw" / timestamp

            # 创建日志器（启用消息聚合）
            logger = StreamEventLogger(
                agent_id="test-agent",
                agent_role="worker",
                agent_name="test-worker",
                console=False,  # 测试时禁用控制台
                detail_dir=str(detail_dir),
                raw_dir=str(raw_dir),
                aggregate_assistant_messages=True,
            )

            # 处理事件序列
            event_count = 0
            for event in self.generate_event_sequence():
                # 模拟 raw 日志写入
                raw_line = json.dumps({
                    "type": event.type.value,
                    "content": event.content,
                    "timestamp": time.time(),
                })
                logger.handle_raw_line(raw_line)
                logger.handle_event(event)
                event_count += 1

            # 关闭日志器
            logger.close()

            # 验证文件生成
            raw_files = list(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
            detail_files = list(detail_dir.glob("*.log")) if detail_dir.exists() else []

            if len(raw_files) == 0:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message="未生成 raw 日志文件",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if len(detail_files) == 0:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message="未生成 detail 日志文件",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 验证 raw 日志格式
            raw_content = raw_files[0].read_text(encoding="utf-8")
            raw_lines = raw_content.strip().split("\n")

            for line in raw_lines[:3]:  # 检查前 3 行
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    return TestResult(
                        name=test_name,
                        passed=False,
                        message="raw 日志不是有效的 NDJSON 格式",
                        duration_ms=(time.time() - start_time) * 1000,
                    )

            # 验证 detail 日志内容
            detail_content = detail_files[0].read_text(encoding="utf-8")

            # 验证消息聚合：ASSISTANT 消息应该被聚合
            # 在聚合模式下，不应该有多个独立的短 ASSISTANT 片段
            # 而应该有完整的句子
            if "正在分析项目结构" not in detail_content:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message="ASSISTANT 消息聚合失败",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 验证其他事件类型记录
            checks = [
                ("角色前缀", "worker:test-agent" in detail_content),
                ("模型信息", "opus-4.5-thinking" in detail_content or "初始化" in detail_content),
                ("工具事件", "工具" in detail_content),
            ]

            failed_checks = [name for name, passed in checks if not passed]
            if failed_checks:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message=f"日志内容验证失败: {', '.join(failed_checks)}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if self.verbose:
                print(f"\n[详细] 日志文件:")
                print(f"  - raw 文件: {raw_files[0]}")
                print(f"  - raw 行数: {len(raw_lines)}")
                print(f"  - detail 文件: {detail_files[0]}")
                print(f"\n[detail 日志预览]\n{detail_content[:500]}...")

            return TestResult(
                name=test_name,
                passed=True,
                message=f"生成 {len(raw_lines)} 行 raw 日志，detail 日志正常",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"异常: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def test_progress_tracker(self) -> TestResult:
        """测试 ProgressTracker 状态跟踪"""
        test_name = "ProgressTracker 状态跟踪"
        start_time = time.time()

        try:
            # 创建 ProgressTracker
            tracker = ProgressTracker(
                verbose=False,
                show_diff=True,
                renderer=TerminalStreamRenderer(verbose=False),
            )

            # 处理事件序列
            event_count = 0
            for event in self.generate_event_sequence():
                tracker.on_event(event)
                event_count += 1

            # 验证跟踪状态
            checks = [
                ("模型记录", tracker.model == "opus-4.5-thinking"),
                ("事件计数", len(tracker.events) == event_count),
                ("工具计数", tracker.tool_count >= 1),
                ("差异计数", tracker.diff_count >= 1),
                ("完成状态", tracker.is_complete is True),
                ("耗时记录", tracker.duration_ms > 0),
                ("文本累积", len(tracker.accumulated_text) > 0),
            ]

            failed_checks = [name for name, passed in checks if not passed]
            if failed_checks:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message=f"状态验证失败: {', '.join(failed_checks)}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 验证摘要
            summary = tracker.get_summary()
            if summary["total_events"] != event_count:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message=f"摘要事件数不匹配: {summary['total_events']} vs {event_count}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if self.verbose:
                print(f"\n[详细] ProgressTracker 状态:")
                print(f"  - 模型: {tracker.model}")
                print(f"  - 事件数: {len(tracker.events)}")
                print(f"  - 工具调用: {tracker.tool_count}")
                print(f"  - 差异操作: {tracker.diff_count}")
                print(f"  - 文本长度: {len(tracker.accumulated_text)}")
                print(f"  - 读取文件: {tracker.files_read}")
                print(f"  - 编辑文件: {tracker.files_edited}")

            return TestResult(
                name=test_name,
                passed=True,
                message=f"跟踪 {event_count} 事件，{tracker.tool_count} 工具，{tracker.diff_count} 差异",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"异常: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def test_diff_formatting(self) -> TestResult:
        """测试差异格式化"""
        test_name = "差异格式化"
        start_time = time.time()

        try:
            old_string = "def old_function():\n    pass"
            new_string = "def new_function():\n    \"\"\"改进后的函数\"\"\"\n    return True"

            # 测试差异统计
            stats = get_diff_stats(old_string, new_string)

            checks = [
                ("旧行数", stats["old_lines"] == 2),
                ("新行数", stats["new_lines"] == 3),
                ("有插入", stats["insertions"] > 0),
                ("有删除", stats["deletions"] > 0),
                ("相似度", 0 <= stats["similarity"] <= 1),
            ]

            failed_checks = [name for name, passed in checks if not passed]
            if failed_checks:
                return TestResult(
                    name=test_name,
                    passed=False,
                    message=f"差异统计验证失败: {', '.join(failed_checks)}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if self.verbose:
                print(f"\n[详细] 差异统计:")
                print(f"  - 旧行数: {stats['old_lines']}")
                print(f"  - 新行数: {stats['new_lines']}")
                print(f"  - 插入: {stats['insertions']}")
                print(f"  - 删除: {stats['deletions']}")
                print(f"  - 相似度: {stats['similarity']:.2%}")

            return TestResult(
                name=test_name,
                passed=True,
                message=f"+{stats['insertions']} -{stats['deletions']} 行",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"异常: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def test_typing_effect_demo(self) -> TestResult:
        """演示打字效果（仅在有延迟时运行）"""
        test_name = "打字效果演示"
        start_time = time.time()

        if self.typing_delay <= 0:
            return TestResult(
                name=test_name,
                passed=True,
                message="跳过（无打字延迟）",
                duration_ms=0,
            )

        try:
            print("\n" + "=" * 50)
            print("打字效果演示")
            print("=" * 50)

            renderer = AdvancedTerminalRenderer(
                use_color=True,
                typing_delay=self.typing_delay,
                word_mode=True,
                show_status_bar=True,
                output=sys.stdout,
            )

            # 处理事件序列（带打字效果）
            renderer.start()
            for event in self.generate_event_sequence():
                renderer.render_event(event)
            renderer.finish()
            renderer.print_summary()

            print("=" * 50)

            return TestResult(
                name=test_name,
                passed=True,
                message=f"演示完成，延迟 {self.typing_delay}s",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"异常: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def run_all_tests(self) -> None:
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("流式处理端到端测试")
        print("=" * 60)

        tests = [
            self.test_advanced_terminal_renderer,
            self.test_stream_event_logger,
            self.test_progress_tracker,
            self.test_diff_formatting,
            self.test_typing_effect_demo,
        ]

        for test_func in tests:
            result = test_func()
            self.results.append(result)

            status = "✓" if result.passed else "✗"
            print(f"\n{status} {result.name}")
            print(f"  {result.message}")
            if result.duration_ms > 0:
                print(f"  耗时: {result.duration_ms:.1f}ms")

        # 打印总结
        self._print_summary()

    def run_single_test(self, mode: str) -> None:
        """运行单个测试模式

        Args:
            mode: 测试模式 (advanced, logger, tracker, diff, demo)
        """
        mode_map = {
            "advanced": self.test_advanced_terminal_renderer,
            "logger": self.test_stream_event_logger,
            "tracker": self.test_progress_tracker,
            "diff": self.test_diff_formatting,
            "demo": self.test_typing_effect_demo,
        }

        if mode not in mode_map:
            print(f"未知的测试模式: {mode}")
            print(f"可用模式: {', '.join(mode_map.keys())}")
            return

        print(f"\n运行测试: {mode}")
        result = mode_map[mode]()
        self.results.append(result)

        status = "✓" if result.passed else "✗"
        print(f"\n{status} {result.name}")
        print(f"  {result.message}")
        if result.duration_ms > 0:
            print(f"  耗时: {result.duration_ms:.1f}ms")

        self._print_summary()

    def _print_summary(self) -> None:
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration_ms for r in self.results)

        print(f"\n通过: {passed}/{len(self.results)}")
        if failed > 0:
            print(f"失败: {failed}")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        print(f"总耗时: {total_time:.1f}ms")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="流式处理端到端测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式:
  all       运行所有测试（默认）
  advanced  测试 AdvancedTerminalRenderer
  logger    测试 StreamEventLogger
  tracker   测试 ProgressTracker
  diff      测试差异格式化
  demo      打字效果演示

示例:
  python scripts/test_streaming_e2e.py
  python scripts/test_streaming_e2e.py --mode advanced
  python scripts/test_streaming_e2e.py --mode demo --typing-delay 0.03
  python scripts/test_streaming_e2e.py --verbose --output-dir /tmp/logs
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "advanced", "logger", "tracker", "diff", "demo"],
        help="测试模式 (默认: all)",
    )

    parser.add_argument(
        "--typing-delay",
        type=float,
        default=0.0,
        help="打字延迟秒数 (默认: 0，用于 demo 模式)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/test_streaming_e2e/",
        help="日志输出目录 (默认: logs/test_streaming_e2e/)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    tester = StreamingE2ETest(
        verbose=args.verbose,
        output_dir=args.output_dir,
        typing_delay=args.typing_delay,
    )

    if args.mode == "all":
        tester.run_all_tests()
    else:
        tester.run_single_test(args.mode)

    # 根据测试结果设置退出码
    failed = sum(1 for r in tester.results if not r.passed)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
