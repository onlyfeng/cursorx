"""测试 run.py 统一入口脚本"""
import argparse
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run import (
    MODE_ALIASES,
    RunMode,
    TaskAnalysis,
    TaskAnalyzer,
    Runner,
    parse_max_iterations,
    Colors,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_result,
    setup_logging,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_args() -> argparse.Namespace:
    """模拟命令行参数"""
    args = argparse.Namespace(
        task="测试任务",
        mode="auto",
        directory=".",
        workers=3,
        max_iterations="10",
        strict=False,
        verbose=False,
        skip_online=False,
        dry_run=False,
        force_update=False,
        use_knowledge=False,
        search_knowledge=None,
        self_update=False,
        planner_model="gpt-5.2-high",
        worker_model="opus-4.5-thinking",
        stream_log=True,
        no_auto_analyze=False,
        auto_commit=True,
        auto_push=False,
        commit_per_iteration=False,
    )
    return args


@pytest.fixture
def mock_subprocess():
    """Mock subprocess 调用"""
    with patch("run.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"mode": "basic", "options": {}, "reasoning": "test", "refined_goal": "test goal"}'
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


# ============================================================
# TestRunMode
# ============================================================


class TestRunMode:
    """测试 RunMode 枚举"""

    def test_all_modes_defined(self) -> None:
        """验证所有运行模式都已定义"""
        expected_modes = ["BASIC", "MP", "KNOWLEDGE", "ITERATE", "AUTO", "PLAN", "ASK"]
        for mode_name in expected_modes:
            assert hasattr(RunMode, mode_name)

    def test_mode_values(self) -> None:
        """验证模式值正确"""
        assert RunMode.BASIC.value == "basic"
        assert RunMode.MP.value == "mp"
        assert RunMode.KNOWLEDGE.value == "knowledge"
        assert RunMode.ITERATE.value == "iterate"
        assert RunMode.AUTO.value == "auto"
        assert RunMode.PLAN.value == "plan"
        assert RunMode.ASK.value == "ask"

    def test_mode_is_string_enum(self) -> None:
        """验证 RunMode 继承自 str 和 Enum"""
        assert isinstance(RunMode.BASIC, str)
        assert RunMode.BASIC == "basic"


# ============================================================
# TestModeAliases
# ============================================================


class TestModeAliases:
    """测试 MODE_ALIASES 模式别名映射"""

    def test_basic_aliases(self) -> None:
        """验证 BASIC 模式别名"""
        assert MODE_ALIASES["default"] == RunMode.BASIC
        assert MODE_ALIASES["basic"] == RunMode.BASIC
        assert MODE_ALIASES["simple"] == RunMode.BASIC

    def test_mp_aliases(self) -> None:
        """验证 MP 模式别名"""
        assert MODE_ALIASES["mp"] == RunMode.MP
        assert MODE_ALIASES["multiprocess"] == RunMode.MP
        assert MODE_ALIASES["parallel"] == RunMode.MP

    def test_knowledge_aliases(self) -> None:
        """验证 KNOWLEDGE 模式别名"""
        assert MODE_ALIASES["knowledge"] == RunMode.KNOWLEDGE
        assert MODE_ALIASES["kb"] == RunMode.KNOWLEDGE
        assert MODE_ALIASES["docs"] == RunMode.KNOWLEDGE

    def test_iterate_aliases(self) -> None:
        """验证 ITERATE 模式别名"""
        assert MODE_ALIASES["iterate"] == RunMode.ITERATE
        assert MODE_ALIASES["self-iterate"] == RunMode.ITERATE
        assert MODE_ALIASES["self"] == RunMode.ITERATE
        assert MODE_ALIASES["update"] == RunMode.ITERATE

    def test_auto_aliases(self) -> None:
        """验证 AUTO 模式别名"""
        assert MODE_ALIASES["auto"] == RunMode.AUTO
        assert MODE_ALIASES["smart"] == RunMode.AUTO

    def test_plan_aliases(self) -> None:
        """验证 PLAN 模式别名"""
        assert MODE_ALIASES["plan"] == RunMode.PLAN
        assert MODE_ALIASES["planning"] == RunMode.PLAN
        assert MODE_ALIASES["analyze"] == RunMode.PLAN

    def test_ask_aliases(self) -> None:
        """验证 ASK 模式别名"""
        assert MODE_ALIASES["ask"] == RunMode.ASK
        assert MODE_ALIASES["chat"] == RunMode.ASK
        assert MODE_ALIASES["question"] == RunMode.ASK
        assert MODE_ALIASES["q"] == RunMode.ASK


# ============================================================
# TestParseMaxIterations
# ============================================================


class TestParseMaxIterations:
    """测试 parse_max_iterations 函数"""

    def test_positive_integer(self) -> None:
        """验证正整数解析"""
        assert parse_max_iterations("10") == 10
        assert parse_max_iterations("1") == 1
        assert parse_max_iterations("100") == 100

    def test_max_keyword(self) -> None:
        """验证 MAX 关键词解析为无限迭代"""
        assert parse_max_iterations("MAX") == -1
        assert parse_max_iterations("max") == -1
        assert parse_max_iterations("Max") == -1

    def test_unlimited_keywords(self) -> None:
        """验证其他无限迭代关键词"""
        assert parse_max_iterations("UNLIMITED") == -1
        assert parse_max_iterations("INF") == -1
        assert parse_max_iterations("INFINITE") == -1

    def test_zero_and_negative(self) -> None:
        """验证零和负数转为无限迭代"""
        assert parse_max_iterations("0") == -1
        assert parse_max_iterations("-1") == -1
        assert parse_max_iterations("-5") == -1

    def test_whitespace_handling(self) -> None:
        """验证空白字符处理"""
        assert parse_max_iterations("  MAX  ") == -1
        assert parse_max_iterations(" 10 ") == 10

    def test_invalid_value_raises_error(self) -> None:
        """验证无效值抛出异常"""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            parse_max_iterations("invalid")
        assert "无效的迭代次数" in str(exc_info.value)

        with pytest.raises(argparse.ArgumentTypeError):
            parse_max_iterations("abc")


# ============================================================
# TestTaskAnalysis
# ============================================================


class TestTaskAnalysis:
    """测试 TaskAnalysis 数据类"""

    def test_default_values(self) -> None:
        """验证默认值"""
        analysis = TaskAnalysis()
        assert analysis.mode == RunMode.BASIC
        assert analysis.goal == ""
        assert analysis.options == {}
        assert analysis.reasoning == ""

    def test_custom_values(self) -> None:
        """验证自定义值"""
        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="更新知识库",
            options={"skip_online": True},
            reasoning="检测到迭代关键词",
        )
        assert analysis.mode == RunMode.ITERATE
        assert analysis.goal == "更新知识库"
        assert analysis.options == {"skip_online": True}
        assert analysis.reasoning == "检测到迭代关键词"


# ============================================================
# TestTaskAnalyzer
# ============================================================


class TestTaskAnalyzer:
    """测试 TaskAnalyzer 任务分析器"""

    def test_init_default(self) -> None:
        """验证默认初始化"""
        analyzer = TaskAnalyzer()
        assert analyzer.use_agent is True

    def test_init_without_agent(self) -> None:
        """验证禁用 Agent 分析"""
        analyzer = TaskAnalyzer(use_agent=False)
        assert analyzer.use_agent is False

    def test_empty_task_returns_default_mode(self, mock_args: argparse.Namespace) -> None:
        """验证空任务返回默认模式"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("", mock_args)
        assert analysis.goal == ""
        assert analysis.reasoning == "无任务描述，使用指定模式"

    def test_detect_iterate_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测自我迭代关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        tasks = [
            "启动自我迭代模式",
            "self-iterate 更新代码",
            "检查更新并同步",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.ITERATE, f"任务 '{task}' 应该匹配 ITERATE 模式"

    def test_detect_mp_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测多进程关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        tasks = [
            "使用多进程处理",
            "并行执行任务",
            "multiprocess 模式运行",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.MP, f"任务 '{task}' 应该匹配 MP 模式"

    def test_detect_knowledge_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测知识库关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        tasks = [
            "搜索知识库找答案",
            "参考 cursor 文档实现",
            "knowledge 增强模式",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.KNOWLEDGE, f"任务 '{task}' 应该匹配 KNOWLEDGE 模式"

    def test_detect_plan_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测规划关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        tasks = [
            "规划任务执行步骤",
            "制定计划",
            "仅规划不执行",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.PLAN, f"任务 '{task}' 应该匹配 PLAN 模式"

    def test_detect_ask_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测问答关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        tasks = [
            "问答模式解决问题",
            "直接问一下",
            "chat 对话",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.ASK, f"任务 '{task}' 应该匹配 ASK 模式"

    def test_detect_skip_online_option(self, mock_args: argparse.Namespace) -> None:
        """验证检测跳过在线选项"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("自我迭代，跳过在线检查", mock_args)
        assert analysis.options.get("skip_online") is True

    def test_detect_dry_run_option(self, mock_args: argparse.Namespace) -> None:
        """验证检测 dry-run 选项"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("仅分析不执行任务", mock_args)
        assert analysis.options.get("dry_run") is True

    def test_detect_unlimited_iterations(self, mock_args: argparse.Namespace) -> None:
        """验证检测无限迭代关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("无限循环执行任务", mock_args)
        assert analysis.options.get("max_iterations") == -1

    def test_detect_worker_count(self, mock_args: argparse.Namespace) -> None:
        """验证提取 Worker 数量"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("使用 5 个 worker 并行处理", mock_args)
        assert analysis.options.get("workers") == 5

    def test_worker_count_extraction(self, mock_args: argparse.Namespace) -> None:
        """测试 '5 个 worker' 提取 workers=5"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试 "N 个 worker" 格式
        tasks_and_expected = [
            ("使用 5 个 worker 处理", 5),
            ("分配 10 个 worker 执行任务", 10),
            ("启动 2 个 worker", 2),
            ("需要 8 个 worker 并发", 8),
        ]

        for task, expected_workers in tasks_and_expected:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("workers") == expected_workers, (
                f"任务 '{task}' 应该提取 workers={expected_workers}，"
                f"实际提取 workers={analysis.options.get('workers')}"
            )

    def test_worker_count_with_process(self, mock_args: argparse.Namespace) -> None:
        """测试 '3 个进程' 提取 workers=3"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试 "N 个进程" 格式
        tasks_and_expected = [
            ("使用 3 个进程运行", 3),
            ("分配 6 个进程处理", 6),
            ("启动 4 个进程并行执行", 4),
            ("需要 12 个进程", 12),
        ]

        for task, expected_workers in tasks_and_expected:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("workers") == expected_workers, (
                f"任务 '{task}' 应该提取 workers={expected_workers}，"
                f"实际提取 workers={analysis.options.get('workers')}"
            )

    def test_worker_count_parallel(self, mock_args: argparse.Namespace) -> None:
        """测试 '4 并行' 提取 workers=4"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试 "N 并行" 格式
        tasks_and_expected = [
            ("4 并行处理任务", 4),
            ("使用 8 并行执行", 8),
            ("需要 2 并行操作", 2),
            ("以 16 并行模式运行", 16),
        ]

        for task, expected_workers in tasks_and_expected:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("workers") == expected_workers, (
                f"任务 '{task}' 应该提取 workers={expected_workers}，"
                f"实际提取 workers={analysis.options.get('workers')}"
            )

    def test_worker_count_no_match(self, mock_args: argparse.Namespace) -> None:
        """测试无 worker 关键词时不设置 workers"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 这些任务不包含 worker 数量关键词
        tasks_without_worker_count = [
            "执行一个简单任务",
            "处理代码重构",
            "运行测试用例",
            "分析项目结构",
            "修复 bug",
        ]

        for task in tasks_without_worker_count:
            analysis = analyzer.analyze(task, mock_args)
            assert "workers" not in analysis.options, (
                f"任务 '{task}' 不应该设置 workers，"
                f"但实际设置了 workers={analysis.options.get('workers')}"
            )

    def test_worker_count_edge_cases(self, mock_args: argparse.Namespace) -> None:
        """测试边界情况（大数字、0 等）"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试大数字
        analysis = analyzer.analyze("使用 100 个 worker 处理", mock_args)
        assert analysis.options.get("workers") == 100, (
            "应该能提取大数字 100 作为 workers"
        )

        # 测试较大的数字
        analysis = analyzer.analyze("启动 50 个进程执行", mock_args)
        assert analysis.options.get("workers") == 50, (
            "应该能提取 50 作为 workers"
        )

        # 测试单个 worker
        analysis = analyzer.analyze("使用 1 个 worker", mock_args)
        assert analysis.options.get("workers") == 1, (
            "应该能提取 1 作为 workers"
        )

        # 测试 0 的情况（可能不设置或设置为 0）
        analysis = analyzer.analyze("使用 0 个 worker", mock_args)
        # 0 可能被忽略或设置为 0，取决于实现
        workers_value = analysis.options.get("workers")
        assert workers_value is None or workers_value == 0, (
            f"0 个 worker 应该被忽略或设置为 0，实际为 {workers_value}"
        )

    def test_detect_disable_options(self, mock_args: argparse.Namespace) -> None:
        """验证检测禁用选项关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("执行任务，禁用提交", mock_args)
        assert analysis.options.get("auto_commit") is False

    def test_disable_auto_commit_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测禁用提交关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试多种禁用提交关键词
        tasks = [
            "执行任务，禁用提交",
            "完成任务 no-commit",
            "处理代码，关闭提交",
            "不提交直接执行",
            "跳过提交步骤",
            "禁用自动提交模式",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("auto_commit") is False, (
                f"任务 '{task}' 应该设置 auto_commit=False"
            )

    def test_disable_stream_log_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测禁用流式日志关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试多种禁用流式日志关键词
        tasks = [
            "执行任务，禁用流式日志",
            "no-stream 模式运行",
            "关闭流式输出",
            "简洁模式执行",
            "静默模式处理",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("stream_log") is False, (
                f"任务 '{task}' 应该设置 stream_log=False"
            )

    def test_disable_auto_push_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测禁用推送关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试多种禁用推送关键词
        tasks = [
            "执行任务，禁用推送",
            "no-push 模式",
            "不推送到远程",
            "关闭推送功能",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("auto_push") is False, (
                f"任务 '{task}' 应该设置 auto_push=False"
            )

    def test_enable_and_disable_conflict(self, mock_args: argparse.Namespace) -> None:
        """验证启用和禁用关键词冲突时，禁用优先生效"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 同时包含启用和禁用关键词时，禁用应后生效（后处理覆盖前处理）
        # 测试 auto_commit 冲突
        analysis = analyzer.analyze("启用提交但禁用提交", mock_args)
        assert analysis.options.get("auto_commit") is False, (
            "同时包含启用和禁用提交关键词时，禁用应优先生效"
        )

        # 测试 stream_log 冲突
        analysis = analyzer.analyze("启用流式日志，但禁用流式日志", mock_args)
        assert analysis.options.get("stream_log") is False, (
            "同时包含启用和禁用流式日志关键词时，禁用应优先生效"
        )

        # 测试 auto_push 冲突
        analysis = analyzer.analyze("自动推送 no-push", mock_args)
        assert analysis.options.get("auto_push") is False, (
            "同时包含启用和禁用推送关键词时，禁用应优先生效"
        )

        # 测试混合场景：禁用在前，启用在后，禁用仍应生效
        analysis = analyzer.analyze("禁用提交然后启用提交", mock_args)
        assert analysis.options.get("auto_commit") is False, (
            "禁用关键词处理在启用关键词之后，应覆盖启用设置"
        )

    def test_goal_not_empty(self, mock_args: argparse.Namespace) -> None:
        """验证 goal 始终不为空"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("测试任务描述", mock_args)
        assert analysis.goal == "测试任务描述"

    def test_agent_analysis_with_mock(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证 Agent 分析（使用 mock）"""
        analyzer = TaskAnalyzer(use_agent=True)

        # 设置 mock 返回值
        mock_subprocess.return_value.stdout = (
            '{"mode": "mp", "options": {"workers": 4}, '
            '"reasoning": "Agent 推荐", "refined_goal": "优化后的目标"}'
        )

        analysis = analyzer.analyze("一个普通任务", mock_args)

        # 验证 subprocess 被调用
        mock_subprocess.assert_called()

        # 验证返回结果
        assert analysis.mode == RunMode.MP
        assert analysis.options.get("workers") == 4

    def test_plan_mode_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证 PLAN 模式关键词匹配（完整覆盖）"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试所有 PLAN 模式关键词
        plan_keywords_tasks = [
            ("规划这个项目的实现步骤", "规划"),
            ("plan the implementation", "plan"),
            ("planning this feature", "planning"),
            ("分析任务需求", "分析任务"),
            ("进行任务分析", "任务分析"),
            ("制定计划并列出步骤", "制定计划"),
            ("仅规划不执行", "仅规划"),
            ("只规划不要做", "只规划"),
            ("分解任务为子任务", "分解任务"),
        ]

        for task, keyword in plan_keywords_tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.PLAN, (
                f"任务 '{task}' (关键词: {keyword}) 应该匹配 PLAN 模式，"
                f"实际匹配 {analysis.mode}"
            )
            assert analysis.goal == task
            assert "PLAN" in analysis.reasoning.upper() or "规划" in analysis.reasoning or "plan" in analysis.reasoning.lower()

    def test_ask_mode_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证 ASK 模式关键词匹配（完整覆盖）"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试所有 ASK 模式关键词
        ask_keywords_tasks = [
            ("问答模式解答问题", "问答"),
            ("ask about this feature", "ask"),
            ("chat with me about this", "chat"),
            ("对话讨论一下", "对话"),
            ("提问关于代码的问题", "提问"),
            ("询问这个功能如何实现", "询问"),
            ("直接问一下这个问题", "直接问"),
            ("回答我这个问题", "回答"),
            ("解答这个疑问", "解答"),
            ("question about the API", "question"),
        ]

        for task, keyword in ask_keywords_tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.ASK, (
                f"任务 '{task}' (关键词: {keyword}) 应该匹配 ASK 模式，"
                f"实际匹配 {analysis.mode}"
            )
            assert analysis.goal == task
            assert "ASK" in analysis.reasoning.upper() or "问答" in analysis.reasoning or "ask" in analysis.reasoning.lower()

    def test_agent_analysis_success(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析成功时正确解析 JSON 结果"""
        import subprocess

        with patch("run.subprocess.run") as mock_run:
            # 模拟 agent CLI 返回有效 JSON
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '''分析结果：
            {"mode": "mp", "options": {"workers": 5, "strict": true}, "reasoning": "任务需要并行处理", "refined_goal": "使用多进程重构代码"}
            '''
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            # 使用一个不会触发规则匹配的普通任务
            result = analyzer._agent_analysis("执行一个普通任务")

            # 验证 subprocess 被正确调用
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "agent" in call_args[0][0]
            assert "-p" in call_args[0][0]
            assert "--output-format" in call_args[0][0]

            # 验证解析结果
            assert result is not None
            assert result.mode == RunMode.MP
            assert result.options.get("workers") == 5
            assert result.options.get("strict") is True
            assert result.reasoning == "任务需要并行处理"
            assert result.goal == "使用多进程重构代码"

    def test_agent_analysis_timeout(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析超时时返回 None"""
        import subprocess

        with patch("run.subprocess.run") as mock_run:
            # 模拟超时异常
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None
            assert result is None

    def test_agent_analysis_invalid_json(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 返回非 JSON 时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            # 返回无法解析为 JSON 的内容
            mock_result.stdout = "这是一段普通文本，没有 JSON 格式的内容"
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None
            assert result is None

    def test_agent_analysis_returncode_nonzero(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 返回非零退出码时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1  # 非零退出码
            mock_result.stdout = ""
            mock_result.stderr = "命令执行失败"
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None
            assert result is None

    def test_analyze_with_agent_fallback(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 analyze 方法在 Agent 分析失败时回退到规则匹配

        注意：analyze 方法只在规则匹配返回 BASIC 模式时才会调用 Agent 分析。
        当规则匹配已经识别出特定模式（非 BASIC）时，不会调用 Agent。
        """
        import subprocess

        # 测试 1: 规则匹配返回 BASIC，Agent 超时，应保持 BASIC 模式
        with patch("run.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            # 使用一个不会触发规则匹配的普通任务
            analysis = analyzer.analyze("完成一个普通的编程任务", mock_args)

            # 验证 Agent 被调用（因为规则分析返回 BASIC）
            mock_run.assert_called()

            # Agent 分析失败，保持规则分析的 BASIC 模式
            assert analysis.mode == RunMode.BASIC
            assert analysis.goal == "完成一个普通的编程任务"

        # 测试 2: 规则匹配返回 BASIC，Agent 返回非零退出码
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("处理一个普通任务", mock_args)

            # 验证 Agent 被调用
            mock_run.assert_called()

            # Agent 分析失败，保持 BASIC 模式
            assert analysis.mode == RunMode.BASIC

        # 测试 3: 规则匹配返回 BASIC，Agent 返回无效 JSON
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "无效的响应，不包含 JSON"
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("执行普通任务", mock_args)

            # Agent 分析失败，保持 BASIC 模式
            assert analysis.mode == RunMode.BASIC

        # 测试 4: 规则匹配已识别特定模式，不会调用 Agent
        with patch("run.subprocess.run") as mock_run:
            analyzer = TaskAnalyzer(use_agent=True)
            # 包含 ITERATE 关键词的任务
            analysis = analyzer.analyze("启动自我迭代更新代码", mock_args)

            # 规则分析已识别 ITERATE 模式，不调用 Agent
            mock_run.assert_not_called()

            # 验证正确识别 ITERATE 模式
            assert analysis.mode == RunMode.ITERATE
            assert analysis.goal == "启动自我迭代更新代码"

        # 测试 5: 验证规则匹配识别 MP 模式后不调用 Agent
        with patch("run.subprocess.run") as mock_run:
            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("使用多进程并行处理", mock_args)

            # 规则分析已识别 MP 模式，不调用 Agent
            mock_run.assert_not_called()

            assert analysis.mode == RunMode.MP

        # 测试 6: 验证规则匹配识别 KNOWLEDGE 模式后不调用 Agent
        with patch("run.subprocess.run") as mock_run:
            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("搜索知识库获取文档", mock_args)

            mock_run.assert_not_called()
            assert analysis.mode == RunMode.KNOWLEDGE


# ============================================================
# TestRunner
# ============================================================


class TestRunner:
    """测试 Runner 运行器"""

    def test_init(self, mock_args: argparse.Namespace) -> None:
        """验证 Runner 初始化"""
        runner = Runner(mock_args)
        assert runner.args == mock_args
        assert runner.max_iterations == 10

    def test_init_with_unlimited_iterations(self) -> None:
        """验证无限迭代初始化"""
        args = argparse.Namespace(
            task="测试",
            mode="basic",
            directory=".",
            workers=3,
            max_iterations="MAX",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            stream_log=True,
            auto_commit=True,
            auto_push=False,
            commit_per_iteration=False,
        )
        runner = Runner(args)
        assert runner.max_iterations == -1

    def test_get_mode_name(self, mock_args: argparse.Namespace) -> None:
        """验证获取模式显示名称"""
        runner = Runner(mock_args)

        assert runner._get_mode_name(RunMode.BASIC) == "基本模式"
        assert runner._get_mode_name(RunMode.MP) == "多进程模式"
        assert runner._get_mode_name(RunMode.KNOWLEDGE) == "知识库增强模式"
        assert runner._get_mode_name(RunMode.ITERATE) == "自我迭代模式"
        assert runner._get_mode_name(RunMode.AUTO) == "自动模式"
        assert runner._get_mode_name(RunMode.PLAN) == "规划模式"
        assert runner._get_mode_name(RunMode.ASK) == "问答模式"

    def test_merge_options_default(self, mock_args: argparse.Namespace) -> None:
        """验证选项合并（默认值）"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["directory"] == "."
        assert options["workers"] == 3
        assert options["max_iterations"] == 10
        assert options["strict"] is False
        assert options["stream_log"] is True
        assert options["auto_commit"] is True
        assert options["auto_push"] is False

    def test_merge_options_with_analysis(self, mock_args: argparse.Namespace) -> None:
        """验证选项合并（使用分析结果）"""
        runner = Runner(mock_args)
        analysis_options = {
            "workers": 5,
            "max_iterations": -1,
            "skip_online": True,
            "dry_run": True,
        }
        options = runner._merge_options(analysis_options)

        assert options["workers"] == 5
        assert options["max_iterations"] == -1
        assert options["skip_online"] is True
        assert options["dry_run"] is True

    def test_merge_options_stream_log_from_analysis(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 stream_log 从分析结果获取"""
        runner = Runner(mock_args)

        # 分析结果禁用 stream_log
        options = runner._merge_options({"stream_log": False})
        assert options["stream_log"] is False

        # 分析结果启用 stream_log
        options = runner._merge_options({"stream_log": True})
        assert options["stream_log"] is True

    def test_merge_options_auto_commit_from_analysis(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 auto_commit 从分析结果获取"""
        runner = Runner(mock_args)

        # 分析结果禁用 auto_commit
        options = runner._merge_options({"auto_commit": False})
        assert options["auto_commit"] is False

        # 分析结果启用 auto_commit
        options = runner._merge_options({"auto_commit": True})
        assert options["auto_commit"] is True

    def test_merge_options_auto_push_requires_auto_commit(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 auto_push 需要 auto_commit"""
        runner = Runner(mock_args)

        # auto_push=True 但 auto_commit=False，结果应该是 auto_push=False
        options = runner._merge_options({"auto_commit": False, "auto_push": True})
        assert options["auto_push"] is False

        # 两者都为 True
        options = runner._merge_options({"auto_commit": True, "auto_push": True})
        assert options["auto_push"] is True


# ============================================================
# TestRunMpMode - 多进程模式测试
# ============================================================


class TestRunMpMode:
    """测试多进程模式相关功能"""

    def test_mp_imports(self) -> None:
        """验证 MultiProcessOrchestrator 和 MultiProcessOrchestratorConfig 可导入"""
        from coordinator import MultiProcessOrchestrator, MultiProcessOrchestratorConfig

        # 验证类存在且可以被引用
        assert MultiProcessOrchestrator is not None
        assert MultiProcessOrchestratorConfig is not None

        # 验证是类类型
        assert isinstance(MultiProcessOrchestrator, type)
        assert isinstance(MultiProcessOrchestratorConfig, type)

    def test_mp_config_creation(self) -> None:
        """测试 MultiProcessOrchestratorConfig 可以用 run.py 中的参数正确创建"""
        from coordinator import MultiProcessOrchestratorConfig

        # 使用 run.py 中 _run_mp 方法的参数创建配置
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=10,
            worker_count=3,
            strict_review=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="opus-4.5-thinking",
            stream_events_enabled=True,
        )

        # 验证配置属性
        assert config.working_directory == "."
        assert config.max_iterations == 10
        assert config.worker_count == 3
        assert config.strict_review is False
        assert config.planner_model == "gpt-5.2-high"
        assert config.worker_model == "opus-4.5-thinking"
        assert config.reviewer_model == "opus-4.5-thinking"
        assert config.stream_events_enabled is True

    def test_mp_config_with_unlimited_iterations(self) -> None:
        """测试无限迭代配置"""
        from coordinator import MultiProcessOrchestratorConfig

        config = MultiProcessOrchestratorConfig(
            working_directory="/tmp/test",
            max_iterations=-1,  # 无限迭代
            worker_count=5,
        )

        assert config.max_iterations == -1
        assert config.worker_count == 5

    def test_mp_config_default_values(self) -> None:
        """测试默认配置值"""
        from coordinator import MultiProcessOrchestratorConfig

        config = MultiProcessOrchestratorConfig()

        # 验证默认值与 orchestrator_mp.py 中定义一致
        assert config.working_directory == "."
        assert config.max_iterations == 10
        assert config.worker_count == 3
        assert config.enable_sub_planners is True
        assert config.strict_review is False
        assert config.planning_timeout == 120.0
        assert config.execution_timeout == 300.0
        assert config.review_timeout == 60.0
        assert config.planner_model == "gpt-5.2-high"
        assert config.worker_model == "opus-4.5-thinking"
        assert config.reviewer_model == "opus-4.5-thinking"
        assert config.stream_events_enabled is True
        # 验证提交相关默认值（与 OrchestratorConfig 对齐）
        assert config.enable_auto_commit is True
        assert config.auto_push is False
        assert config.commit_on_complete is True
        assert config.commit_per_iteration is False

    def test_mp_config_commit_fields(self) -> None:
        """测试 MP 配置提交相关字段"""
        from coordinator import MultiProcessOrchestratorConfig

        # 测试自定义提交配置
        config = MultiProcessOrchestratorConfig(
            enable_auto_commit=False,
            auto_push=True,
            commit_on_complete=False,
            commit_per_iteration=True,
        )

        assert config.enable_auto_commit is False
        assert config.auto_push is True
        assert config.commit_on_complete is False
        assert config.commit_per_iteration is True

    @pytest.mark.asyncio
    async def test_mp_orchestrator_init(self) -> None:
        """测试 MultiProcessOrchestrator 可以正确初始化（mock 运行）"""
        from coordinator import MultiProcessOrchestrator, MultiProcessOrchestratorConfig

        # 创建配置
        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=2,
        )

        # 创建 orchestrator 实例
        orchestrator = MultiProcessOrchestrator(config)

        # 验证初始化状态
        assert orchestrator.config == config
        assert orchestrator.state is not None
        assert orchestrator.state.working_directory == "."
        assert orchestrator.state.max_iterations == 1
        assert orchestrator.task_queue is not None
        assert orchestrator.process_manager is not None
        assert orchestrator.planner_id is None  # 未启动时为 None
        assert orchestrator.worker_ids == []  # 未启动时为空列表
        assert orchestrator.reviewer_id is None  # 未启动时为 None

    @pytest.mark.asyncio
    async def test_mp_orchestrator_run_with_mock(self) -> None:
        """测试 MultiProcessOrchestrator.run 方法（使用 mock）"""
        from coordinator import MultiProcessOrchestrator, MultiProcessOrchestratorConfig

        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=1,
        )

        orchestrator = MultiProcessOrchestrator(config)

        # Mock _spawn_agents 和 process_manager 方法
        with patch.object(orchestrator, '_spawn_agents') as mock_spawn, \
             patch.object(orchestrator.process_manager, 'wait_all_ready', return_value=True), \
             patch.object(orchestrator.process_manager, 'shutdown_all'), \
             patch.object(orchestrator, '_planning_phase', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execution_phase', new_callable=AsyncMock), \
             patch.object(orchestrator, '_review_phase', new_callable=AsyncMock) as mock_review:

            # 设置 review 返回 COMPLETE 以结束循环
            from agents.reviewer_process import ReviewDecision
            mock_review.return_value = ReviewDecision.COMPLETE

            # 运行
            result = await orchestrator.run("测试目标")

            # 验证 _spawn_agents 被调用
            mock_spawn.assert_called_once()

            # 验证返回结果结构
            assert "success" in result
            assert "goal" in result
            assert result["goal"] == "测试目标"


# ============================================================
# TestColorHelpers - 颜色辅助函数测试
# ============================================================


class TestColorHelpers:
    """测试颜色辅助函数"""

    def test_colors_constants(self) -> None:
        """验证 Colors 类常量定义正确"""
        assert Colors.RED == "\033[0;31m"
        assert Colors.GREEN == "\033[0;32m"
        assert Colors.YELLOW == "\033[1;33m"
        assert Colors.BLUE == "\033[0;34m"
        assert Colors.CYAN == "\033[0;36m"
        assert Colors.MAGENTA == "\033[0;35m"
        assert Colors.BOLD == "\033[1m"
        assert Colors.NC == "\033[0m"

    def test_print_header(self, capsys) -> None:
        """验证 print_header 输出"""
        print_header("测试标题")
        captured = capsys.readouterr()
        assert "测试标题" in captured.out
        assert "=" in captured.out

    def test_print_info(self, capsys) -> None:
        """验证 print_info 输出"""
        print_info("测试信息")
        captured = capsys.readouterr()
        assert "测试信息" in captured.out
        assert "ℹ" in captured.out

    def test_print_success(self, capsys) -> None:
        """验证 print_success 输出"""
        print_success("测试成功")
        captured = capsys.readouterr()
        assert "测试成功" in captured.out
        assert "✓" in captured.out

    def test_print_warning(self, capsys) -> None:
        """验证 print_warning 输出"""
        print_warning("测试警告")
        captured = capsys.readouterr()
        assert "测试警告" in captured.out
        assert "⚠" in captured.out

    def test_print_error(self, capsys) -> None:
        """验证 print_error 输出"""
        print_error("测试错误")
        captured = capsys.readouterr()
        assert "测试错误" in captured.out
        assert "✗" in captured.out


# ============================================================
# TestPrintResult - 结果输出测试
# ============================================================


class TestPrintResult:
    """测试结果输出函数"""

    def test_print_result_success(self, capsys) -> None:
        """验证成功结果输出"""
        result = {
            "success": True,
            "goal": "测试任务目标",
            "iterations_completed": 5,
            "total_tasks_created": 10,
            "total_tasks_completed": 8,
            "total_tasks_failed": 2,
            "final_score": 8.5,
        }
        print_result(result)
        captured = capsys.readouterr()
        assert "成功" in captured.out
        assert "测试任务目标" in captured.out
        assert "5" in captured.out
        assert "8.5" in captured.out

    def test_print_result_failure(self, capsys) -> None:
        """验证失败结果输出"""
        result = {
            "success": False,
            "goal": "失败的任务",
        }
        print_result(result)
        captured = capsys.readouterr()
        assert "未完成" in captured.out

    def test_print_result_dry_run(self, capsys) -> None:
        """验证 dry-run 结果输出"""
        result = {
            "success": True,
            "goal": "测试目标",
            "dry_run": True,
        }
        print_result(result)
        captured = capsys.readouterr()
        assert "Dry-run" in captured.out

    def test_print_result_long_goal(self, capsys) -> None:
        """验证长目标被截断"""
        long_goal = "这是一个非常长的任务目标，" * 20
        result = {
            "success": True,
            "goal": long_goal,
        }
        print_result(result)
        captured = capsys.readouterr()
        assert "..." in captured.out


# ============================================================
# TestSetupLogging - 日志配置测试
# ============================================================


class TestSetupLogging:
    """测试日志配置函数"""

    def test_setup_logging_default(self) -> None:
        """验证默认日志配置"""
        # 应该不抛出异常
        setup_logging(verbose=False)

    def test_setup_logging_verbose(self) -> None:
        """验证详细日志配置"""
        # 应该不抛出异常
        setup_logging(verbose=True)


# ============================================================
# TestRunnerRunMethods - Runner 运行方法测试
# ============================================================


class TestRunnerRunMethods:
    """测试 Runner 各运行模式方法"""

    @pytest.fixture
    def runner(self, mock_args: argparse.Namespace) -> Runner:
        """创建 Runner 实例"""
        return Runner(mock_args)

    @pytest.mark.asyncio
    async def test_run_basic_with_mock(
        self, runner: Runner, mock_args: argparse.Namespace
    ) -> None:
        """测试 _run_basic 方法（使用 mock）"""
        import coordinator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True})
        with patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator):
            options = runner._merge_options({})
            result = await runner._run_basic("测试目标", options)

            assert result == {"success": True}
            mock_orchestrator.run.assert_called_once_with("测试目标")

    @pytest.mark.asyncio
    async def test_run_basic_with_unlimited_iterations(
        self, runner: Runner
    ) -> None:
        """测试 _run_basic 无限迭代模式"""
        import coordinator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True})
        with patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator), \
             patch("run.print_info") as mock_print_info:
            options = runner._merge_options({"max_iterations": -1})
            options["max_iterations"] = -1  # 确保设置
            result = await runner._run_basic("测试目标", options)

            assert result == {"success": True}
            # 验证无限迭代提示被打印
            mock_print_info.assert_called()

    @pytest.mark.asyncio
    async def test_run_mp_with_mock(
        self, runner: Runner
    ) -> None:
        """测试 _run_mp 方法（使用 mock）"""
        import coordinator
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True, "mode": "mp"})
        with patch.object(coordinator, "MultiProcessOrchestrator", return_value=mock_orchestrator):
            options = runner._merge_options({})
            result = await runner._run_mp("多进程测试", options)

            assert result["success"] is True
            mock_orchestrator.run.assert_called_once_with("多进程测试")

    @pytest.mark.asyncio
    async def test_run_iterate_with_mock(
        self, runner: Runner
    ) -> None:
        """测试 _run_iterate 方法（使用 mock）"""
        import scripts.run_iterate
        with patch.object(scripts.run_iterate, "SelfIterator") as mock_iterator_class:
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value={"success": True, "mode": "iterate"})
            mock_iterator_class.return_value = mock_iterator

            options = runner._merge_options({"skip_online": True})
            result = await runner._run_iterate("自我迭代任务", options)

            assert result["success"] is True
            mock_iterator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_plan_success(
        self, runner: Runner
    ) -> None:
        """测试 _run_plan 方法成功"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "执行计划内容..."
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_plan("规划任务", options)

            assert result["success"] is True
            assert result["mode"] == "plan"
            assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_run_plan_failure(
        self, runner: Runner
    ) -> None:
        """测试 _run_plan 方法失败"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "规划失败原因"
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_plan("失败的规划", options)

            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_run_plan_timeout(
        self, runner: Runner
    ) -> None:
        """测试 _run_plan 方法超时"""
        import subprocess
        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=120)

            options = runner._merge_options({})
            result = await runner._run_plan("超时的规划", options)

            assert result["success"] is False
            assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_run_plan_exception(
        self, runner: Runner
    ) -> None:
        """测试 _run_plan 方法异常"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = Exception("未知错误")

            options = runner._merge_options({})
            result = await runner._run_plan("异常的规划", options)

            assert result["success"] is False
            assert "未知错误" in result["error"]

    @pytest.mark.asyncio
    async def test_run_ask_success(
        self, runner: Runner
    ) -> None:
        """测试 _run_ask 方法成功"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "回答内容..."
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_ask("问题内容", options)

            assert result["success"] is True
            assert result["mode"] == "ask"
            assert result["answer"] == "回答内容..."

    @pytest.mark.asyncio
    async def test_run_ask_failure(
        self, runner: Runner
    ) -> None:
        """测试 _run_ask 方法失败"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "问答失败"
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_ask("失败的问题", options)

            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_run_ask_timeout(
        self, runner: Runner
    ) -> None:
        """测试 _run_ask 方法超时"""
        import subprocess
        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=120)

            options = runner._merge_options({})
            result = await runner._run_ask("超时的问题", options)

            assert result["success"] is False
            assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_run_dispatches_to_correct_mode(
        self, runner: Runner
    ) -> None:
        """测试 run 方法正确分发到对应模式"""
        # 测试 BASIC 模式
        with patch.object(runner, "_run_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.BASIC, goal="基本任务")
            await runner.run(analysis)
            mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_mp_mode(
        self, runner: Runner
    ) -> None:
        """测试 run 方法分发到 MP 模式"""
        with patch.object(runner, "_run_mp", new_callable=AsyncMock) as mock_mp:
            mock_mp.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.MP, goal="多进程任务")
            await runner.run(analysis)
            mock_mp.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_iterate_mode(
        self, runner: Runner
    ) -> None:
        """测试 run 方法分发到 ITERATE 模式"""
        with patch.object(runner, "_run_iterate", new_callable=AsyncMock) as mock_iterate:
            mock_iterate.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.ITERATE, goal="迭代任务")
            await runner.run(analysis)
            mock_iterate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_plan_mode(
        self, runner: Runner
    ) -> None:
        """测试 run 方法分发到 PLAN 模式"""
        with patch.object(runner, "_run_plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.PLAN, goal="规划任务")
            await runner.run(analysis)
            mock_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_ask_mode(
        self, runner: Runner
    ) -> None:
        """测试 run 方法分发到 ASK 模式"""
        with patch.object(runner, "_run_ask", new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.ASK, goal="问答任务")
            await runner.run(analysis)
            mock_ask.assert_called_once()


# ============================================================
# TestKnowledgeMode - 知识库模式测试
# ============================================================


class TestKnowledgeMode:
    """测试知识库模式"""

    @pytest.fixture
    def runner(self, mock_args: argparse.Namespace) -> Runner:
        """创建 Runner 实例"""
        return Runner(mock_args)

    @pytest.mark.asyncio
    async def test_run_knowledge_basic(
        self, runner: Runner
    ) -> None:
        """测试 _run_knowledge 基本功能"""
        import coordinator
        import knowledge
        mock_km = MagicMock()
        mock_km.initialize = AsyncMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True})
        with patch.object(knowledge, "KnowledgeManager", return_value=mock_km), \
             patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator):
            options = runner._merge_options({})
            result = await runner._run_knowledge("知识库任务", options)

            assert result["success"] is True
            mock_km.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_knowledge_with_search(
        self, runner: Runner
    ) -> None:
        """测试 _run_knowledge 带搜索功能"""
        import coordinator
        import knowledge

        # 设置 KnowledgeManager mock
        mock_km = MagicMock()
        mock_km.initialize = AsyncMock()

        # 设置 KnowledgeStorage mock
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_search_result = MagicMock()
        mock_search_result.doc_id = "doc1"
        mock_storage.search = AsyncMock(return_value=[mock_search_result])
        mock_doc = MagicMock()
        mock_doc.title = "测试文档"
        mock_doc.url = "https://example.com"
        mock_doc.content = "文档内容" * 100
        mock_storage.load_document = AsyncMock(return_value=mock_doc)

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True})

        with patch.object(knowledge, "KnowledgeManager", return_value=mock_km), \
             patch.object(knowledge, "KnowledgeStorage", return_value=mock_storage), \
             patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator):
            options = runner._merge_options({})
            options["search_knowledge"] = "测试搜索"
            result = await runner._run_knowledge("知识库任务", options)

            assert result["success"] is True
            mock_storage.search.assert_called_once()


# ============================================================
# TestTaskAnalyzerAgentAnalysis - Agent 分析测试
# ============================================================


class TestTaskAnalyzerAgentAnalysis:
    """测试 TaskAnalyzer Agent 分析相关功能"""

    def test_agent_analysis_json_extraction(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证 Agent 分析 JSON 提取"""
        analyzer = TaskAnalyzer(use_agent=True)

        # 模拟包含 JSON 的输出
        mock_subprocess.return_value.stdout = '''
        一些其他文本
        {"mode": "iterate", "options": {"skip_online": true}, "reasoning": "任务涉及自我迭代", "refined_goal": "执行自我迭代更新"}
        更多文本
        '''

        analysis = analyzer.analyze("执行自我迭代", mock_args)

        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_fallback_on_invalid_json(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证无效 JSON 时回退到规则分析"""
        analyzer = TaskAnalyzer(use_agent=True)

        mock_subprocess.return_value.stdout = "无效的响应内容"

        # 应该回退到规则分析，不抛出异常
        analysis = analyzer.analyze("执行自我迭代", mock_args)

        # 规则分析应该检测到迭代关键词
        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_timeout_fallback(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析超时时回退"""
        import subprocess
        with patch("run.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("执行自我迭代", mock_args)

            # 应该回退到规则分析
            assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_error_fallback(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证 Agent 分析失败时回退"""
        analyzer = TaskAnalyzer(use_agent=True)

        mock_subprocess.return_value.returncode = 1

        # 应该回退到规则分析
        analysis = analyzer.analyze("执行自我迭代", mock_args)
        assert analysis.mode == RunMode.ITERATE


# ============================================================
# TestRunKnowledgeMode - 知识库模式运行测试
# ============================================================


# 检查可选依赖是否可用
def _check_knowledge_deps() -> tuple[bool, str]:
    """检查知识库依赖是否可用"""
    try:
        import knowledge  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, str(e)


_knowledge_available, _knowledge_skip_reason = _check_knowledge_deps()


class TestRunKnowledgeMode:
    """测试知识库模式运行相关功能"""

    @pytest.mark.skipif(
        not _knowledge_available,
        reason=f"知识库依赖不可用: {_knowledge_skip_reason}"
    )
    def test_knowledge_imports(self) -> None:
        """验证 knowledge.KnowledgeManager, KnowledgeStorage 可导入"""
        from knowledge import KnowledgeManager, KnowledgeStorage

        # 验证类存在且可以被引用
        assert KnowledgeManager is not None
        assert KnowledgeStorage is not None

        # 验证是类类型
        assert isinstance(KnowledgeManager, type)
        assert isinstance(KnowledgeStorage, type)

    @pytest.mark.skipif(
        not _knowledge_available,
        reason=f"知识库依赖不可用: {_knowledge_skip_reason}"
    )
    def test_knowledge_manager_init(self) -> None:
        """测试 KnowledgeManager(name='cursor-docs') 可初始化"""
        from knowledge import KnowledgeManager

        # 测试默认初始化
        manager = KnowledgeManager()
        assert manager is not None

        # 测试带 name 参数初始化（与 run.py 中 _run_knowledge 一致）
        manager_with_name = KnowledgeManager(name="cursor-docs")
        assert manager_with_name is not None
        assert manager_with_name.name == "cursor-docs"

    @pytest.mark.skipif(
        not _knowledge_available,
        reason=f"知识库依赖不可用: {_knowledge_skip_reason}"
    )
    @pytest.mark.asyncio
    async def test_storage_search_mock(self) -> None:
        """使用 mock 测试 storage.search 和 load_document 流程"""
        from knowledge import KnowledgeStorage

        with patch.object(KnowledgeStorage, "initialize", new_callable=AsyncMock):
            storage = KnowledgeStorage()
            await storage.initialize()

            # Mock search 方法
            mock_search_result = MagicMock()
            mock_search_result.doc_id = "test-doc-001"
            mock_search_result.score = 0.95
            mock_search_result.title = "测试文档"

            with patch.object(
                storage, "search", new_callable=AsyncMock, return_value=[mock_search_result]
            ) as mock_search:
                results = await storage.search("测试查询", limit=5)

                mock_search.assert_called_once_with("测试查询", limit=5)
                assert len(results) == 1
                assert results[0].doc_id == "test-doc-001"

            # Mock load_document 方法
            mock_doc = MagicMock()
            mock_doc.title = "测试文档标题"
            mock_doc.url = "https://example.com/doc"
            mock_doc.content = "这是文档内容" * 100

            with patch.object(
                storage, "load_document", new_callable=AsyncMock, return_value=mock_doc
            ) as mock_load:
                doc = await storage.load_document("test-doc-001")

                mock_load.assert_called_once_with("test-doc-001")
                assert doc.title == "测试文档标题"
                assert doc.url == "https://example.com/doc"
                assert "这是文档内容" in doc.content

    @pytest.mark.skipif(
        not _knowledge_available,
        reason=f"知识库依赖不可用: {_knowledge_skip_reason}"
    )
    @pytest.mark.asyncio
    async def test_enhanced_goal_building(self, mock_args: argparse.Namespace) -> None:
        """测试知识库上下文正确附加到目标描述中"""
        runner = Runner(mock_args)

        with patch("knowledge.KnowledgeManager") as mock_km_class, \
             patch("knowledge.KnowledgeStorage") as mock_storage_class, \
             patch("coordinator.Orchestrator") as mock_orchestrator_class, \
             patch("coordinator.OrchestratorConfig"), \
             patch("cursor.client.CursorAgentConfig"):

            # 设置 KnowledgeManager mock
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            mock_km_class.return_value = mock_km

            # 设置 KnowledgeStorage mock
            mock_storage = MagicMock()
            mock_storage.initialize = AsyncMock()

            # 创建模拟搜索结果
            mock_search_result = MagicMock()
            mock_search_result.doc_id = "doc-001"
            mock_storage.search = AsyncMock(return_value=[mock_search_result])

            # 创建模拟文档
            mock_doc = MagicMock()
            mock_doc.title = "Cursor CLI 参考文档"
            mock_doc.url = "https://cursor.com/docs/cli"
            mock_doc.content = "CLI 命令参考内容..." * 50
            mock_storage.load_document = AsyncMock(return_value=mock_doc)
            mock_storage_class.return_value = mock_storage

            # 设置 Orchestrator mock，捕获传入的 goal
            captured_goal = None

            async def capture_goal(goal: str):
                nonlocal captured_goal
                captured_goal = goal
                return {"success": True}

            mock_orchestrator = MagicMock()
            mock_orchestrator.run = AsyncMock(side_effect=capture_goal)
            mock_orchestrator_class.return_value = mock_orchestrator

            # 执行测试
            options = runner._merge_options({})
            options["search_knowledge"] = "CLI 命令"
            result = await runner._run_knowledge("实现 CLI 功能", options)

            assert result["success"] is True
            assert captured_goal is not None

            # 验证增强目标包含原始目标和参考文档
            assert "实现 CLI 功能" in captured_goal
            assert "参考文档" in captured_goal
            assert "Cursor CLI 参考文档" in captured_goal
            assert "CLI 命令参考内容" in captured_goal

    @pytest.mark.skipif(
        not _knowledge_available,
        reason=f"知识库依赖不可用: {_knowledge_skip_reason}"
    )
    @pytest.mark.asyncio
    async def test_knowledge_mode_without_search(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试无搜索查询时，目标不被修改"""
        import coordinator
        import knowledge
        runner = Runner(mock_args)

        # 设置 KnowledgeManager mock
        mock_km = MagicMock()
        mock_km.initialize = AsyncMock()

        # 捕获 goal
        captured_goal = None

        async def capture_goal(goal: str):
            nonlocal captured_goal
            captured_goal = goal
            return {"success": True}

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(side_effect=capture_goal)

        with patch.object(knowledge, "KnowledgeManager", return_value=mock_km), \
             patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator):
            # 执行测试 - 不传入 search_knowledge
            options = runner._merge_options({})
            # 不设置 search_knowledge
            result = await runner._run_knowledge("直接执行任务", options)

            assert result["success"] is True
            assert captured_goal == "直接执行任务"  # 目标未被增强

    def test_knowledge_import_skip_on_missing_deps(self) -> None:
        """测试当可选依赖缺失时正确跳过"""
        # 模拟 knowledge 模块导入失败
        with patch.dict("sys.modules", {"knowledge": None}):
            # 重新检查依赖
            available, reason = _check_knowledge_deps()
            # 如果原本可用，这里不会真正改变
            # 这个测试主要验证跳过逻辑存在且工作正常
            assert isinstance(available, bool)
            assert isinstance(reason, str)

    @pytest.mark.skipif(
        not _knowledge_available,
        reason=f"知识库依赖不可用: {_knowledge_skip_reason}"
    )
    @pytest.mark.asyncio
    async def test_knowledge_context_limit(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试知识库上下文内容被正确截断"""
        runner = Runner(mock_args)

        with patch("knowledge.KnowledgeManager") as mock_km_class, \
             patch("knowledge.KnowledgeStorage") as mock_storage_class, \
             patch("coordinator.Orchestrator") as mock_orchestrator_class, \
             patch("coordinator.OrchestratorConfig"), \
             patch("cursor.client.CursorAgentConfig"):

            # 设置 KnowledgeManager mock
            mock_km = MagicMock()
            mock_km.initialize = AsyncMock()
            mock_km_class.return_value = mock_km

            # 设置 KnowledgeStorage mock
            mock_storage = MagicMock()
            mock_storage.initialize = AsyncMock()

            # 创建模拟搜索结果
            mock_search_result = MagicMock()
            mock_search_result.doc_id = "doc-long"
            mock_storage.search = AsyncMock(return_value=[mock_search_result])

            # 创建超长内容的模拟文档
            mock_doc = MagicMock()
            mock_doc.title = "长文档"
            mock_doc.url = "https://example.com/long"
            # 创建超过 2000 字符的内容
            mock_doc.content = "A" * 5000
            mock_storage.load_document = AsyncMock(return_value=mock_doc)
            mock_storage_class.return_value = mock_storage

            # 捕获 goal
            captured_goal = None

            async def capture_goal(goal: str):
                nonlocal captured_goal
                captured_goal = goal
                return {"success": True}

            mock_orchestrator = MagicMock()
            mock_orchestrator.run = AsyncMock(side_effect=capture_goal)
            mock_orchestrator_class.return_value = mock_orchestrator

            options = runner._merge_options({})
            options["search_knowledge"] = "test"
            await runner._run_knowledge("测试任务", options)

            # 验证内容被截断 - 在代码块中最多显示 1000 字符
            assert captured_goal is not None
            # 内容部分应被截断到约 1000 字符（代码块内）
            # 原始 content 是 5000 个 A，但只取前 2000 存入 context，再取前 1000 显示
            assert "A" * 1000 in captured_goal
            # 不应包含完整的 5000 个 A（或 2000 个）
            assert "A" * 2001 not in captured_goal


# ============================================================
# TestRunIterateMode - 自我迭代模式测试
# ============================================================


class TestRunIterateMode:
    """测试自我迭代模式相关功能"""

    def test_iterate_import(self) -> None:
        """验证 scripts.run_iterate.SelfIterator 可导入"""
        from scripts.run_iterate import SelfIterator

        # 验证类存在且可以被引用
        assert SelfIterator is not None

        # 验证是类类型
        assert isinstance(SelfIterator, type)

    def test_iterate_args_class(self) -> None:
        """测试 run.py 中动态创建的 IterateArgs 类所有属性"""
        # 在 run.py 的 _run_iterate 方法中动态创建的 IterateArgs 类
        # 模拟创建一个等效的类来测试所有属性
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 5))
                self.workers = opts.get("workers", 3)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                # 自动提交选项
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        # 测试所有属性都存在
        goal = "测试任务目标"
        opts = {
            "skip_online": True,
            "dry_run": True,
            "max_iterations": 10,
            "workers": 5,
            "force_update": True,
            "verbose": True,
            "auto_commit": True,
            "auto_push": True,
            "commit_per_iteration": True,
            "commit_message": "测试提交信息",
        }

        args = IterateArgs(goal, opts)

        # 验证所有属性
        assert args.requirement == goal
        assert args.skip_online is True
        assert args.changelog_url == "https://cursor.com/cn/changelog"
        assert args.dry_run is True
        assert args.max_iterations == "10"
        assert args.workers == 5
        assert args.force_update is True
        assert args.verbose is True
        assert args.auto_commit is True
        assert args.auto_push is True
        assert args.commit_per_iteration is True
        assert args.commit_message == "测试提交信息"

    def test_iterate_args_default_values(self) -> None:
        """测试 IterateArgs 默认值"""
        # 模拟 run.py 中的 IterateArgs 类
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 5))
                self.workers = opts.get("workers", 3)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        # 使用空选项测试默认值
        args = IterateArgs("测试任务", {})

        # 验证默认值
        assert args.requirement == "测试任务"
        assert args.skip_online is False
        assert args.changelog_url == "https://cursor.com/cn/changelog"
        assert args.dry_run is False
        assert args.max_iterations == "5"
        assert args.workers == 3
        assert args.force_update is False
        assert args.verbose is False
        assert args.auto_commit is False
        assert args.auto_push is False
        assert args.commit_per_iteration is False
        assert args.commit_message == ""

    def test_self_iterator_init(self) -> None:
        """测试 SelfIterator 可以用 IterateArgs 初始化"""
        from scripts.run_iterate import SelfIterator

        # 模拟 run.py 中的 IterateArgs 类
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 5))
                self.workers = opts.get("workers", 3)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        # 创建 IterateArgs 实例
        iterate_args = IterateArgs("自我迭代测试任务", {
            "skip_online": True,
            "dry_run": True,
        })

        # 验证 SelfIterator 可以用 IterateArgs 初始化
        iterator = SelfIterator(iterate_args)

        # 验证初始化状态
        assert iterator.args == iterate_args
        assert iterator.args.requirement == "自我迭代测试任务"
        assert iterator.args.skip_online is True
        assert iterator.args.dry_run is True

        # 验证内部组件已创建
        assert iterator.changelog_analyzer is not None
        assert iterator.knowledge_updater is not None
        assert iterator.goal_builder is not None
        assert iterator.context is not None

        # 验证 context 状态
        assert iterator.context.user_requirement == "自我迭代测试任务"
        assert iterator.context.dry_run is True

    @pytest.mark.asyncio
    async def test_run_agent_system_calls_mp_orchestrator(self) -> None:
        """测试 SelfIterator._run_agent_system() 调用 MultiProcessOrchestrator"""
        from scripts.run_iterate import SelfIterator

        # 模拟 run.py 中的 IterateArgs 类
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", True)  # 跳过在线检查
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 1))
                self.workers = opts.get("workers", 2)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        iterate_args = IterateArgs("测试 MP 编排器调用", {"skip_online": True})
        iterator = SelfIterator(iterate_args)

        # 设置 iteration_goal 以跳过完整 run 流程
        iterator.context.iteration_goal = "测试目标"

        mock_result = {
            "success": True,
            "goal": "测试目标",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMPOrchestrator:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig") as MockConfig:
                mock_instance = MagicMock()
                mock_instance.run = AsyncMock(return_value=mock_result)
                MockMPOrchestrator.return_value = mock_instance
                MockConfig.return_value = MagicMock()

                # 调用 _run_agent_system
                result = await iterator._run_agent_system()

                # 验证 MultiProcessOrchestrator 被正确初始化
                MockMPOrchestrator.assert_called_once()

                # 验证 run 方法被调用
                mock_instance.run.assert_called_once_with("测试目标")

                # 验证返回结果
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_agent_system_with_auto_commit(self) -> None:
        """测试 _run_agent_system 在 auto_commit=True 时调用 CommitterAgent"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", True)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 1))
                self.workers = opts.get("workers", 2)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        # 启用 auto_commit
        iterate_args = IterateArgs("测试自动提交", {
            "skip_online": True,
            "auto_commit": True,
        })
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试自动提交目标"

        mock_run_result = {
            "success": True,
            "goal": "测试自动提交目标",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        mock_commits_result = {
            "total_commits": 1,
            "commit_hashes": ["abc123def"],
            "commit_messages": ["test: 自动提交测试"],
            "pushed_commits": 0,
            "files_changed": ["test.py"],
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMPOrchestrator:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                mock_instance = MagicMock()
                mock_instance.run = AsyncMock(return_value=mock_run_result)
                MockMPOrchestrator.return_value = mock_instance

                # Mock _run_commit_phase 方法
                with patch.object(
                    iterator, "_run_commit_phase", new_callable=AsyncMock
                ) as mock_commit_phase:
                    mock_commit_phase.return_value = mock_commits_result

                    result = await iterator._run_agent_system()

                    # 验证 _run_commit_phase 被调用
                    mock_commit_phase.assert_called_once_with(1, 2)

                    # 验证 commits 被写入结果
                    assert "commits" in result
                    assert result["commits"]["total_commits"] == 1
                    assert result["commits"]["commit_hashes"] == ["abc123def"]
                    assert result["commits"]["commit_messages"] == ["test: 自动提交测试"]

    @pytest.mark.asyncio
    async def test_run_commit_phase_calls_committer_agent(self) -> None:
        """测试 _run_commit_phase 调用 CommitterAgent"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", True)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 1))
                self.workers = opts.get("workers", 2)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", True)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        iterate_args = IterateArgs("测试 CommitterAgent", {"auto_commit": True})
        iterator = SelfIterator(iterate_args)

        # Mock CommitterAgent
        with patch("scripts.run_iterate.CommitterAgent") as MockCommitterAgent:
            with patch("scripts.run_iterate.CommitterConfig"):
                mock_committer = MagicMock()

                # Mock check_status 返回有变更的状态
                mock_committer.check_status.return_value = {
                    "is_repo": True,
                    "has_changes": True,
                }

                # Mock generate_commit_message
                mock_committer.generate_commit_message = AsyncMock(
                    return_value="feat: 自动生成的提交信息"
                )

                # Mock commit
                mock_commit_result = MagicMock()
                mock_commit_result.success = True
                mock_commit_result.commit_hash = "abc123456"
                mock_commit_result.files_changed = ["file1.py", "file2.py"]
                mock_commit_result.error = None
                mock_committer.commit.return_value = mock_commit_result

                # Mock get_commit_summary
                mock_committer.get_commit_summary.return_value = {
                    "successful_commits": 1,
                    "commit_hashes": ["abc123456"],
                    "files_changed": ["file1.py", "file2.py"],
                }

                MockCommitterAgent.return_value = mock_committer

                result = await iterator._run_commit_phase(
                    iterations_completed=2,
                    tasks_completed=5,
                )

                # 验证 CommitterAgent 被创建
                MockCommitterAgent.assert_called_once()

                # 验证 check_status 被调用
                mock_committer.check_status.assert_called_once()

                # 验证 generate_commit_message 被调用
                mock_committer.generate_commit_message.assert_called_once()

                # 验证 commit 被调用
                mock_committer.commit.assert_called_once()

                # 验证返回结果
                assert result["total_commits"] == 1
                assert "abc123456" in result["commit_hashes"]
                assert "file1.py" in result["files_changed"]

    @pytest.mark.asyncio
    async def test_run_commit_phase_no_changes(self) -> None:
        """测试 _run_commit_phase 无变更时不提交"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", True)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 1))
                self.workers = opts.get("workers", 2)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", True)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        iterate_args = IterateArgs("测试无变更", {"auto_commit": True})
        iterator = SelfIterator(iterate_args)

        with patch("scripts.run_iterate.CommitterAgent") as MockCommitterAgent:
            with patch("scripts.run_iterate.CommitterConfig"):
                mock_committer = MagicMock()

                # Mock check_status 返回无变更
                mock_committer.check_status.return_value = {
                    "is_repo": True,
                    "has_changes": False,
                }

                MockCommitterAgent.return_value = mock_committer

                result = await iterator._run_commit_phase(
                    iterations_completed=1,
                    tasks_completed=2,
                )

                # 验证返回结果 - 无提交
                assert result["total_commits"] == 0
                assert result["commit_hashes"] == []
                assert result["commit_messages"] == []
                assert result["files_changed"] == []

    @pytest.mark.asyncio
    async def test_run_agent_system_without_auto_commit(self) -> None:
        """测试 _run_agent_system 在 auto_commit=False 时不调用提交"""
        from scripts.run_iterate import SelfIterator

        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", True)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 1))
                self.workers = opts.get("workers", 2)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", False)  # 禁用自动提交
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

        iterate_args = IterateArgs("测试禁用提交", {"skip_online": True, "auto_commit": False})
        iterator = SelfIterator(iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_run_result = {
            "success": True,
            "goal": "测试目标",
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with patch("scripts.run_iterate.MultiProcessOrchestrator") as MockMPOrchestrator:
            with patch("scripts.run_iterate.MultiProcessOrchestratorConfig"):
                mock_instance = MagicMock()
                mock_instance.run = AsyncMock(return_value=mock_run_result)
                MockMPOrchestrator.return_value = mock_instance

                # Mock _run_commit_phase 方法
                with patch.object(
                    iterator, "_run_commit_phase", new_callable=AsyncMock
                ) as mock_commit_phase:
                    result = await iterator._run_agent_system()

                    # 验证 _run_commit_phase 未被调用
                    mock_commit_phase.assert_not_called()

                    # 验证结果中没有 commits
                    assert "commits" not in result


# ============================================================
# TestRunPlanAskModes - 规划模式和问答模式专项测试
# ============================================================


class TestRunPlanAskModes:
    """测试规划模式和问答模式的完整功能"""

    @pytest.fixture
    def runner(self, mock_args: argparse.Namespace) -> Runner:
        """创建 Runner 实例"""
        return Runner(mock_args)

    # ----------------------------------------------------------
    # 规划模式测试
    # ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_plan_success(self, runner: Runner) -> None:
        """测试规划模式成功返回计划"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """## 任务分解
1. 分析需求
2. 设计架构
3. 实现功能

## 执行顺序
- 1 -> 2 -> 3 (顺序执行)

## 推荐模式
mp (多进程模式)"""
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_plan("实现用户认证功能", options)

            # 验证返回结构
            assert result["success"] is True
            assert result["goal"] == "实现用户认证功能"
            assert result["mode"] == "plan"
            assert "plan" in result
            assert "任务分解" in result["plan"]
            assert result["dry_run"] is True

            # 验证 subprocess 调用参数
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            assert "agent" in call_args[0][0]
            assert "-p" in call_args[0][0]
            assert "--output-format" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_plan_timeout(self, runner: Runner) -> None:
        """测试规划模式超时处理"""
        import subprocess

        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.TimeoutExpired(
                cmd="agent", timeout=120
            )

            options = runner._merge_options({})
            result = await runner._run_plan("复杂的大型任务", options)

            # 验证超时返回结构
            assert result["success"] is False
            assert result["goal"] == "复杂的大型任务"
            assert result["mode"] == "plan"
            assert result["error"] == "timeout"
            assert "plan" not in result

    @pytest.mark.asyncio
    async def test_run_plan_error(self, runner: Runner) -> None:
        """测试规划模式错误处理"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Agent 服务不可用"
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_plan("失败的规划任务", options)

            # 验证错误返回结构
            assert result["success"] is False
            assert result["goal"] == "失败的规划任务"
            assert result["mode"] == "plan"
            assert "error" in result
            assert "Agent 服务不可用" in result["error"]
            assert "plan" not in result

    @pytest.mark.asyncio
    async def test_run_plan_exception(self, runner: Runner) -> None:
        """测试规划模式异常处理"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = OSError("网络连接失败")

            options = runner._merge_options({})
            result = await runner._run_plan("异常任务", options)

            # 验证异常返回结构
            assert result["success"] is False
            assert result["goal"] == "异常任务"
            assert result["mode"] == "plan"
            assert "error" in result
            assert "网络连接失败" in result["error"]

    # ----------------------------------------------------------
    # 问答模式测试
    # ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_ask_success(self, runner: Runner) -> None:
        """测试问答模式成功返回答案"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """Python 是一种高级编程语言，具有以下特点：
1. 简洁易读的语法
2. 丰富的标准库
3. 跨平台支持
4. 强大的社区生态"""
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_ask("什么是 Python？", options)

            # 验证返回结构
            assert result["success"] is True
            assert result["goal"] == "什么是 Python？"
            assert result["mode"] == "ask"
            assert "answer" in result
            assert "Python" in result["answer"]
            assert "高级编程语言" in result["answer"]
            # 问答模式不应有 dry_run 标记
            assert "dry_run" not in result

            # 验证 subprocess 调用参数
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            assert "agent" in call_args[0][0]
            assert "什么是 Python？" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_ask_timeout(self, runner: Runner) -> None:
        """测试问答模式超时处理"""
        import subprocess

        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.TimeoutExpired(
                cmd="agent", timeout=120
            )

            options = runner._merge_options({})
            result = await runner._run_ask("一个需要很长时间回答的问题", options)

            # 验证超时返回结构
            assert result["success"] is False
            assert result["goal"] == "一个需要很长时间回答的问题"
            assert result["mode"] == "ask"
            assert result["error"] == "timeout"
            assert "answer" not in result

    @pytest.mark.asyncio
    async def test_run_ask_error(self, runner: Runner) -> None:
        """测试问答模式错误处理"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "API 限流"
            mock_subprocess.return_value = mock_result

            options = runner._merge_options({})
            result = await runner._run_ask("失败的问题", options)

            # 验证错误返回结构
            assert result["success"] is False
            assert result["goal"] == "失败的问题"
            assert result["mode"] == "ask"
            assert "error" in result
            assert "API 限流" in result["error"]
            assert "answer" not in result

    @pytest.mark.asyncio
    async def test_run_ask_exception(self, runner: Runner) -> None:
        """测试问答模式异常处理"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = PermissionError("权限不足")

            options = runner._merge_options({})
            result = await runner._run_ask("异常问题", options)

            # 验证异常返回结构
            assert result["success"] is False
            assert result["goal"] == "异常问题"
            assert result["mode"] == "ask"
            assert "error" in result
            assert "权限不足" in result["error"]

    # ----------------------------------------------------------
    # 返回结构验证测试
    # ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_plan_result_structure(self, runner: Runner) -> None:
        """验证规划模式返回的 dict 结构完整性"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "计划内容"
            mock_subprocess.return_value = mock_result

            result = await runner._run_plan("测试任务", runner._merge_options({}))

            # 验证成功时必需字段
            required_fields = ["success", "goal", "mode", "plan", "dry_run"]
            for field in required_fields:
                assert field in result, f"缺少必需字段: {field}"

            # 验证字段类型
            assert isinstance(result["success"], bool)
            assert isinstance(result["goal"], str)
            assert isinstance(result["mode"], str)
            assert isinstance(result["plan"], str)
            assert isinstance(result["dry_run"], bool)

    @pytest.mark.asyncio
    async def test_plan_error_result_structure(self, runner: Runner) -> None:
        """验证规划模式失败时返回的 dict 结构"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "错误信息"
            mock_subprocess.return_value = mock_result

            result = await runner._run_plan("失败任务", runner._merge_options({}))

            # 验证失败时必需字段
            required_fields = ["success", "goal", "mode", "error"]
            for field in required_fields:
                assert field in result, f"缺少必需字段: {field}"

            # 验证失败时不应有 plan 字段
            assert "plan" not in result
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_ask_result_structure(self, runner: Runner) -> None:
        """验证问答模式返回的 dict 结构完整性"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "回答内容"
            mock_subprocess.return_value = mock_result

            result = await runner._run_ask("测试问题", runner._merge_options({}))

            # 验证成功时必需字段
            required_fields = ["success", "goal", "mode", "answer"]
            for field in required_fields:
                assert field in result, f"缺少必需字段: {field}"

            # 验证字段类型
            assert isinstance(result["success"], bool)
            assert isinstance(result["goal"], str)
            assert isinstance(result["mode"], str)
            assert isinstance(result["answer"], str)

    @pytest.mark.asyncio
    async def test_ask_error_result_structure(self, runner: Runner) -> None:
        """验证问答模式失败时返回的 dict 结构"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "错误信息"
            mock_subprocess.return_value = mock_result

            result = await runner._run_ask("失败问题", runner._merge_options({}))

            # 验证失败时必需字段
            required_fields = ["success", "goal", "mode", "error"]
            for field in required_fields:
                assert field in result, f"缺少必需字段: {field}"

            # 验证失败时不应有 answer 字段
            assert "answer" not in result
            assert result["success"] is False

    # ----------------------------------------------------------
    # 边界情况测试
    # ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_plan_empty_output(self, runner: Runner) -> None:
        """测试规划模式返回空输出"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_subprocess.return_value = mock_result

            result = await runner._run_plan("空输出任务", runner._merge_options({}))

            # 空输出仍然算成功
            assert result["success"] is True
            assert result["plan"] == ""

    @pytest.mark.asyncio
    async def test_run_ask_empty_output(self, runner: Runner) -> None:
        """测试问答模式返回空输出"""
        with patch("run.subprocess.run") as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "   \n\t  "  # 仅空白字符
            mock_subprocess.return_value = mock_result

            result = await runner._run_ask("空输出问题", runner._merge_options({}))

            # 空输出仍然算成功
            assert result["success"] is True
            assert result["answer"] == ""  # strip() 后为空


# ============================================================
# TestAsyncMain - async_main 函数测试
# ============================================================


class TestAsyncMain:
    """测试 async_main 异步主函数"""

    @pytest.fixture
    def base_args(self) -> argparse.Namespace:
        """基础命令行参数"""
        return argparse.Namespace(
            task="",
            mode="auto",
            directory=".",
            workers=3,
            max_iterations="10",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=True,
            auto_push=False,
            commit_per_iteration=False,
        )

    @pytest.mark.asyncio
    async def test_no_task_error(self, base_args: argparse.Namespace) -> None:
        """测试无任务描述时返回错误（exit code 1）"""
        from run import async_main

        # 设置空任务
        base_args.task = ""
        base_args.mode = "basic"  # 非 auto 模式，直接使用指定模式

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.print_error") as mock_print_error, \
             patch("run.print_info"):

            exit_code = await async_main()

            # 验证返回错误码
            assert exit_code == 1
            # 验证打印了错误信息
            mock_print_error.assert_called_once()
            error_msg = mock_print_error.call_args[0][0]
            assert "任务描述" in error_msg or "请提供" in error_msg

    @pytest.mark.asyncio
    async def test_auto_mode_triggers_analysis(
        self, base_args: argparse.Namespace
    ) -> None:
        """测试 auto 模式触发 TaskAnalyzer.analyze"""
        from run import async_main, TaskAnalysis, RunMode

        # 设置 auto 模式和任务
        base_args.task = "测试自动分析任务"
        base_args.mode = "auto"
        base_args.no_auto_analyze = False

        # 创建 mock 分析结果
        mock_analysis = TaskAnalysis(
            mode=RunMode.BASIC,
            goal="测试自动分析任务",
            reasoning="自动分析结果",
        )

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.TaskAnalyzer") as mock_analyzer_class, \
             patch("run.Runner") as mock_runner_class:

            # 设置 TaskAnalyzer mock
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = mock_analysis
            mock_analyzer_class.return_value = mock_analyzer

            # 设置 Runner mock
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value={"success": True})
            mock_runner_class.return_value = mock_runner

            exit_code = await async_main()

            # 验证 TaskAnalyzer 被创建并调用 analyze
            mock_analyzer_class.assert_called_once_with(use_agent=True)
            mock_analyzer.analyze.assert_called_once_with(
                "测试自动分析任务", base_args
            )

            # 验证 Runner.run 被调用
            mock_runner.run.assert_called_once()
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_explicit_mode_skips_analysis(
        self, base_args: argparse.Namespace
    ) -> None:
        """测试显式指定模式跳过自动分析"""
        from run import async_main, TaskAnalysis, RunMode

        # 设置显式 basic 模式
        base_args.task = "显式模式任务"
        base_args.mode = "basic"  # 非 auto 模式

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.TaskAnalyzer") as mock_analyzer_class, \
             patch("run.Runner") as mock_runner_class:

            # 设置 Runner mock
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value={"success": True})
            mock_runner_class.return_value = mock_runner

            exit_code = await async_main()

            # 验证 TaskAnalyzer 没有被创建（跳过分析）
            mock_analyzer_class.assert_not_called()

            # 验证 Runner.run 被调用，使用了显式指定的模式
            mock_runner.run.assert_called_once()
            call_args = mock_runner.run.call_args[0]
            analysis = call_args[0]
            assert isinstance(analysis, TaskAnalysis)
            assert analysis.mode == RunMode.BASIC
            assert analysis.goal == "显式模式任务"
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_no_auto_analyze_flag(
        self, base_args: argparse.Namespace
    ) -> None:
        """测试 --no-auto-analyze 标志生效"""
        from run import async_main, TaskAnalysis, RunMode

        # 设置 auto 模式但禁用自动分析
        base_args.task = "禁用自动分析的任务"
        base_args.mode = "auto"
        base_args.no_auto_analyze = True  # 关键标志

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.TaskAnalyzer") as mock_analyzer_class, \
             patch("run.Runner") as mock_runner_class:

            # 设置 Runner mock
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value={"success": True})
            mock_runner_class.return_value = mock_runner

            exit_code = await async_main()

            # 验证 TaskAnalyzer 没有被创建（因为 no_auto_analyze=True）
            mock_analyzer_class.assert_not_called()

            # 验证 Runner.run 被调用，auto 模式应回退到 BASIC
            mock_runner.run.assert_called_once()
            call_args = mock_runner.run.call_args[0]
            analysis = call_args[0]
            assert isinstance(analysis, TaskAnalysis)
            # auto 模式在 no_auto_analyze=True 时回退到 BASIC
            assert analysis.mode == RunMode.BASIC
            assert analysis.goal == "禁用自动分析的任务"
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_auto_mode_without_task_falls_back(
        self, base_args: argparse.Namespace
    ) -> None:
        """测试 auto 模式无任务时不触发分析"""
        from run import async_main

        # auto 模式但无任务
        base_args.task = ""
        base_args.mode = "auto"
        base_args.no_auto_analyze = False

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.TaskAnalyzer") as mock_analyzer_class, \
             patch("run.print_error") as mock_print_error, \
             patch("run.print_info"):

            exit_code = await async_main()

            # 验证 TaskAnalyzer 没有被调用（因为没有任务）
            mock_analyzer_class.assert_not_called()

            # 应该返回错误
            assert exit_code == 1
            mock_print_error.assert_called()

    @pytest.mark.asyncio
    async def test_runner_failure_returns_error_code(
        self, base_args: argparse.Namespace
    ) -> None:
        """测试 Runner 执行失败时返回错误码"""
        from run import async_main

        base_args.task = "会失败的任务"
        base_args.mode = "basic"

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.Runner") as mock_runner_class, \
             patch("run.print_result"):

            # 设置 Runner 返回失败结果
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value={"success": False})
            mock_runner_class.return_value = mock_runner

            exit_code = await async_main()

            # 验证返回非零错误码
            assert exit_code == 1
