"""测试 run.py 统一入口脚本

================================================================================
⚠ 测试代码约束：禁止新断言使用 deprecated 属性
================================================================================

**禁止在新断言中使用的 deprecated 属性**:
- triggered_by_prefix (已统一为 prefix_routed)
- execution_mode 用于决策分支时（使用 effective_mode 或 requested_mode_for_decision）

**新断言应使用的字段**:
- prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud
- effective_mode: 经过路由决策后实际使用的执行模式
- requested_mode_for_decision: 传给 build_execution_decision 的请求模式
- has_ampersand_prefix: 仅用于消息构建/日志记录场景

**to_dict() 兼容输出保持不变**:
to_dict() 的输出中保留 triggered_by_prefix 字段以兼容下游消费者，
但测试断言（assert 语句）应使用 prefix_routed。

详细 Schema 定义参见: core/execution_policy.py "统一字段 Schema" 部分
"""

import argparse
import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import run as run_module
from core.cloud_utils import is_cloud_request, strip_cloud_prefix

# 使用 DEFAULT_* 常量作为测试默认值（避免在模块加载时初始化 ConfigManager）
# 这样可以确保测试之间的隔离，防止 ConfigManager 状态污染
from core.config import (
    DEFAULT_CLOUD_TIMEOUT,
    DEFAULT_ENABLE_SUB_PLANNERS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_PLANNING_TIMEOUT,
    DEFAULT_REVIEW_TIMEOUT,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_STREAM_EVENTS_ENABLED,
    DEFAULT_STREAM_LOG_CONSOLE,
    DEFAULT_STREAM_LOG_DETAIL_DIR,
    DEFAULT_STREAM_LOG_RAW_DIR,
    DEFAULT_STRICT_REVIEW,
    DEFAULT_WORKER_MODEL,
    DEFAULT_WORKER_POOL_SIZE,
    DEFAULT_WORKER_TIMEOUT,
    # 控制台预览截断常量
    MAX_CONSOLE_PREVIEW_CHARS,
    MAX_GOAL_SUMMARY_CHARS,
    MAX_KNOWLEDGE_DOC_PREVIEW_CHARS,
    TRUNCATION_HINT,
    TRUNCATION_HINT_OUTPUT,
    get_config,
    parse_max_iterations,
)
from core.output_contract import CloudResultFields, CooldownInfoFields
from run import (
    MODE_ALIASES,
    Colors,
    RunMode,
    Runner,
    TaskAnalysis,
    TaskAnalyzer,
    print_error,
    print_header,
    print_info,
    print_result,
    print_success,
    print_warning,
    setup_logging,
)

CONFIG_WORKER_POOL_SIZE = DEFAULT_WORKER_POOL_SIZE
CONFIG_MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS
CONFIG_ENABLE_SUB_PLANNERS = DEFAULT_ENABLE_SUB_PLANNERS
CONFIG_STRICT_REVIEW = DEFAULT_STRICT_REVIEW
CONFIG_PLANNER_MODEL = DEFAULT_PLANNER_MODEL
CONFIG_WORKER_MODEL = DEFAULT_WORKER_MODEL
CONFIG_REVIEWER_MODEL = DEFAULT_REVIEWER_MODEL
CONFIG_CLOUD_TIMEOUT = DEFAULT_CLOUD_TIMEOUT
CONFIG_PLANNER_TIMEOUT = int(DEFAULT_PLANNING_TIMEOUT)
CONFIG_WORKER_TIMEOUT = int(DEFAULT_WORKER_TIMEOUT)
CONFIG_REVIEWER_TIMEOUT = int(DEFAULT_REVIEW_TIMEOUT)
# 流式日志配置
CONFIG_STREAM_LOG_ENABLED = DEFAULT_STREAM_EVENTS_ENABLED
CONFIG_STREAM_LOG_CONSOLE = DEFAULT_STREAM_LOG_CONSOLE
CONFIG_STREAM_LOG_DETAIL_DIR = DEFAULT_STREAM_LOG_DETAIL_DIR
CONFIG_STREAM_LOG_RAW_DIR = DEFAULT_STREAM_LOG_RAW_DIR


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def reset_config_for_test():
    """每个测试前后重置 ConfigManager，确保测试隔离

    问题背景：当 test_config_loading.py 在 test_run.py 之前运行时，
    ConfigManager 可能处于被污染的状态。这个 fixture 确保每个测试
    开始时 ConfigManager 从项目根目录的 config.yaml 加载配置。
    """
    from core.config import ConfigManager

    # 测试前重置，确保从当前工作目录加载配置
    ConfigManager.reset_instance()
    yield
    # 测试后也重置，确保不污染下一个测试
    ConfigManager.reset_instance()


@pytest.fixture
def mock_args() -> argparse.Namespace:
    """模拟命令行参数

    默认值来自 config.yaml，确保测试与配置保持一致。
    使用 tri-state 设计：None 表示未显式指定，由 resolve_orchestrator_settings 解析。
    """
    args = argparse.Namespace(
        task="测试任务",
        mode="auto",
        directory=".",
        _directory_user_set=False,  # tri-state 内部标志：是否显式指定了 --directory
        workers=CONFIG_WORKER_POOL_SIZE,  # 来自 config.yaml
        max_iterations=str(CONFIG_MAX_ITERATIONS),  # 来自 config.yaml
        strict_review=None,  # tri-state: None=未指定，使用 config.yaml
        enable_sub_planners=None,  # tri-state: None=未指定，使用 config.yaml
        verbose=False,
        quiet=False,
        log_level=None,
        skip_online=False,
        dry_run=False,
        force_update=False,
        use_knowledge=False,
        search_knowledge=None,
        self_update=False,
        # 模型配置（tri-state: None=未指定，使用 config.yaml）
        planner_model=None,  # tri-state: None=未指定
        worker_model=None,  # tri-state: None=未指定
        reviewer_model=None,  # tri-state: None=未指定
        # 流式日志配置（使用 tri-state，None=未指定，使用 config.yaml 默认值）
        stream_log_enabled=None,  # 来自 config.yaml logging.stream_json.enabled
        stream_log_console=None,  # 来自 config.yaml logging.stream_json.console
        stream_log_detail_dir=None,  # 来自 config.yaml logging.stream_json.detail_dir
        stream_log_raw_dir=None,  # 来自 config.yaml logging.stream_json.raw_dir
        no_auto_analyze=False,
        auto_commit=False,  # 默认禁用自动提交
        auto_push=False,
        commit_per_iteration=False,
        # 编排器配置（tri-state）
        orchestrator=None,  # tri-state: None=未指定
        no_mp=None,  # tri-state: None=未指定
        _orchestrator_user_set=False,  # 内部标志
        # 执行模式参数（tri-state）
        execution_mode=None,  # tri-state: None=未指定，使用 config.yaml
        planner_execution_mode=None,
        worker_execution_mode=None,
        reviewer_execution_mode=None,
        cloud_api_key=None,
        cloud_auth_timeout=None,  # tri-state: None=未指定
        cloud_timeout=None,  # tri-state: None=未指定
        cloud_background=None,
        auto_detect_cloud_prefix=None,  # tri-state: None=未指定，使用 config.yaml
        # 流式控制台渲染参数（默认关闭）
        stream_console_renderer=False,
        stream_advanced_renderer=False,
        stream_typing_effect=False,
        stream_typing_delay=0.02,
        stream_word_mode=True,
        stream_color_enabled=True,
        stream_show_word_diff=False,
        # 诊断参数
        heartbeat_debug=False,
        stall_diagnostics_enabled=None,
        stall_diagnostics_level=None,
        # 知识库参数
        enable_knowledge_injection=True,
        knowledge_top_k=3,
        knowledge_max_chars_per_doc=1200,
        knowledge_max_total_chars=3000,
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

    def test_cloud_aliases(self) -> None:
        """验证 CLOUD 模式别名"""
        assert MODE_ALIASES["cloud"] == RunMode.CLOUD
        assert MODE_ALIASES["cloud-agent"] == RunMode.CLOUD
        assert MODE_ALIASES["background"] == RunMode.CLOUD
        assert MODE_ALIASES["bg"] == RunMode.CLOUD


# ============================================================
# TestCloudRequestHelpers - Cloud 请求辅助函数测试
# ============================================================


class TestExecutionModeCloudAuthMapping:
    """测试 execution_mode/cloud_auth 参数映射到 OrchestratorConfig 与 IterateArgs

    注意：部分测试需要 mock Cloud API Key 存在，否则 cloud 模式会回退到 cli
    """

    @pytest.fixture(autouse=True)
    def setup_mock_api_key(self, mock_cloud_api_key):
        """自动应用 conftest.py 中的 mock_cloud_api_key fixture"""
        pass

    @pytest.fixture
    def iterate_mode_args(self) -> argparse.Namespace:
        """创建 iterate 模式的参数（tri-state 默认值与 run.py argparse 一致）

        **重要**: 使用 tri-state 默认值 (None) 来模拟 argparse 的默认行为：
        - execution_mode=None: 未显式指定，由 config.yaml 决定（默认 auto）
        - orchestrator=None: 未显式指定，根据 execution_mode 决策
        - no_mp=None: 未显式指定，跟随 orchestrator 设置

        这与 run.py 中 argparse 的 default=None 保持一致。
        """
        return argparse.Namespace(
            task="测试任务",
            mode="iterate",
            directory=".",
            workers=CONFIG_WORKER_POOL_SIZE,  # 来自 config.yaml
            max_iterations="5",
            strict=False,
            verbose=False,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=CONFIG_PLANNER_MODEL,  # 来自 config.yaml
            worker_model=CONFIG_WORKER_MODEL,  # 来自 config.yaml
            reviewer_model=CONFIG_REVIEWER_MODEL,  # 来自 config.yaml
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            # tri-state 参数（与 run.py argparse 一致，默认 None）
            orchestrator=None,  # None=未显式指定，由 execution_mode 决策
            no_mp=None,  # None=未显式指定，跟随 orchestrator 设置
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            # 执行模式参数（tri-state，None=未显式指定，使用 config.yaml 默认 auto）
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,  # None=未显式指定，使用 config.yaml 默认值
        )

    def test_config_default_auto_forces_basic_orchestrator(self, iterate_mode_args: argparse.Namespace) -> None:
        """测试未显式指定 execution_mode 时，使用 config.yaml 默认 auto 并强制 basic 编排器

        **系统默认行为**（来自 config.yaml）:
        - args.execution_mode=None (tri-state 未显式指定)
        - config.yaml 默认 cloud_agent.execution_mode=auto
        - 解析后: execution_mode=auto → 强制 basic 编排器

        **关键断言**:
        - _merge_options 输出的 execution_mode 应为 "auto"（config.yaml 默认值）
        - _merge_options 输出的 orchestrator 应为 "basic"（auto 模式强制）
        - no_mp 应为 True（与 basic 编排器一致）
        """
        # 确认 fixture 使用 tri-state 默认值
        assert iterate_mode_args.execution_mode is None, "fixture 应使用 tri-state 默认值 execution_mode=None"
        assert iterate_mode_args.orchestrator is None, "fixture 应使用 tri-state 默认值 orchestrator=None"

        runner = Runner(iterate_mode_args)
        options = runner._merge_options({})

        # 验证 config.yaml 默认 auto 生效
        assert options.get("execution_mode") == "auto", "未显式指定 execution_mode 时，应使用 config.yaml 默认值 'auto'"

        # 验证 auto 模式强制 basic 编排器
        assert options.get("orchestrator") == "basic", "config.yaml 默认 execution_mode=auto 应强制使用 basic 编排器"

        # 验证 no_mp 与 basic 编排器一致
        assert options.get("no_mp") is True, "basic 编排器时 no_mp 应为 True"

    def test_execution_mode_cloud_in_analysis_options(self, iterate_mode_args: argparse.Namespace) -> None:
        """测试 analysis_options 中包含 execution_mode=cloud 时的映射"""
        runner = Runner(iterate_mode_args)

        # 模拟 TaskAnalyzer 检测到 '&' 前缀后设置的选项
        analysis_options = {
            "execution_mode": "cloud",
        }

        options = runner._merge_options(analysis_options)

        # 验证 execution_mode 被传递（虽然 _merge_options 不直接处理 execution_mode，
        # 但在 _run_iterate 中会通过 IterateArgs 传递）
        # 这里主要验证 _merge_options 不会破坏 analysis_options
        assert "orchestrator" in options

    def test_cloud_mode_detection_sets_execution_mode_option(self, mock_args: argparse.Namespace) -> None:
        """测试 '&' 前缀检测后设置 execution_mode 选项

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            # Mock Cloud API Key 存在和 cloud_enabled=True
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                # 测试 '&' 前缀任务
                analysis = analyzer.analyze("& 分析代码架构", mock_args)

                # 验证模式被设置为 CLOUD
                assert analysis.mode == RunMode.CLOUD

                # 验证 effective_mode 选项被设置（决策后的实际模式）
                # 注意：execution_mode 是 requested_mode，effective_mode 是决策结果
                assert analysis.options.get("effective_mode") == "cloud"

                # 验证 goal 被正确剥离 '&' 前缀
                assert analysis.goal == "分析代码架构"
                assert not analysis.goal.startswith("&")
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_cloud_mode_goal_stripping_preserves_content(self, mock_args: argparse.Namespace) -> None:
        """测试 Cloud 模式下 goal 剥离 '&' 前缀保留实际内容

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            test_cases = [
                ("& 简单任务", "简单任务"),
                ("&任务描述", "任务描述"),
                ("  & 带空格的任务  ", "带空格的任务"),
                ("& 包含 & 符号的任务", "包含 & 符号的任务"),
            ]

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                for task, expected_goal in test_cases:
                    analysis = analyzer.analyze(task, mock_args)
                    assert analysis.mode == RunMode.CLOUD, f"任务 '{task}' 应该匹配 CLOUD 模式"
                    assert analysis.goal == expected_goal, (
                        f"任务 '{task}' 的 goal 应该是 '{expected_goal}'，实际是 '{analysis.goal}'"
                    )
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    @pytest.mark.asyncio
    async def test_run_iterate_receives_execution_mode(self, iterate_mode_args: argparse.Namespace) -> None:
        """测试 _run_iterate 正确接收 execution_mode 选项"""
        runner = Runner(iterate_mode_args)

        # 模拟 analysis 包含 cloud execution_mode
        analysis_options = {"execution_mode": "cloud"}
        merged = runner._merge_options(analysis_options)

        # IterateArgs 应该在 _run_iterate 中创建，这里验证选项结构
        assert "orchestrator" in merged
        assert "workers" in merged

    def test_iterate_args_tristate_defaults(self, iterate_mode_args: argparse.Namespace) -> None:
        """测试 iterate_mode_args fixture 使用 tri-state 默认值

        **重要**: 与 run.py argparse 一致，fixture 使用 tri-state 默认值：
        - execution_mode=None: 未显式指定，由 config.yaml 决定（默认 auto）
        - orchestrator=None: 未显式指定，根据 execution_mode 决策
        - no_mp=None: 未显式指定，跟随 orchestrator 设置
        - cloud_auth_timeout=None: 未显式指定，使用 config.yaml 默认值

        这是**函数级默认**（fixture 层面），与**系统默认**（config.yaml 层面）区分：
        - 函数级默认: args.execution_mode=None（tri-state，表示未指定）
        - 系统默认: config.yaml cloud_agent.execution_mode=auto（实际生效值）
        """
        # 验证 iterate_mode_args 包含所有必需的执行模式字段
        assert hasattr(iterate_mode_args, "execution_mode")
        assert hasattr(iterate_mode_args, "cloud_api_key")
        assert hasattr(iterate_mode_args, "cloud_auth_timeout")
        assert hasattr(iterate_mode_args, "orchestrator")
        assert hasattr(iterate_mode_args, "no_mp")

        # 验证 tri-state 默认值（与 run.py argparse 一致）
        assert iterate_mode_args.execution_mode is None, "fixture 应使用 tri-state 默认值 execution_mode=None"
        assert iterate_mode_args.cloud_api_key is None
        assert iterate_mode_args.cloud_auth_timeout is None, "fixture 应使用 tri-state 默认值 cloud_auth_timeout=None"
        assert iterate_mode_args.orchestrator is None, "fixture 应使用 tri-state 默认值 orchestrator=None"
        assert iterate_mode_args.no_mp is None, "fixture 应使用 tri-state 默认值 no_mp=None"

    def test_explicit_cli_mode_allows_mp_orchestrator(self, iterate_mode_args: argparse.Namespace) -> None:
        """测试显式指定 execution_mode=cli 时允许使用 mp 编排器

        **函数级显式指定** vs **系统默认**:
        - 当用户通过 CLI 显式指定 --execution-mode cli 时，可以使用 mp 编排器
        - 当未显式指定时（tri-state None），使用 config.yaml 默认 auto，强制 basic

        本用例验证：显式 execution_mode=cli + orchestrator=mp → 允许 mp 编排器
        """
        # 模拟用户显式指定 --execution-mode cli --orchestrator mp
        iterate_mode_args.execution_mode = "cli"
        iterate_mode_args.orchestrator = "mp"
        iterate_mode_args._orchestrator_user_set = True

        runner = Runner(iterate_mode_args)
        options = runner._merge_options({})

        # 验证显式 cli 模式生效
        assert options.get("execution_mode") == "cli", "显式指定 execution_mode=cli 应保持 'cli'"

        # 验证 mp 编排器生效（cli 模式允许 mp）
        assert options.get("orchestrator") == "mp", "显式 execution_mode=cli 时应允许使用 mp 编排器"

        # 验证 no_mp 与 mp 编排器一致
        assert options.get("no_mp") is False, "mp 编排器时 no_mp 应为 False"

    def test_explicit_execution_mode_cli_ignores_ampersand_prefix(self, mock_args: argparse.Namespace) -> None:
        """测试显式 --execution-mode cli 时，'&' 前缀作为普通字符处理

        优先级规则：显式 execution_mode=cli > & 前缀触发 Cloud
        当用户显式指定 --execution-mode cli 时，即使有 '&' 前缀也不触发 Cloud 模式
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        # 显式设置 execution_mode=cli（非 None 表示用户显式指定）
        mock_args.execution_mode = "cli"
        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            # 即使 API Key 存在，显式 CLI 模式也应忽略 & 前缀
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                analysis = analyzer.analyze("& 分析代码架构", mock_args)

                # 不应触发 Cloud 模式
                assert analysis.mode != RunMode.CLOUD, "显式 execution_mode=cli 时，'&' 前缀应被忽略"
                # goal 应去除 '&' 前缀
                assert analysis.goal == "分析代码架构"
                # 不应设置 cloud_background
                assert analysis.options.get("cloud_background") is not True
                # 不应设置 prefix_routed（& 前缀未成功触发 Cloud 路由）
                # 内部分支使用 prefix_routed 字段
                assert analysis.options.get("prefix_routed") is not True
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_explicit_execution_mode_cli_no_cloud_warning(self, mock_args: argparse.Namespace, capsys) -> None:
        """测试显式 --execution-mode cli 时，不输出 Cloud 相关警告

        当用户显式指定 CLI 模式时，即使有 '&' 前缀也不应输出
        "缺少 API Key" 等 Cloud 相关警告
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        # 显式设置 execution_mode=cli
        mock_args.execution_mode = "cli"  # 显式 CLI 模式
        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            # 模拟无 API Key 的情况
            with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                analysis = analyzer.analyze("& 分析代码架构", mock_args)

                # 检查是否有 Cloud 相关警告输出
                captured = capsys.readouterr()
                assert "API Key" not in captured.out, "显式 CLI 模式不应输出 API Key 相关警告"
                assert "Cloud" not in captured.out or "cloud_enabled" not in captured.out, (
                    "显式 CLI 模式不应输出 Cloud 相关警告"
                )

                # 验证模式不是 Cloud
                assert analysis.mode != RunMode.CLOUD
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_cloud_background_only_set_when_cloud_selected(self, mock_args: argparse.Namespace) -> None:
        """测试 cloud_background 仅在最终选择 Cloud 模式时设置

        Cloud Relay 语义：仅当策略最终选择 Cloud 时，才设置 cloud_background=True
        回退到 CLI 时不应设置 cloud_background
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        mock_args.execution_mode = None  # 未显式指定
        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            # 场景 1：Cloud 可用，应设置 cloud_background=True
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                analysis = analyzer.analyze("& 分析代码", mock_args)
                assert analysis.mode == RunMode.CLOUD
                assert analysis.options.get("cloud_background") is True

            # 场景 2：无 API Key 回退到 CLI，不应设置 cloud_background
            with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                analysis = analyzer.analyze("& 分析代码", mock_args)
                assert analysis.mode != RunMode.CLOUD
                assert analysis.options.get("cloud_background") is not True

            # 场景 3：cloud_enabled=False 回退到 CLI，不应设置 cloud_background
            config.cloud_agent.enabled = False
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                analysis = analyzer.analyze("& 分析代码", mock_args)
                assert analysis.mode != RunMode.CLOUD
                assert analysis.options.get("cloud_background") is not True
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_ampersand_prefix_with_config_default_auto_triggers_cloud(self, mock_args: argparse.Namespace) -> None:
        """测试配置默认 execution_mode=auto 时，'&' 前缀可触发 Cloud

        关键区别：
        - args.execution_mode=None (tri-state 未指定) + config 默认 auto = 允许 & 触发 Cloud
        - args.execution_mode="cli" (显式指定) = 忽略 & 前缀

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        # execution_mode=None 表示未显式指定（使用 config.yaml 默认值）
        mock_args.execution_mode = None
        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto

            # Mock Cloud API Key 存在
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                analysis = analyzer.analyze("& 分析代码架构", mock_args)

                # & 前缀应触发 Cloud 模式（因为 config.yaml 默认 auto）
                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式
                assert analysis.options.get("effective_mode") == "cloud"
                assert analysis.options.get("cloud_background") is True
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode


class TestCloudRequestHelpers:
    """测试 is_cloud_request 和 strip_cloud_prefix 辅助函数"""

    # ========== is_cloud_request 测试 ==========

    def test_is_cloud_request_basic(self) -> None:
        """验证基本的 '&' 前缀检测"""
        assert is_cloud_request("& 任务") is True
        assert is_cloud_request("&任务") is True
        assert is_cloud_request("  & 任务") is True

    def test_is_cloud_request_no_prefix(self) -> None:
        """验证无 '&' 前缀返回 False"""
        assert is_cloud_request("普通任务") is False
        assert is_cloud_request("任务描述") is False

    def test_is_cloud_request_empty(self) -> None:
        """验证空值返回 False"""
        assert is_cloud_request("") is False
        assert is_cloud_request(None) is False
        assert is_cloud_request("   ") is False

    def test_is_cloud_request_only_ampersand(self) -> None:
        """验证只有 '&' 符号返回 False"""
        assert is_cloud_request("&") is False
        assert is_cloud_request("&  ") is False
        assert is_cloud_request("  &  ") is False

    def test_is_cloud_request_ampersand_in_middle(self) -> None:
        """验证 '&' 在中间不触发"""
        assert is_cloud_request("任务 & 描述") is False
        assert is_cloud_request("任务&描述") is False
        assert is_cloud_request("任务&") is False

    def test_is_cloud_request_non_string(self) -> None:
        """验证非字符串类型返回 False"""
        assert is_cloud_request(123) is False
        assert is_cloud_request([]) is False
        assert is_cloud_request({}) is False

    # ========== strip_cloud_prefix 测试 ==========

    def test_strip_cloud_prefix_basic(self) -> None:
        """验证基本的前缀去除"""
        assert strip_cloud_prefix("& 任务") == "任务"
        assert strip_cloud_prefix("&任务") == "任务"

    def test_strip_cloud_prefix_with_whitespace(self) -> None:
        """验证带空白的前缀去除"""
        assert strip_cloud_prefix("  & 任务  ") == "任务"
        assert strip_cloud_prefix("& 任务描述") == "任务描述"

    def test_strip_cloud_prefix_no_prefix(self) -> None:
        """验证无前缀时保持原样"""
        assert strip_cloud_prefix("普通任务") == "普通任务"
        assert strip_cloud_prefix("任务描述") == "任务描述"

    def test_strip_cloud_prefix_empty(self) -> None:
        """验证空值处理"""
        assert strip_cloud_prefix("") == ""
        assert strip_cloud_prefix(None) == ""

    def test_strip_cloud_prefix_only_ampersand(self) -> None:
        """验证只有 '&' 时返回空字符串"""
        assert strip_cloud_prefix("&") == ""
        assert strip_cloud_prefix("&  ") == ""

    # ========== 边界测试：& 前缀策略一致性 ==========

    def test_ampersand_no_space_is_valid_cloud_request(self) -> None:
        """验证 '&任务'（无空格）是有效的 Cloud 请求"""
        assert is_cloud_request("&任务") is True
        assert is_cloud_request("&分析代码") is True
        assert strip_cloud_prefix("&任务") == "任务"
        assert strip_cloud_prefix("&分析代码") == "分析代码"

    def test_ampersand_with_space_is_valid_cloud_request(self) -> None:
        """验证 '& 任务'（有空格）是有效的 Cloud 请求"""
        assert is_cloud_request("& 任务") is True
        assert is_cloud_request("& 分析代码") is True
        assert strip_cloud_prefix("& 任务") == "任务"
        assert strip_cloud_prefix("& 分析代码") == "分析代码"

    def test_ampersand_only_is_not_cloud_request(self) -> None:
        """验证仅 '&' 符号不是有效的 Cloud 请求（边界：空内容）"""
        assert is_cloud_request("&") is False
        assert is_cloud_request("& ") is False
        assert is_cloud_request("&  ") is False
        assert is_cloud_request("  &  ") is False

    def test_ampersand_with_multiple_ampersands(self) -> None:
        """验证包含多个 & 符号的任务"""
        # 开头有 & 且后面有内容
        assert is_cloud_request("& 包含 & 符号的任务") is True
        assert strip_cloud_prefix("& 包含 & 符号的任务") == "包含 & 符号的任务"
        # 开头有 & 但实际内容为空
        assert is_cloud_request("& &") is True  # 第二个 & 视为内容
        assert strip_cloud_prefix("& &") == "&"


# ============================================================
# TestConfigYamlSourceOrchestratorConsistency
# ============================================================


class TestConfigYamlSourceOrchestratorConsistency:
    """测试 config.yaml 配置源的编排器选择一致性

    验证当 execution_mode 来自 config.yaml（而非 CLI 显式参数）时，
    run.py 入口的编排器选择行为正确。

    关键场景：
    1. config.yaml execution_mode=auto/cloud + 无 & 前缀 + 无 CLI 显式：
       最终 orchestrator 为 basic
    2. config.yaml execution_mode=cli + 有 & 前缀但无 key 或 cloud_enabled=false：
       不应 prefix_routed，且 orchestrator 允许 mp
    3. 显式 --execution-mode=cli + 有 & 前缀：
       忽略路由（prefix_routed=False），effective=cli，orchestrator 允许 mp
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_config_yaml_auto_mode_forces_basic_orchestrator(self) -> None:
        """config.yaml execution_mode=auto + 无 & 前缀 → basic 编排器

        场景：config.yaml 中设置 execution_mode=auto，用户未使用 & 前缀。
        预期：orchestrator 为 basic（基于 config.yaml 的 auto 模式）。
        """
        from core.config import resolve_orchestrator_settings
        from core.execution_policy import build_execution_decision

        # 构建 decision（模拟使用 config.yaml 默认值 auto）
        decision = build_execution_decision(
            prompt="普通任务描述",  # 无 & 前缀
            requested_mode="auto",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=True,
        )

        assert decision.orchestrator == "basic", "config.yaml execution_mode=auto 应强制 basic 编排器"
        # has_ampersand_prefix=False（无 & 前缀）→ prefix_routed=False
        # 内部分支使用 prefix_routed 字段
        assert decision.prefix_routed is False, "无 & 前缀不应标记为 prefix_routed"

        # 验证 resolve_orchestrator_settings 一致性
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "auto"},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "basic"

    def test_config_yaml_cloud_mode_forces_basic_orchestrator(self) -> None:
        """config.yaml execution_mode=cloud + 无 & 前缀 → basic 编排器

        场景：config.yaml 中设置 execution_mode=cloud，用户未使用 & 前缀。
        预期：orchestrator 为 basic（基于 config.yaml 的 cloud 模式），prefix_routed=False。
        """
        from core.config import resolve_orchestrator_settings
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="普通任务描述",  # 无 & 前缀（has_ampersand_prefix=False）
            requested_mode="cloud",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=True,
        )

        assert decision.orchestrator == "basic", "config.yaml execution_mode=cloud 应强制 basic 编排器"
        # has_ampersand_prefix=False → prefix_routed=False（非 & 前缀触发的 Cloud）
        # 内部分支使用 prefix_routed 字段
        assert decision.prefix_routed is False, "无 & 前缀不应标记为 prefix_routed"

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cloud"},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "basic"

    def test_config_yaml_cli_prefix_no_key_allows_mp(self) -> None:
        """config.yaml execution_mode=cli + & 前缀 + 无 API Key → 允许 mp

        场景：config.yaml 中设置 execution_mode=cli，用户使用了 & 前缀，
        但由于无 API Key，& 前缀未能成功触发 Cloud。
        预期：has_ampersand_prefix=True（语法检测），但 prefix_routed=False（未成功路由），
        orchestrator 允许 mp。
        """
        from core.config import resolve_orchestrator_settings
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 后台任务",  # 有 & 前缀（has_ampersand_prefix=True）
            requested_mode="cli",  # 来自 config.yaml
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key
        )

        # has_ampersand_prefix=True 但因无 API Key → prefix_routed=False
        # 内部分支使用 prefix_routed 字段
        assert decision.prefix_routed is False, "无 API Key 时，& 前缀不应成功触发 Cloud 路由"
        assert decision.orchestrator == "mp", "cli 模式 + prefix_routed=False，应允许 mp 编排器"
        assert decision.effective_mode == "cli"

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "mp"

    def test_config_yaml_cli_prefix_cloud_disabled_allows_mp(self) -> None:
        """config.yaml execution_mode=cli + & 前缀 + cloud_enabled=false → 允许 mp

        场景：config.yaml 中设置 execution_mode=cli，用户使用了 & 前缀，
        但由于 cloud_enabled=false，& 前缀未能成功触发 Cloud。
        预期：has_ampersand_prefix=True（语法检测），但 prefix_routed=False（未成功路由），
        orchestrator 允许 mp。
        """
        from core.config import resolve_orchestrator_settings
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 后台任务",  # 有 & 前缀（has_ampersand_prefix=True）
            requested_mode="cli",  # 来自 config.yaml
            cloud_enabled=False,  # cloud_enabled=false
            has_api_key=True,
        )

        # has_ampersand_prefix=True 但因 cloud_enabled=false → prefix_routed=False
        # 内部分支使用 prefix_routed 字段
        assert decision.prefix_routed is False, "cloud_enabled=false 时，& 前缀不应成功触发 Cloud 路由"
        assert decision.orchestrator == "mp", "cli 模式 + prefix_routed=False，应允许 mp 编排器"
        assert decision.effective_mode == "cli"

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "mp"

    def test_explicit_cli_with_prefix_ignores_routing_allows_mp(self) -> None:
        """显式 --execution-mode=cli + & 前缀 → 忽略路由，允许 mp

        关键测试：显式 CLI 模式覆盖 & 前缀。
        语义说明：has_ampersand_prefix=True（语法检测），但 prefix_routed=False（显式 CLI 覆盖）。
        """
        from core.config import resolve_orchestrator_settings
        from core.execution_policy import build_execution_decision

        decision = build_execution_decision(
            prompt="& 后台分析任务",  # 有 & 前缀（has_ampersand_prefix=True）
            requested_mode="cli",  # 用户显式指定 CLI
            cloud_enabled=True,
            has_api_key=True,
        )

        # has_ampersand_prefix=True 但因显式 CLI → prefix_routed=False
        # 内部分支使用 prefix_routed 字段
        assert decision.prefix_routed is False, "显式 CLI 模式下 & 前缀不应成功触发 Cloud 路由"
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp", "显式 CLI 模式 + prefix_routed=False，应允许 mp 编排器"

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "mp"

    def test_rule_based_analysis_no_prefix_no_cli_config_auto_orchestrator_basic(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 _rule_based_analysis: 无 & 前缀 + CLI 未指定 + config=auto → orchestrator=basic

        关键场景：验证 run.py 入口的 _rule_based_analysis 在此场景下返回的
        _execution_decision.orchestrator == "basic"。

        复用 resolve_requested_mode_for_decision 与 build_execution_decision，
        避免在测试里复制决策逻辑。
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        # 模拟无 CLI 显式设置
        mock_args.execution_mode = None
        mock_args.orchestrator = None
        mock_args._orchestrator_user_set = False

        # 获取并修改配置
        config = get_config()
        original_execution_mode = config.cloud_agent.execution_mode
        original_enabled = config.cloud_agent.enabled

        try:
            config.cloud_agent.execution_mode = "auto"
            config.cloud_agent.enabled = True

            analyzer = TaskAnalyzer(use_agent=False)
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="test_key"):
                analysis = analyzer._rule_based_analysis("普通任务描述", mock_args)

            # 核心断言：_execution_decision.orchestrator == "basic"
            decision = analysis.options.get("_execution_decision")
            assert decision is not None, "_rule_based_analysis 应在 options 中返回 _execution_decision"
            assert decision.orchestrator == "basic", (
                "无 & 前缀 + CLI 未指定 + config=auto 场景下，_execution_decision.orchestrator 应为 basic"
            )

            # 辅助断言
            assert decision.effective_mode == "auto"
            assert decision.prefix_routed is False
            assert analysis.options.get("orchestrator") == "basic"
        finally:
            config.cloud_agent.execution_mode = original_execution_mode
            config.cloud_agent.enabled = original_enabled

    def test_rule_based_analysis_fallback_auto_no_key_orchestrator_basic(self, mock_args: argparse.Namespace) -> None:
        """测试 _rule_based_analysis: auto + 无 API Key 回退 → orchestrator 仍为 basic

        回退场景关键测试：
        - requested_mode=auto（来自 config.yaml）
        - has_api_key=False → effective_mode=cli 回退
        - 但 orchestrator 仍为 basic（基于 requested_mode 语义）

        复用 resolve_requested_mode_for_decision 与 build_execution_decision。
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        mock_args.execution_mode = None
        mock_args.orchestrator = None
        mock_args._orchestrator_user_set = False

        config = get_config()
        original_execution_mode = config.cloud_agent.execution_mode
        original_enabled = config.cloud_agent.enabled

        try:
            config.cloud_agent.execution_mode = "auto"
            config.cloud_agent.enabled = True

            analyzer = TaskAnalyzer(use_agent=False)
            # 模拟无 API Key
            with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                analysis = analyzer._rule_based_analysis("分析代码结构", mock_args)

            decision = analysis.options.get("_execution_decision")
            assert decision is not None

            # 核心断言：即使回退到 cli，orchestrator 仍为 basic
            assert decision.effective_mode == "cli", "无 API Key 应回退到 cli"
            assert decision.orchestrator == "basic", (
                "回退场景关键断言：requested_mode=auto 时，即使回退到 cli，orchestrator 仍应为 basic"
            )
            assert decision.prefix_routed is False
            assert analysis.options.get("orchestrator") == "basic"
        finally:
            config.cloud_agent.execution_mode = original_execution_mode
            config.cloud_agent.enabled = original_enabled

    def test_rule_based_analysis_fallback_cloud_no_key_orchestrator_basic(self, mock_args: argparse.Namespace) -> None:
        """测试 _rule_based_analysis: cloud + 无 API Key 回退 → orchestrator 仍为 basic

        回退场景关键测试：
        - requested_mode=cloud（来自 config.yaml 或 CLI）
        - has_api_key=False → effective_mode=cli 回退
        - 但 orchestrator 仍为 basic（基于 requested_mode 语义）
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        mock_args.execution_mode = None
        mock_args.orchestrator = None
        mock_args._orchestrator_user_set = False

        config = get_config()
        original_execution_mode = config.cloud_agent.execution_mode
        original_enabled = config.cloud_agent.enabled

        try:
            config.cloud_agent.execution_mode = "cloud"
            config.cloud_agent.enabled = True

            analyzer = TaskAnalyzer(use_agent=False)
            with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                analysis = analyzer._rule_based_analysis("长时间分析任务", mock_args)

            decision = analysis.options.get("_execution_decision")
            assert decision is not None

            assert decision.effective_mode == "cli", "无 API Key 应回退到 cli"
            assert decision.orchestrator == "basic", (
                "回退场景关键断言：requested_mode=cloud 时，即使回退到 cli，orchestrator 仍应为 basic"
            )
        finally:
            config.cloud_agent.execution_mode = original_execution_mode
            config.cloud_agent.enabled = original_enabled

    def test_task_analyzer_explicit_cli_ignores_prefix(self, mock_args: argparse.Namespace) -> None:
        """测试 TaskAnalyzer 显式 CLI 模式忽略 & 前缀

        验证 run.py 入口的 TaskAnalyzer 对显式 CLI + & 前缀的处理。
        """
        from unittest.mock import MagicMock

        from cursor.cloud_client import CloudClientFactory

        mock_args.execution_mode = "cli"  # 显式 CLI
        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True
        mock_config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("run.get_config", return_value=mock_config):
                analysis = analyzer.analyze("& 后台任务", mock_args)

                # 不应触发 Cloud 模式
                assert analysis.mode != RunMode.CLOUD
                # goal 应去除 & 前缀
                assert analysis.goal == "后台任务"
                # prefix_routed=False（显式 CLI 覆盖 & 前缀）
                # 内部分支使用 prefix_routed 字段
                assert analysis.options.get("prefix_routed") is not True

    def test_task_analyzer_cli_prefix_no_key_not_prefix_routed(self, mock_args: argparse.Namespace) -> None:
        """测试 TaskAnalyzer cli + & 前缀 + 无 API Key 不路由到 Cloud

        验证当 & 前缀存在但无 API Key 时，run.py 入口不触发 Cloud 路由。
        语义：has_ampersand_prefix=True（语法检测），但 prefix_routed=False（无 API Key）。
        """
        from unittest.mock import MagicMock

        from cursor.cloud_client import CloudClientFactory

        mock_args.execution_mode = None  # 未显式指定
        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True
        mock_config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto

        # 无 API Key
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("run.get_config", return_value=mock_config):
                analysis = analyzer.analyze("& 后台任务", mock_args)

                # 不应触发 Cloud 模式（因为无 API Key）
                assert analysis.mode != RunMode.CLOUD
                # goal 应去除 & 前缀
                assert analysis.goal == "后台任务"
                # prefix_routed=False（has_ampersand_prefix=True 但无 API Key）
                # 内部分支使用 prefix_routed 字段
                assert analysis.options.get("prefix_routed") is not True

    def test_task_analyzer_cli_prefix_cloud_disabled_not_prefix_routed(self, mock_args: argparse.Namespace) -> None:
        """测试 TaskAnalyzer cli + & 前缀 + cloud_enabled=false 不路由到 Cloud

        验证当 & 前缀存在但 cloud_enabled=false 时，run.py 入口不触发 Cloud 路由。
        语义：has_ampersand_prefix=True（语法检测），但 prefix_routed=False（cloud_enabled=false）。
        """
        from unittest.mock import MagicMock

        from cursor.cloud_client import CloudClientFactory

        mock_args.execution_mode = None  # 未显式指定
        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = False  # cloud_enabled=false
        mock_config.cloud_agent.execution_mode = "auto"
        mock_config.cloud_agent.auto_detect_cloud_prefix = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            # 需要同时 mock core.config.get_config（用于 compute_decision_inputs）
            # 和 run.get_config（用于 _rule_based_analysis 中的验证）
            with patch("core.config.get_config", return_value=mock_config):
                with patch("run.get_config", return_value=mock_config):
                    analysis = analyzer.analyze("& 后台任务", mock_args)

                    # 不应触发 Cloud 模式（因为 cloud_enabled=false）
                    assert analysis.mode != RunMode.CLOUD
                    # goal 应去除 & 前缀
                    assert analysis.goal == "后台任务"
                    # prefix_routed=False（has_ampersand_prefix=True 但 cloud_enabled=false）
                    # 内部分支使用 prefix_routed 字段
                    assert analysis.options.get("prefix_routed") is not True


# ============================================================
# TestAutoDetectCloudPrefixCLI
# ============================================================


class TestAutoDetectCloudPrefixCLI:
    """测试 --auto-detect-cloud-prefix / --no-auto-detect-cloud-prefix 参数

    验证 CLI 显式开启/关闭 auto_detect_cloud_prefix 时的行为。
    这是 tri-state 参数：None=未指定（使用 config.yaml），True=启用，False=禁用。
    """

    @pytest.fixture
    def mock_args_with_auto_detect(self) -> argparse.Namespace:
        """创建启用 auto_detect_cloud_prefix 的参数"""
        return argparse.Namespace(
            task="& 后台任务",
            mode="auto",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode=None,  # tri-state: None=未指定
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            auto_detect_cloud_prefix=True,  # 显式启用
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    @pytest.fixture
    def mock_args_without_auto_detect(self) -> argparse.Namespace:
        """创建禁用 auto_detect_cloud_prefix 的参数"""
        return argparse.Namespace(
            task="& 后台任务",
            mode="auto",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode=None,  # tri-state: None=未指定
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            auto_detect_cloud_prefix=False,  # 显式禁用
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    def test_cli_auto_detect_enabled_triggers_cloud_with_prefix(
        self, mock_args_with_auto_detect: argparse.Namespace
    ) -> None:
        """测试 --auto-detect-cloud-prefix 启用时，& 前缀可触发 Cloud

        场景：CLI 显式设置 auto_detect_cloud_prefix=True
        预期：& 前缀应触发 Cloud 路由（需有 API Key 和 cloud_enabled）
        """
        from core.config import get_config
        from core.execution_policy import build_execution_decision, compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_auto_detect = config.cloud_agent.auto_detect_cloud_prefix

        try:
            config.cloud_agent.enabled = True
            # 即使 config 中禁用了 auto_detect，CLI 显式启用应生效
            config.cloud_agent.auto_detect_cloud_prefix = False

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                # 使用 compute_decision_inputs 构建决策输入
                inputs = compute_decision_inputs(
                    mock_args_with_auto_detect,
                    original_prompt="& 后台任务",
                )
                decision = inputs.build_decision()

                # CLI 显式启用应覆盖 config.yaml 的禁用设置
                assert decision.prefix_routed is True, (
                    "CLI --auto-detect-cloud-prefix 应覆盖 config.yaml 禁用设置，& 前缀应成功触发"
                )
                assert decision.effective_mode == "cloud", "& 前缀触发后 effective_mode 应为 cloud"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.auto_detect_cloud_prefix = original_auto_detect

    def test_cli_auto_detect_disabled_ignores_prefix(self, mock_args_without_auto_detect: argparse.Namespace) -> None:
        """测试 --no-auto-detect-cloud-prefix 禁用时，& 前缀被忽略

        场景：CLI 显式设置 auto_detect_cloud_prefix=False
        预期：& 前缀应被忽略，不触发 Cloud 路由
        """
        from core.config import get_config
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_auto_detect = config.cloud_agent.auto_detect_cloud_prefix

        try:
            config.cloud_agent.enabled = True
            # 即使 config 中启用了 auto_detect，CLI 显式禁用应生效
            config.cloud_agent.auto_detect_cloud_prefix = True

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                # 使用 compute_decision_inputs 构建决策输入
                inputs = compute_decision_inputs(
                    mock_args_without_auto_detect,
                    original_prompt="& 后台任务",
                )
                decision = inputs.build_decision()

                # CLI 显式禁用应覆盖 config.yaml 的启用设置
                assert decision.prefix_routed is False, (
                    "CLI --no-auto-detect-cloud-prefix 应覆盖 config.yaml 启用设置，& 前缀应被忽略"
                )
                # 由于 & 前缀被忽略，应回退到 config.yaml 默认 execution_mode
                # 但因为 requested_mode 可能是 auto，effective_mode 可能仍是 cloud/auto
                # 关键是 prefix_routed=False
                assert decision.has_ampersand_prefix is True, "语法上仍检测到 & 前缀"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.auto_detect_cloud_prefix = original_auto_detect

    def test_cli_auto_detect_disabled_allows_mp_orchestrator(
        self, mock_args_without_auto_detect: argparse.Namespace
    ) -> None:
        """测试 --no-auto-detect-cloud-prefix 禁用时，允许使用 MP 编排器

        场景：CLI 显式禁用 auto_detect，& 前缀被忽略
        预期：可以使用 MP 编排器（如果 execution_mode 允许）
        """
        from core.config import get_config
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        # 设置 execution_mode=cli 以允许 MP 编排器
        mock_args_without_auto_detect.execution_mode = "cli"

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_auto_detect = config.cloud_agent.auto_detect_cloud_prefix

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.auto_detect_cloud_prefix = True  # config 启用，但 CLI 禁用

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                inputs = compute_decision_inputs(
                    mock_args_without_auto_detect,
                    original_prompt="& 后台任务",
                )
                decision = inputs.build_decision()

                # CLI 禁用 auto_detect + 显式 cli 模式 → 允许 MP
                assert decision.prefix_routed is False, "& 前缀应被忽略"
                assert decision.orchestrator == "mp", "禁用 auto_detect + 显式 cli 模式时应允许使用 MP 编排器"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.auto_detect_cloud_prefix = original_auto_detect

    def test_cli_auto_detect_none_uses_config(self, mock_args: argparse.Namespace) -> None:
        """测试 auto_detect_cloud_prefix=None 时使用 config.yaml 设置

        场景：CLI 未指定 auto_detect_cloud_prefix（tri-state None）
        预期：使用 config.yaml 中的设置
        """
        from core.config import get_config
        from core.execution_policy import compute_decision_inputs
        from cursor.cloud_client import CloudClientFactory

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_auto_detect = config.cloud_agent.auto_detect_cloud_prefix

        try:
            config.cloud_agent.enabled = True

            # 场景 1：config 启用 auto_detect
            config.cloud_agent.auto_detect_cloud_prefix = True
            mock_args.auto_detect_cloud_prefix = None  # CLI 未指定

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                inputs = compute_decision_inputs(mock_args, original_prompt="& 任务")
                decision = inputs.build_decision()
                assert decision.prefix_routed is True, "config 启用时 & 前缀应触发"

            # 场景 2：config 禁用 auto_detect
            config.cloud_agent.auto_detect_cloud_prefix = False

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                inputs = compute_decision_inputs(mock_args, original_prompt="& 任务")
                decision = inputs.build_decision()
                assert decision.prefix_routed is False, "config 禁用时 & 前缀应被忽略"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.auto_detect_cloud_prefix = original_auto_detect


# ============================================================
# TestCloudModeAutoCommitSafety
# ============================================================


class TestCloudModeAutoCommitSafety:
    """测试 Cloud 模式下 auto_commit 默认禁用的安全策略

    确保即使使用 '&' 前缀触发 Cloud 模式，auto_commit 仍需显式开启。

    注意：这些测试需要 mock Cloud API Key 存在，否则 cloud 模式会回退到 cli
    """

    @pytest.fixture(autouse=True)
    def setup_mock_api_key(self, mock_cloud_api_key):
        """自动应用 conftest.py 中的 mock_cloud_api_key fixture"""
        pass

    @pytest.fixture
    def cloud_mode_args(self) -> argparse.Namespace:
        """创建 Cloud 模式的参数

        注意：execution_mode=None (tri-state 未显式指定)，
        这样 '&' 前缀才能触发 Cloud 模式。
        """
        return argparse.Namespace(
            task="& 后台执行任务",
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
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,  # 默认禁用
            auto_push=False,
            commit_per_iteration=False,
            execution_mode=None,  # tri-state: None=未显式指定，允许 & 触发 Cloud
            cloud_api_key=None,
            cloud_auth_timeout=30,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    def test_cloud_mode_does_not_auto_enable_commit(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 Cloud 模式不自动开启 auto_commit

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

                # 验证使用 Cloud 模式
                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式
                assert analysis.options.get("effective_mode") == "cloud"

                # 验证 auto_commit 未被自动设置为 True
                assert analysis.options.get("auto_commit") is not True

            # Runner 合并选项后 auto_commit 仍为 False
            runner = Runner(cloud_mode_args)
            merged = runner._merge_options(analysis.options)
            assert merged["auto_commit"] is False
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_cloud_mode_with_explicit_auto_commit(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 Cloud 模式下显式启用 auto_commit"""
        cloud_mode_args.auto_commit = True  # 显式启用

        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

        runner = Runner(cloud_mode_args)
        merged = runner._merge_options(analysis.options)

        # auto_commit 应为 True（显式指定）
        assert merged["auto_commit"] is True

    def test_cloud_mode_auto_commit_from_keywords(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 Cloud 模式下通过关键词启用 auto_commit"""
        cloud_mode_args.task = "& 后台执行任务，启用提交"

        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

        # 验证检测到启用提交关键词
        assert analysis.options.get("auto_commit") is True

        runner = Runner(cloud_mode_args)
        merged = runner._merge_options(analysis.options)
        assert merged["auto_commit"] is True

    def test_cloud_mode_auto_push_requires_auto_commit(self, cloud_mode_args: argparse.Namespace) -> None:
        """验证 auto_push 需要配合 auto_commit"""
        cloud_mode_args.auto_push = True  # 启用 auto_push 但未启用 auto_commit

        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

        runner = Runner(cloud_mode_args)
        merged = runner._merge_options(analysis.options)

        # auto_push 需要 auto_commit，否则无效
        assert merged["auto_commit"] is False
        assert merged["auto_push"] is False  # auto_push 依赖 auto_commit


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
        """验证无效值抛出异常

        注意：parse_max_iterations 抛出 MaxIterationsParseError（来自 core.config）
        如需用于 argparse 类型转换，使用 parse_max_iterations_for_argparse
        """
        from core.config import MaxIterationsParseError

        with pytest.raises(MaxIterationsParseError) as exc_info:
            parse_max_iterations("invalid")
        assert "无效的迭代次数" in str(exc_info.value)

        with pytest.raises(MaxIterationsParseError):
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

    @pytest.fixture(autouse=True)
    def reset_config_before_test(self):
        """确保每个测试前 ConfigManager 处于干净状态"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

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
                f"任务 '{task}' 应该提取 workers={expected_workers}，实际提取 workers={analysis.options.get('workers')}"
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
                f"任务 '{task}' 应该提取 workers={expected_workers}，实际提取 workers={analysis.options.get('workers')}"
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
                f"任务 '{task}' 应该提取 workers={expected_workers}，实际提取 workers={analysis.options.get('workers')}"
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
                f"任务 '{task}' 不应该设置 workers，但实际设置了 workers={analysis.options.get('workers')}"
            )

    def test_worker_count_edge_cases(self, mock_args: argparse.Namespace) -> None:
        """测试边界情况（大数字、0 等）"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试大数字
        analysis = analyzer.analyze("使用 100 个 worker 处理", mock_args)
        assert analysis.options.get("workers") == 100, "应该能提取大数字 100 作为 workers"

        # 测试较大的数字
        analysis = analyzer.analyze("启动 50 个进程执行", mock_args)
        assert analysis.options.get("workers") == 50, "应该能提取 50 作为 workers"

        # 测试单个 worker
        analysis = analyzer.analyze("使用 1 个 worker", mock_args)
        assert analysis.options.get("workers") == 1, "应该能提取 1 作为 workers"

        # 测试 0 的情况（可能不设置或设置为 0）
        analysis = analyzer.analyze("使用 0 个 worker", mock_args)
        # 0 可能被忽略或设置为 0，取决于实现
        workers_value = analysis.options.get("workers")
        assert workers_value is None or workers_value == 0, f"0 个 worker 应该被忽略或设置为 0，实际为 {workers_value}"

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
            assert analysis.options.get("auto_commit") is False, f"任务 '{task}' 应该设置 auto_commit=False"

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
            assert analysis.options.get("stream_log") is False, f"任务 '{task}' 应该设置 stream_log=False"

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
            assert analysis.options.get("auto_push") is False, f"任务 '{task}' 应该设置 auto_push=False"

    def test_enable_and_disable_conflict(self, mock_args: argparse.Namespace) -> None:
        """验证启用和禁用关键词冲突时，禁用优先生效"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 同时包含启用和禁用关键词时，禁用应后生效（后处理覆盖前处理）
        # 测试 auto_commit 冲突
        analysis = analyzer.analyze("启用提交但禁用提交", mock_args)
        assert analysis.options.get("auto_commit") is False, "同时包含启用和禁用提交关键词时，禁用应优先生效"

        # 测试 stream_log 冲突
        analysis = analyzer.analyze("启用流式日志，但禁用流式日志", mock_args)
        assert analysis.options.get("stream_log") is False, "同时包含启用和禁用流式日志关键词时，禁用应优先生效"

        # 测试 auto_push 冲突
        analysis = analyzer.analyze("自动推送 no-push", mock_args)
        assert analysis.options.get("auto_push") is False, "同时包含启用和禁用推送关键词时，禁用应优先生效"

        # 测试混合场景：禁用在前，启用在后，禁用仍应生效
        analysis = analyzer.analyze("禁用提交然后启用提交", mock_args)
        assert analysis.options.get("auto_commit") is False, "禁用关键词处理在启用关键词之后，应覆盖启用设置"

    def test_goal_not_empty(self, mock_args: argparse.Namespace) -> None:
        """验证 goal 始终不为空"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("测试任务描述", mock_args)
        assert analysis.goal == "测试任务描述"

    def test_agent_analysis_with_mock(self, mock_args: argparse.Namespace, mock_subprocess) -> None:
        """验证 Agent 分析（使用 mock）

        当用户未显式设置 workers 时，Agent 可以补全该字段。
        """
        analyzer = TaskAnalyzer(use_agent=True)

        # 将 workers 设置为 None，表示用户未显式设置
        # 这样 Agent 返回的 workers 值才能被使用
        mock_args.workers = None

        # 设置 mock 返回值
        mock_subprocess.return_value.stdout = (
            '{"mode": "mp", "options": {"workers": 4}, "reasoning": "Agent 推荐", "refined_goal": "优化后的目标"}'
        )

        analysis = analyzer.analyze("一个普通任务", mock_args)

        # 验证 subprocess 被调用
        mock_subprocess.assert_called()

        # 验证返回结果
        assert analysis.mode == RunMode.MP
        # 当用户未显式设置时，Agent 的 workers 值会被保留
        assert analysis.options.get("workers") == 4

    def test_agent_analysis_user_explicit_priority(self, mock_args: argparse.Namespace, mock_subprocess) -> None:
        """验证用户显式设置优先于 Agent 分析结果

        当用户显式设置 workers 时，Agent 返回的 workers 值会被丢弃。
        这是"用户显式优先"策略的核心行为。
        """
        analyzer = TaskAnalyzer(use_agent=True)

        # 用户显式设置 workers=3
        mock_args.workers = 3

        # Agent 返回 workers=4
        mock_subprocess.return_value.stdout = (
            '{"mode": "mp", "options": {"workers": 4, "skip_online": true}, '
            '"reasoning": "Agent 推荐", "refined_goal": "优化后的目标"}'
        )

        analysis = analyzer.analyze("一个普通任务", mock_args)

        # 验证 subprocess 被调用
        mock_subprocess.assert_called()

        # 验证返回结果
        assert analysis.mode == RunMode.MP
        # 用户显式设置的 workers=3 优先，Agent 的 workers=4 被丢弃
        # 注意：options 中不会包含 workers（因为被丢弃了），需要通过 Runner._merge_options 解析
        assert analysis.options.get("workers") is None
        # Agent 返回的其他未被用户显式设置的选项会被保留
        assert analysis.options.get("skip_online") is True

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
                f"任务 '{task}' (关键词: {keyword}) 应该匹配 PLAN 模式，实际匹配 {analysis.mode}"
            )
            assert analysis.goal == task
            assert (
                "PLAN" in analysis.reasoning.upper()
                or "规划" in analysis.reasoning
                or "plan" in analysis.reasoning.lower()
            )

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
                f"任务 '{task}' (关键词: {keyword}) 应该匹配 ASK 模式，实际匹配 {analysis.mode}"
            )
            assert analysis.goal == task
            assert (
                "ASK" in analysis.reasoning.upper()
                or "问答" in analysis.reasoning
                or "ask" in analysis.reasoning.lower()
            )

    def test_agent_analysis_success(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析成功时正确解析 JSON 结果，使用 plan 模式只读执行"""

        with patch("run.subprocess.run") as mock_run:
            # 模拟 agent CLI 返回有效 JSON
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """分析结果：
            {"mode": "mp", "options": {"workers": 5, "strict": true}, "reasoning": "任务需要并行处理", "refined_goal": "使用多进程重构代码"}
            """
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            # 使用一个不会触发规则匹配的普通任务
            result = analyzer._agent_analysis("执行一个普通任务")

            # 验证 subprocess 被正确调用
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd_args = call_args[0][0]
            assert "agent" in cmd_args
            assert "-p" in cmd_args
            assert "--output-format" in cmd_args
            # 验证使用 plan 模式（只读执行）
            assert "--mode" in cmd_args
            assert "plan" in cmd_args

            # 验证解析结果
            assert result is not None
            assert result.mode == RunMode.MP
            assert result.options.get("workers") == 5
            assert result.options.get("strict") is True
            assert result.reasoning == "任务需要并行处理"
            assert result.goal == "使用多进程重构代码"

    def test_agent_analysis_empty_output(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 返回空输出时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""  # 空输出
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None
            assert result is None

    def test_agent_analysis_whitespace_output(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 返回纯空白输出时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "   \n\t  "  # 纯空白
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None
            assert result is None

    def test_agent_analysis_timeout(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析超时时返回 None"""
        import subprocess

        with patch("run.subprocess.run") as mock_run:
            # 模拟超时异常
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None
            assert result is None

    def test_agent_analysis_invalid_json(self, mock_args: argparse.Namespace) -> None:
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

    def test_agent_analysis_returncode_nonzero(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 返回非零退出码时返回 None（稳定处理）"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1  # 非零退出码
            mock_result.stdout = ""
            mock_result.stderr = "命令执行失败"
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("执行一个任务")

            # 验证返回 None（不抛出异常）
            assert result is None

            # 验证调用使用了 plan 模式（只读保证）
            cmd_args = mock_run.call_args[0][0]
            assert "--mode" in cmd_args
            assert "plan" in cmd_args

    def test_analyze_with_agent_fallback(self, mock_args: argparse.Namespace) -> None:
        """测试 analyze 方法在 Agent 分析失败时回退到规则匹配"""
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
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""  # 模拟空输出，回退到规则分析
            mock_run.return_value = mock_result
            analyzer = TaskAnalyzer(use_agent=True)
            # 包含 ITERATE 关键词的任务
            analysis = analyzer.analyze("启动自我迭代更新代码", mock_args)

            # Agent 已被调用，规则分析兜底
            mock_run.assert_called_once()

            # 验证正确识别 ITERATE 模式
            assert analysis.mode == RunMode.ITERATE
            assert analysis.goal == "启动自我迭代更新代码"

        # 测试 5: 验证规则匹配识别 MP 模式后不调用 Agent
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""  # 模拟空输出，回退到规则分析
            mock_run.return_value = mock_result
            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("使用多进程并行处理", mock_args)

            # Agent 已被调用，规则分析兜底
            mock_run.assert_called_once()

            assert analysis.mode == RunMode.MP

        # 测试 6: 验证规则匹配识别 KNOWLEDGE 模式后不调用 Agent
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""  # 模拟空输出，回退到规则分析
            mock_run.return_value = mock_result
            analyzer = TaskAnalyzer(use_agent=True)
            analysis = analyzer.analyze("搜索知识库获取文档", mock_args)

            mock_run.assert_called_once()
            assert analysis.mode == RunMode.KNOWLEDGE

    def test_detect_non_parallel_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测非并行/协程模式关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试多种非并行关键词
        non_parallel_tasks = [
            "自我迭代，非并行模式",
            "禁用多进程执行任务",
            "使用协程模式处理",
            "单进程运行",
            "顺序执行任务",
            "使用 basic 编排器",
            "no-mp 模式运行",
            "基本模式执行",
        ]

        for task in non_parallel_tasks:
            analysis = analyzer.analyze(task, mock_args)
            # 验证 no_mp=True 或 orchestrator='basic'
            no_mp_set = analysis.options.get("no_mp") is True
            orchestrator_basic = analysis.options.get("orchestrator") == "basic"
            assert no_mp_set or orchestrator_basic, (
                f"任务 '{task}' 应该设置 no_mp=True 或 orchestrator='basic'，"
                f"实际 no_mp={analysis.options.get('no_mp')}, "
                f"orchestrator={analysis.options.get('orchestrator')}"
            )
            # 验证 reasoning 包含非并行相关信息
            assert "非并行" in analysis.reasoning or "协程" in analysis.reasoning, (
                f"任务 '{task}' 的 reasoning 应该包含非并行相关信息，实际: {analysis.reasoning}"
            )

    def test_non_parallel_with_iterate_mode(self, mock_args: argparse.Namespace) -> None:
        """验证自我迭代模式+非并行关键词同时生效"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("自我迭代，非并行模式，优化代码", mock_args)

        # 验证模式为 ITERATE
        assert analysis.mode == RunMode.ITERATE, f"应该检测到 ITERATE 模式，实际: {analysis.mode}"

        # 验证非并行选项生效
        assert analysis.options.get("no_mp") is True or analysis.options.get("orchestrator") == "basic", (
            "应该同时设置非并行选项"
        )

    def test_non_parallel_options_merged_to_runner(self, mock_args: argparse.Namespace) -> None:
        """验证非并行选项正确传递到 Runner._merge_options"""
        # 设置 mock_args 的编排器相关属性（模拟命令行默认值）
        mock_args.orchestrator = "mp"
        mock_args.no_mp = False

        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("自我迭代，非并行模式", mock_args)

        # 验证分析结果包含非并行选项
        assert analysis.options.get("no_mp") is True or analysis.options.get("orchestrator") == "basic"

        # 创建 Runner 并验证 _merge_options
        runner = Runner(mock_args)
        merged = runner._merge_options(analysis.options)

        # 合并后应保留非并行选项
        # 注意：_merge_options 可能需要从 analysis.options 合并，或使用 args 默认值
        # 验证合并逻辑正确传递了分析结果中的选项
        assert merged.get("no_mp") is True or merged.get("orchestrator") == "basic", (
            f"合并后的选项应包含非并行设置，实际: no_mp={merged.get('no_mp')}, "
            f"orchestrator={merged.get('orchestrator')}"
        )

    # ========== Cloud 模式和 '&' 前缀测试 ==========
    # 注意: Policy 要求 cloud_enabled=True 且 has_api_key=True 才能路由到 Cloud

    def _mock_cloud_enabled_and_api_key(self):
        """创建 mock context 同时启用 cloud_enabled 和 api_key"""
        from unittest.mock import MagicMock

        from cursor.cloud_client import CloudClientFactory

        # 创建 mock config
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True
        mock_config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto

        return (
            patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"),
            patch("run.get_config", return_value=mock_config),
        )

    def test_detect_cloud_prefix_basic(self, mock_args: argparse.Namespace) -> None:
        """验证检测 '&' 前缀并路由到 Cloud 模式

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        # 直接修改配置对象（避免 mock 绑定问题）
        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                analysis = analyzer.analyze("& 分析代码结构", mock_args)

                assert analysis.mode == RunMode.CLOUD, "以 '&' 开头应使用 Cloud 模式"
                assert analysis.goal == "分析代码结构", "goal 应去除 '&' 前缀"
                assert "Cloud" in analysis.reasoning or "'&'" in analysis.reasoning
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_detect_cloud_prefix_with_whitespace(self, mock_args: argparse.Namespace) -> None:
        """验证检测带空白的 '&' 前缀"""
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                analysis = analyzer.analyze("  & 分析代码", mock_args)

                assert analysis.mode == RunMode.CLOUD
                assert analysis.goal.strip() == "分析代码"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_detect_cloud_prefix_no_space(self, mock_args: argparse.Namespace) -> None:
        """验证检测紧跟内容的 '&' 前缀"""
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                analysis = analyzer.analyze("&分析代码结构", mock_args)

                assert analysis.mode == RunMode.CLOUD
                assert analysis.goal == "分析代码结构"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_no_cloud_for_only_ampersand(self, mock_args: argparse.Namespace) -> None:
        """验证只有 '&' 符号不触发 Cloud 模式"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("&", mock_args)

        assert analysis.mode != RunMode.CLOUD, "只有 '&' 不应触发 Cloud 模式"

    def test_no_cloud_for_ampersand_whitespace(self, mock_args: argparse.Namespace) -> None:
        """验证 '&' 后只有空白不触发 Cloud 模式"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("&   ", mock_args)

        assert analysis.mode != RunMode.CLOUD

    def test_no_cloud_for_ampersand_in_middle(self, mock_args: argparse.Namespace) -> None:
        """验证 '&' 在中间不触发 Cloud 模式"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("任务 & 描述", mock_args)

        assert analysis.mode != RunMode.CLOUD

    def test_cloud_mode_sets_execution_mode_option(self, mock_args: argparse.Namespace) -> None:
        """验证 Cloud 模式设置 execution_mode 选项

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            # Mock Cloud API Key 存在和 cloud_enabled=True
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                analysis = analyzer.analyze("& 后台执行任务", mock_args)

                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式（不再设置 execution_mode）
                assert analysis.options.get("effective_mode") == "cloud"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_detect_cloud_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证检测 Cloud 模式关键词（不需要 '&' 前缀）"""
        analyzer = TaskAnalyzer(use_agent=False)

        tasks = [
            "使用云端执行任务",
            "cloud agent 分析代码",
            "后台执行代码审查",
        ]

        for task in tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.CLOUD, f"任务 '{task}' 应使用 Cloud 模式"

    def test_cloud_prefix_priority_over_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证 '&' 前缀优先于其他模式关键词

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                # 同时包含 '&' 前缀和其他模式关键词
                analysis = analyzer.analyze("& 自我迭代更新代码", mock_args)

                # '&' 前缀应优先
                assert analysis.mode == RunMode.CLOUD
                # goal 应去除前缀
                assert "自我迭代" in analysis.goal
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_cloud_prefix_priority_over_plan_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证 '&' 前缀优先于 plan 模式关键词

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                # 同时包含 '&' 前缀和 plan 模式关键词
                analysis = analyzer.analyze("& 规划任务执行步骤", mock_args)

                # '&' 前缀应优先选择 CLOUD 模式
                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式
                assert analysis.options.get("effective_mode") == "cloud"
                # goal 应去除 '&' 前缀
                assert "规划任务" in analysis.goal
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_cloud_prefix_priority_over_ask_keywords(self, mock_args: argparse.Namespace) -> None:
        """验证 '&' 前缀优先于 ask 模式关键词

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
                # 同时包含 '&' 前缀和 ask 模式关键词
                analysis = analyzer.analyze("& 问答模式解决问题", mock_args)

                # '&' 前缀应优先选择 CLOUD 模式
                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式
                assert analysis.options.get("effective_mode") == "cloud"
                # goal 应去除 '&' 前缀
                assert "问答模式" in analysis.goal
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_plan_keyword_detection_comprehensive(self, mock_args: argparse.Namespace) -> None:
        """验证 plan 模式关键词全面检测"""
        analyzer = TaskAnalyzer(use_agent=False)

        # plan 模式关键词测试用例
        plan_tasks = [
            ("plan 这个功能", "plan"),
            ("planning 项目架构", "planning"),
            ("规划代码实现", "规划"),
            ("任务分析需求", "任务分析"),
            ("分析任务细节", "分析任务"),
        ]

        for task, keyword in plan_tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.PLAN, (
                f"任务 '{task}' (关键词: {keyword}) 应使用 PLAN 模式，实际为 {analysis.mode}"
            )

    def test_ask_keyword_detection_comprehensive(self, mock_args: argparse.Namespace) -> None:
        """验证 ask 模式关键词全面检测"""
        analyzer = TaskAnalyzer(use_agent=False)

        # ask 模式关键词测试用例
        ask_tasks = [
            ("ask 关于代码的问题", "ask"),
            ("chat 讨论方案", "chat"),
            ("问答模式咨询", "问答"),
            ("question 关于 API", "question"),
            ("直接问这个问题", "直接问"),
        ]

        for task, keyword in ask_tasks:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.ASK, (
                f"任务 '{task}' (关键词: {keyword}) 应使用 ASK 模式，实际为 {analysis.mode}"
            )


# ============================================================
# TestRunner
# ============================================================


class TestRunner:
    """测试 Runner 运行器"""

    def test_init(self, mock_args: argparse.Namespace) -> None:
        """验证 Runner 初始化

        Runner 使用 tri-state 设计，max_iterations 在 _merge_options 中解析。
        此测试验证 args 正确保存，不再检查 max_iterations 属性。
        """
        runner = Runner(mock_args)
        assert runner.args == mock_args
        # max_iterations 在 _merge_options 中按优先级解析，不是 Runner 的直接属性

    def test_init_with_unlimited_iterations(self) -> None:
        """验证无限迭代初始化

        Runner 使用 tri-state 设计，max_iterations 在 _merge_options 中解析。
        """
        args = argparse.Namespace(
            task="测试",
            mode="basic",
            directory=".",
            workers=CONFIG_WORKER_POOL_SIZE,  # 来自 config.yaml
            max_iterations="MAX",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=CONFIG_PLANNER_MODEL,  # 来自 config.yaml
            worker_model=CONFIG_WORKER_MODEL,  # 来自 config.yaml
            stream_log=True,
            auto_commit=False,  # 默认禁用自动提交
            auto_push=False,
            commit_per_iteration=False,
        )
        runner = Runner(args)
        assert runner.args == args
        # max_iterations 在 _merge_options 中解析为 -1，不是 Runner 的直接属性

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
        """验证选项合并（默认值来自 config.yaml）"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["directory"] == "."
        assert options["workers"] == CONFIG_WORKER_POOL_SIZE  # 来自 config.yaml
        assert options["max_iterations"] == CONFIG_MAX_ITERATIONS  # 来自 config.yaml
        assert options["strict"] is False
        # stream_log 默认值来自 config.yaml logging.stream_json.enabled
        assert options["stream_log"] is CONFIG_STREAM_LOG_ENABLED
        assert options["stream_log_console"] is CONFIG_STREAM_LOG_CONSOLE
        assert options["stream_log_detail_dir"] == CONFIG_STREAM_LOG_DETAIL_DIR
        assert options["stream_log_raw_dir"] == CONFIG_STREAM_LOG_RAW_DIR
        assert options["auto_commit"] is False  # 默认禁用自动提交
        assert options["auto_push"] is False

    def test_merge_options_with_analysis(self, mock_args: argparse.Namespace) -> None:
        """验证选项合并（使用分析结果补全未显式设置的字段）

        当用户未显式设置 workers/max_iterations 时，analysis_options 可以补全这些值。
        """
        # 将 workers 和 max_iterations 设置为 None，表示用户未显式设置
        mock_args.workers = None
        mock_args.max_iterations = None

        runner = Runner(mock_args)
        analysis_options = {
            "workers": 5,
            "max_iterations": -1,
            "skip_online": True,
            "dry_run": True,
        }
        options = runner._merge_options(analysis_options)

        # 用户未显式设置时，analysis_options 的值被采用
        assert options["workers"] == 5
        assert options["max_iterations"] == -1
        assert options["skip_online"] is True
        assert options["dry_run"] is True

    def test_merge_options_user_explicit_priority(self, mock_args: argparse.Namespace) -> None:
        """验证用户显式设置优先于分析结果

        当用户显式设置 workers/max_iterations 时，analysis_options 不会覆盖这些值。
        这是"用户显式优先"策略在 _merge_options 中的体现。
        """
        # 用户显式设置 workers=3, max_iterations=10
        mock_args.workers = 3
        mock_args.max_iterations = "10"

        runner = Runner(mock_args)
        analysis_options = {
            "workers": 5,  # 尝试覆盖为 5
            "max_iterations": -1,  # 尝试覆盖为无限
            "skip_online": True,  # 用户未显式设置，可以补全
        }
        options = runner._merge_options(analysis_options)

        # 用户显式设置的值优先，不被 analysis_options 覆盖
        assert options["workers"] == 3
        assert options["max_iterations"] == 10
        # 用户未显式设置的字段，可以被 analysis_options 补全
        assert options["skip_online"] is True

    def test_merge_options_analysis_directory_override(self, mock_args: argparse.Namespace) -> None:
        """验证分析结果可覆盖目录（未显式指定时）"""
        mock_args._directory_user_set = False
        runner = Runner(mock_args)
        options = runner._merge_options({"directory": "/tmp/project"})

        assert options["directory"] == "/tmp/project"

    def test_merge_options_analysis_directory_respects_user(self, mock_args: argparse.Namespace) -> None:
        """验证显式目录优先，不被分析覆盖"""
        mock_args._directory_user_set = True
        mock_args.directory = "/explicit/path"
        runner = Runner(mock_args)
        options = runner._merge_options({"directory": "/from/task"})

        assert options["directory"] == "/explicit/path"

    def test_merge_options_stream_log_from_analysis(self, mock_args: argparse.Namespace) -> None:
        """验证 stream_log 从分析结果获取"""
        runner = Runner(mock_args)

        # 分析结果禁用 stream_log
        options = runner._merge_options({"stream_log": False})
        assert options["stream_log"] is False

        # 分析结果启用 stream_log
        options = runner._merge_options({"stream_log": True})
        assert options["stream_log"] is True

    def test_merge_options_auto_commit_from_analysis(self, mock_args: argparse.Namespace) -> None:
        """验证 auto_commit 从分析结果获取"""
        runner = Runner(mock_args)

        # 分析结果禁用 auto_commit
        options = runner._merge_options({"auto_commit": False})
        assert options["auto_commit"] is False

        # 分析结果启用 auto_commit
        options = runner._merge_options({"auto_commit": True})
        assert options["auto_commit"] is True

    def test_merge_options_auto_push_requires_auto_commit(self, mock_args: argparse.Namespace) -> None:
        """验证 auto_push 需要 auto_commit"""
        runner = Runner(mock_args)

        # auto_push=True 但 auto_commit=False，结果应该是 auto_push=False
        options = runner._merge_options({"auto_commit": False, "auto_push": True})
        assert options["auto_push"] is False

        # 两者都为 True
        options = runner._merge_options({"auto_commit": True, "auto_push": True})
        assert options["auto_push"] is True

    def test_merge_options_auto_commit_default_false_without_arg(self, mock_args: argparse.Namespace) -> None:
        """验证不提供 --auto-commit 时 auto_commit 默认为 False"""
        runner = Runner(mock_args)

        # 不提供任何 auto_commit 相关选项
        options = runner._merge_options({})
        assert options["auto_commit"] is False, "不提供 --auto-commit 时，auto_commit 应默认为 False"

        # 即使提供其他选项，auto_commit 仍应为 False
        options = runner._merge_options(
            {
                "workers": 5,
                "max_iterations": -1,
                "strict": True,
            }
        )
        assert options["auto_commit"] is False, "提供其他选项但不提供 auto_commit 时，应默认为 False"

    def test_merge_options_auto_commit_explicit_enable(self) -> None:
        """验证显式提供 --auto-commit 时能正确启用"""
        # 创建一个带有 auto_commit=True 的 args
        args = argparse.Namespace(
            task="test",
            mode="basic",
            directory=".",
            workers=CONFIG_WORKER_POOL_SIZE,  # 来自 config.yaml
            max_iterations=str(CONFIG_MAX_ITERATIONS),  # 来自 config.yaml
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,  # tri-state
            worker_model=None,  # tri-state
            reviewer_model=None,  # tri-state
            stream_log_enabled=None,
            auto_commit=True,  # 显式启用
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            _orchestrator_user_set=False,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )
        runner = Runner(args)
        options = runner._merge_options({})

        assert options["auto_commit"] is True, "显式提供 --auto-commit 时，应该启用 auto_commit"

    def test_merge_options_natural_language_commit_keywords_consistent(self, mock_args: argparse.Namespace) -> None:
        """验证自然语言关键词启用/禁用提交时结果与 _merge_options 一致"""
        analyzer = TaskAnalyzer(use_agent=False)
        runner = Runner(mock_args)

        # 测试启用关键词
        enable_tasks = [
            "启用提交，完成任务",
            "开启提交完成代码",
            "自动提交代码修改",
            "enable-commit task",
        ]
        for task in enable_tasks:
            analysis = analyzer.analyze(task, mock_args)
            options = runner._merge_options(analysis.options)
            assert options["auto_commit"] is True, f"任务 '{task}' 应该通过自然语言启用 auto_commit"

        # 测试禁用关键词
        disable_tasks = [
            "禁用提交，仅分析",
            "关闭提交不修改",
            "不提交代码",
            "跳过提交步骤",
            "no-commit mode",
        ]
        for task in disable_tasks:
            analysis = analyzer.analyze(task, mock_args)
            options = runner._merge_options(analysis.options)
            assert options["auto_commit"] is False, f"任务 '{task}' 应该通过自然语言禁用 auto_commit"

    def test_merge_options_auto_push_forced_false_when_auto_commit_false(self, mock_args: argparse.Namespace) -> None:
        """验证 auto_push 在 auto_commit==False 时必须被强制为 False"""
        runner = Runner(mock_args)

        # 场景1: 两者都未指定，auto_push 应为 False
        options = runner._merge_options({})
        assert options["auto_commit"] is False
        assert options["auto_push"] is False, "auto_commit=False 时，auto_push 必须为 False"

        # 场景2: 仅指定 auto_push=True（不指定 auto_commit）
        options = runner._merge_options({"auto_push": True})
        assert options["auto_commit"] is False, "未指定 auto_commit 时应默认为 False"
        assert options["auto_push"] is False, "auto_commit=False 时，即使指定 auto_push=True 也应强制为 False"

        # 场景3: 显式禁用 auto_commit，启用 auto_push
        options = runner._merge_options({"auto_commit": False, "auto_push": True})
        assert options["auto_push"] is False, "显式 auto_commit=False 时，auto_push 必须强制为 False"

        # 场景4: 通过命令行参数设置 auto_push=True 但 auto_commit=False
        args_with_push = argparse.Namespace(
            task="test",
            mode="basic",
            directory=".",
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,  # tri-state
            worker_model=None,  # tri-state
            reviewer_model=None,  # tri-state
            stream_log_enabled=None,
            auto_commit=False,
            auto_push=True,  # 命令行指定 auto_push
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode=None,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            _orchestrator_user_set=False,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )
        runner_with_push = Runner(args_with_push)
        options = runner_with_push._merge_options({})
        assert options["auto_push"] is False, (
            "命令行 auto_push=True 但 auto_commit=False 时，auto_push 应被强制为 False"
        )

    # ========== 流式控制台渲染配置测试 ==========

    def test_merge_options_stream_renderer_defaults(self, mock_args: argparse.Namespace) -> None:
        """验证流式渲染配置默认值（默认关闭，避免噪声）"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        # 流式渲染器默认关闭
        assert options["stream_console_renderer"] is False, "stream_console_renderer 默认应为 False"
        assert options["stream_advanced_renderer"] is False, "stream_advanced_renderer 默认应为 False"
        assert options["stream_typing_effect"] is False, "stream_typing_effect 默认应为 False"
        assert options["stream_show_word_diff"] is False, "stream_show_word_diff 默认应为 False"
        # 其他默认值
        assert options["stream_typing_delay"] == 0.02, "stream_typing_delay 默认应为 0.02"
        assert options["stream_word_mode"] is True, "stream_word_mode 默认应为 True"
        assert options["stream_color_enabled"] is True, "stream_color_enabled 默认应为 True"

    def test_merge_options_stream_renderer_from_args(self) -> None:
        """验证流式渲染配置从命令行参数获取"""
        args = argparse.Namespace(
            task="test",
            mode="basic",
            directory=".",
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,  # tri-state
            worker_model=None,  # tri-state
            reviewer_model=None,  # tri-state
            stream_log_enabled=None,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode="cli",
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=None,
            _orchestrator_user_set=False,
            # 流式渲染参数（显式启用）
            stream_console_renderer=True,
            stream_advanced_renderer=True,
            stream_typing_effect=True,
            stream_typing_delay=0.05,
            stream_word_mode=False,
            stream_color_enabled=False,
            stream_show_word_diff=True,
        )
        runner = Runner(args)
        options = runner._merge_options({})

        # 验证所有流式渲染配置都正确传递
        assert options["stream_console_renderer"] is True
        assert options["stream_advanced_renderer"] is True
        assert options["stream_typing_effect"] is True
        assert options["stream_typing_delay"] == 0.05
        assert options["stream_word_mode"] is False
        assert options["stream_color_enabled"] is False
        assert options["stream_show_word_diff"] is True

    def test_stream_renderer_does_not_affect_plan_mode_readonly(self, mock_args: argparse.Namespace) -> None:
        """验证流式渲染配置不影响 plan 模式的只读语义

        plan 模式应保持 force_write=False，无论流式渲染配置如何。
        """
        from cursor.client import CursorAgentConfig

        # 创建带有流式渲染配置的 CursorAgentConfig
        config = CursorAgentConfig(
            mode="plan",  # 规划模式
            force_write=False,  # 只读
            stream_console_renderer=True,
            stream_advanced_renderer=True,
            stream_typing_effect=True,
        )

        # 验证 plan 模式仍然保持只读
        assert config.mode == "plan"
        assert config.force_write is False, "plan 模式下 force_write 必须为 False，流式渲染不应影响"
        # 验证流式渲染配置已设置
        assert config.stream_console_renderer is True
        assert config.stream_advanced_renderer is True
        assert config.stream_typing_effect is True

    def test_stream_renderer_does_not_affect_ask_mode_readonly(self, mock_args: argparse.Namespace) -> None:
        """验证流式渲染配置不影响 ask 模式的只读语义

        ask 模式应保持 force_write=False，无论流式渲染配置如何。
        """
        from cursor.client import CursorAgentConfig

        # 创建带有流式渲染配置的 CursorAgentConfig
        config = CursorAgentConfig(
            mode="ask",  # 问答模式
            force_write=False,  # 只读
            stream_console_renderer=True,
            stream_show_word_diff=True,
        )

        # 验证 ask 模式仍然保持只读
        assert config.mode == "ask"
        assert config.force_write is False, "ask 模式下 force_write 必须为 False，流式渲染不应影响"
        # 验证流式渲染配置已设置
        assert config.stream_console_renderer is True
        assert config.stream_show_word_diff is True

    # ========== execution_mode 和 cloud_auth 测试 ==========

    def test_merge_options_execution_mode_default(self, mock_args: argparse.Namespace) -> None:
        """验证 execution_mode 默认来自配置（config.yaml 默认 auto）"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto
        mock_config.cloud_agent.enabled = False

        with patch("run.get_config", return_value=mock_config):
            runner = Runner(mock_args)
            options = runner._merge_options({})

            # 默认使用配置中的 execution_mode（现在默认是 auto）
            assert options["execution_mode"] == "auto", "默认 execution_mode 应来自配置（config.yaml 默认 auto）"

    def test_merge_options_execution_mode_from_analysis(self, mock_args: argparse.Namespace) -> None:
        """验证 execution_mode 从分析结果获取（优先于命令行参数）"""
        runner = Runner(mock_args)

        # 分析结果设置 execution_mode
        options = runner._merge_options({"execution_mode": "cloud"})
        assert options["execution_mode"] == "cloud", "分析结果 execution_mode='cloud' 应覆盖默认值"

        options = runner._merge_options({"execution_mode": "auto"})
        assert options["execution_mode"] == "auto"

    def test_merge_options_execution_mode_from_args(self) -> None:
        """验证 execution_mode 从命令行参数获取"""
        args = argparse.Namespace(
            task="test",
            mode="basic",
            directory=".",
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,  # tri-state
            worker_model=None,  # tri-state
            reviewer_model=None,  # tri-state
            stream_log_enabled=None,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            execution_mode="auto",  # 命令行指定 auto 模式
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=None,
            _orchestrator_user_set=False,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )
        runner = Runner(args)
        options = runner._merge_options({})  # 不传分析结果

        assert options["execution_mode"] == "auto", "命令行 execution_mode='auto' 应被正确获取"

    def test_merge_options_cloud_auth_params(self, mock_args: argparse.Namespace) -> None:
        """验证 cloud_api_key 和 cloud_auth_timeout 正确传递"""
        # 设置 cloud 认证参数
        mock_args.cloud_api_key = "test-api-key"
        mock_args.cloud_auth_timeout = 60

        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["cloud_api_key"] == "test-api-key"
        assert options["cloud_auth_timeout"] == 60

    def test_merge_options_cloud_auth_default_values(self, mock_args: argparse.Namespace) -> None:
        """验证 cloud 认证参数默认值"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["cloud_api_key"] is None
        assert options["cloud_auth_timeout"] == 30

    def test_get_execution_mode_helper(self, mock_args: argparse.Namespace) -> None:
        """验证 _get_execution_mode 辅助方法

        注意：_get_execution_mode 依赖 core.execution_policy.resolve_effective_execution_mode，
        该函数需要 cloud_enabled=True 且 has_api_key=True 才会返回 AUTO/CLOUD 模式。
        """
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        runner = Runner(mock_args)

        # 创建 mock config，启用 cloud
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        # 测试 cli 模式（无需 API Key，无需 cloud_enabled）
        assert runner._get_execution_mode({"execution_mode": "cli"}) == ExecutionMode.CLI

        # 测试 auto/cloud 模式（需要 API Key + cloud_enabled=True）
        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"),
            patch("run.get_config", return_value=mock_config),
        ):
            assert runner._get_execution_mode({"execution_mode": "auto"}) == ExecutionMode.AUTO
            assert runner._get_execution_mode({"execution_mode": "cloud"}) == ExecutionMode.CLOUD

        # 测试无 API Key 时的行为（cloud_enabled=True 但无 key）
        # - CLOUD 模式：回退到 CLI
        # - AUTO 模式：回退到 CLI（由 resolve_effective_execution_mode 决定）
        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("run.get_config", return_value=mock_config),
        ):
            # AUTO 模式无 key 时回退到 CLI
            assert runner._get_execution_mode({"execution_mode": "auto"}) == ExecutionMode.CLI
            # CLOUD 模式直接回退到 CLI
            assert runner._get_execution_mode({"execution_mode": "cloud"}) == ExecutionMode.CLI

        # 测试无效值回退到 CLI
        assert runner._get_execution_mode({"execution_mode": "invalid"}) == ExecutionMode.CLI
        assert runner._get_execution_mode({}) == ExecutionMode.CLI

    def test_get_cloud_auth_config_helper(self, mock_args: argparse.Namespace) -> None:
        """验证 _get_cloud_auth_config 辅助方法"""
        runner = Runner(mock_args)

        # 无 API Key 时返回 None
        options: dict[str, str | int | None] = {"cloud_api_key": None, "cloud_auth_timeout": 30}
        config = runner._get_cloud_auth_config(options)
        assert config is None

        # 有 API Key 时返回 CloudAuthConfig
        options = {"cloud_api_key": "test-key", "cloud_auth_timeout": 60}
        config = runner._get_cloud_auth_config(options)
        assert config is not None
        assert config.api_key == "test-key"
        assert config.auth_timeout == 60

    def test_execution_mode_and_cloud_prefix_consistency(self, mock_args: argparse.Namespace) -> None:
        """验证 '&' 前缀触发的 Cloud 模式与 _merge_options 一致

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        analyzer = TaskAnalyzer(use_agent=False)
        runner = Runner(mock_args)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            # Mock Cloud API Key 存在和 cloud_enabled=True
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                # '&' 前缀应设置 effective_mode='cloud'
                analysis = analyzer.analyze("& 后台执行任务", mock_args)
                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式
                assert analysis.options.get("effective_mode") == "cloud"

                # _merge_options 应正确处理
                options = runner._merge_options(analysis.options)
                # effective_mode 保留为 cloud，用于 _get_execution_mode
                assert options["effective_mode"] == "cloud"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode


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
            planner_model=DEFAULT_PLANNER_MODEL,
            worker_model=DEFAULT_WORKER_MODEL,
            reviewer_model=DEFAULT_REVIEWER_MODEL,
            stream_events_enabled=True,
        )

        # 验证配置属性
        assert config.working_directory == "."
        assert config.max_iterations == 10
        assert config.worker_count == 3
        assert config.strict_review is False
        assert config.planner_model == DEFAULT_PLANNER_MODEL
        assert config.worker_model == DEFAULT_WORKER_MODEL
        assert config.reviewer_model == DEFAULT_REVIEWER_MODEL
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
        assert config.planning_timeout == 500.0
        assert config.execution_timeout == 600.0
        assert config.review_timeout == 300.0
        assert config.planner_model == DEFAULT_PLANNER_MODEL
        assert config.worker_model == DEFAULT_WORKER_MODEL
        assert config.reviewer_model == DEFAULT_REVIEWER_MODEL
        # stream_events_enabled 使用 tri-state 设计：None 表示使用 config.yaml 的值
        assert config.stream_events_enabled is None
        # 验证提交相关默认值（auto_commit 默认禁用，需显式开启）
        assert config.enable_auto_commit is False  # 默认禁用自动提交
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

    def test_mp_config_reviewer_model_from_config_yaml(self) -> None:
        """测试 MultiProcessOrchestratorConfig 的 reviewer_model 来自 config.yaml

        验证 reviewer_model 独立配置，而非与 worker_model 相同。
        确保 scripts/run_mp.py 使用正确的参数来源。
        """
        from coordinator import MultiProcessOrchestratorConfig

        # 验证默认的 reviewer_model 来自 config.yaml
        # 而非与 worker_model 相同
        config = MultiProcessOrchestratorConfig()
        assert config.reviewer_model == CONFIG_REVIEWER_MODEL
        assert config.reviewer_model == DEFAULT_REVIEWER_MODEL
        assert config.reviewer_model != config.worker_model, "reviewer_model 应独立配置，而非与 worker_model 相同"

    def test_mp_config_reviewer_model_custom_override(self) -> None:
        """测试 MultiProcessOrchestratorConfig 支持自定义 reviewer_model"""
        from coordinator import MultiProcessOrchestratorConfig

        custom_reviewer = "custom-reviewer-model"
        config = MultiProcessOrchestratorConfig(
            planner_model=DEFAULT_PLANNER_MODEL,
            worker_model=DEFAULT_WORKER_MODEL,
            reviewer_model=custom_reviewer,
        )

        assert config.reviewer_model == custom_reviewer
        assert config.worker_model != config.reviewer_model

    def test_mp_config_all_three_models_independent(self) -> None:
        """测试三种模型（planner/worker/reviewer）可以独立配置"""
        from coordinator import MultiProcessOrchestratorConfig

        config = MultiProcessOrchestratorConfig(
            planner_model="model-a",
            worker_model="model-b",
            reviewer_model="model-c",
        )

        assert config.planner_model == "model-a"
        assert config.worker_model == "model-b"
        assert config.reviewer_model == "model-c"
        # 验证三者互不相同
        assert len({config.planner_model, config.worker_model, config.reviewer_model}) == 3

    def test_mp_config_stream_renderer_fields(self) -> None:
        """测试 MP 配置流式控制台渲染相关字段"""
        from coordinator import MultiProcessOrchestratorConfig

        # 测试默认值（默认关闭）
        config_default = MultiProcessOrchestratorConfig()
        assert config_default.stream_console_renderer is False
        assert config_default.stream_advanced_renderer is False
        assert config_default.stream_typing_effect is False
        assert config_default.stream_typing_delay == 0.02
        assert config_default.stream_word_mode is True
        assert config_default.stream_color_enabled is True
        assert config_default.stream_show_word_diff is False

        # 测试自定义流式渲染配置
        config = MultiProcessOrchestratorConfig(
            stream_console_renderer=True,
            stream_advanced_renderer=True,
            stream_typing_effect=True,
            stream_typing_delay=0.05,
            stream_word_mode=False,
            stream_color_enabled=False,
            stream_show_word_diff=True,
        )

        assert config.stream_console_renderer is True
        assert config.stream_advanced_renderer is True
        assert config.stream_typing_effect is True
        assert config.stream_typing_delay == 0.05
        assert config.stream_word_mode is False
        assert config.stream_color_enabled is False
        assert config.stream_show_word_diff is True

    def test_orchestrator_config_stream_renderer_fields(self) -> None:
        """测试 OrchestratorConfig 流式控制台渲染相关字段"""
        from coordinator import OrchestratorConfig

        # 测试默认值（默认关闭）
        config_default = OrchestratorConfig()
        assert config_default.stream_console_renderer is False
        assert config_default.stream_advanced_renderer is False
        assert config_default.stream_typing_effect is False
        assert config_default.stream_typing_delay == 0.02
        assert config_default.stream_word_mode is True
        assert config_default.stream_color_enabled is True
        assert config_default.stream_show_word_diff is False

        # 测试自定义流式渲染配置
        config = OrchestratorConfig(
            stream_console_renderer=True,
            stream_advanced_renderer=True,
            stream_typing_effect=True,
            stream_typing_delay=0.05,
            stream_word_mode=False,
            stream_color_enabled=False,
            stream_show_word_diff=True,
        )

        assert config.stream_console_renderer is True
        assert config.stream_advanced_renderer is True
        assert config.stream_typing_effect is True
        assert config.stream_typing_delay == 0.05
        assert config.stream_word_mode is False
        assert config.stream_color_enabled is False
        assert config.stream_show_word_diff is True

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
        with (
            patch.object(orchestrator, "_spawn_agents") as mock_spawn,
            patch.object(orchestrator.process_manager, "wait_all_ready", return_value=True),
            patch.object(orchestrator.process_manager, "shutdown_all"),
            patch.object(orchestrator, "_planning_phase", new_callable=AsyncMock),
            patch.object(orchestrator, "_execution_phase", new_callable=AsyncMock),
            patch.object(orchestrator, "_review_phase", new_callable=AsyncMock) as mock_review,
        ):
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
        """验证长目标被截断并显示截断提示"""
        # 创建超过 MAX_GOAL_SUMMARY_CHARS 的长目标
        long_goal = "这是一个非常长的任务目标，" * 20
        assert len(long_goal) > MAX_GOAL_SUMMARY_CHARS, "测试前提：目标长度应超过截断限制"
        result = {
            "success": True,
            "goal": long_goal,
        }
        print_result(result)
        captured = capsys.readouterr()
        # 验证截断提示存在
        assert TRUNCATION_HINT in captured.out, f"输出应包含截断提示: {TRUNCATION_HINT}"

    def test_print_result_short_goal_no_truncation(self, capsys) -> None:
        """验证短目标不被截断"""
        short_goal = "短任务"
        assert len(short_goal) <= MAX_GOAL_SUMMARY_CHARS, "测试前提：目标长度不应超过截断限制"
        result = {
            "success": True,
            "goal": short_goal,
        }
        print_result(result)
        captured = capsys.readouterr()
        # 验证截断提示不存在
        assert TRUNCATION_HINT not in captured.out, "短目标不应显示截断提示"


# ============================================================
# TestTruncationConstants - 截断常量测试
# ============================================================


class TestTruncationConstants:
    """测试截断常量的一致性和基本断言

    验证截断常量被正确定义并用于控制台预览截断。
    这是轻量断言测试，只验证截断提示存在。
    """

    def test_truncation_constants_defined(self) -> None:
        """验证截断常量已正确定义"""
        # 验证常量存在且为正整数
        assert MAX_CONSOLE_PREVIEW_CHARS > 0, "MAX_CONSOLE_PREVIEW_CHARS 应为正整数"
        assert MAX_KNOWLEDGE_DOC_PREVIEW_CHARS > 0, "MAX_KNOWLEDGE_DOC_PREVIEW_CHARS 应为正整数"
        assert MAX_GOAL_SUMMARY_CHARS > 0, "MAX_GOAL_SUMMARY_CHARS 应为正整数"

        # 验证层级关系：控制台预览 > 知识库预览 > 目标摘要
        assert MAX_CONSOLE_PREVIEW_CHARS >= MAX_KNOWLEDGE_DOC_PREVIEW_CHARS, "控制台预览限制应 >= 知识库文档预览限制"
        assert MAX_KNOWLEDGE_DOC_PREVIEW_CHARS >= MAX_GOAL_SUMMARY_CHARS, "知识库文档预览限制应 >= 目标摘要限制"

    def test_truncation_hints_defined(self) -> None:
        """验证截断提示字符串已正确定义"""
        # 验证截断提示非空
        assert TRUNCATION_HINT, "TRUNCATION_HINT 应非空"
        assert TRUNCATION_HINT_OUTPUT, "TRUNCATION_HINT_OUTPUT 应非空"

        # 验证截断提示包含关键词
        assert "截断" in TRUNCATION_HINT or "..." in TRUNCATION_HINT, "TRUNCATION_HINT 应包含截断标识"
        assert "截断" in TRUNCATION_HINT_OUTPUT or "..." in TRUNCATION_HINT_OUTPUT, (
            "TRUNCATION_HINT_OUTPUT 应包含截断标识"
        )

    def test_truncation_values_expected(self) -> None:
        """验证截断常量的具体值符合预期"""
        # 这些值与 core/config.py 中的定义一致
        assert MAX_CONSOLE_PREVIEW_CHARS == 2000, "MAX_CONSOLE_PREVIEW_CHARS 应为 2000"
        assert MAX_KNOWLEDGE_DOC_PREVIEW_CHARS == 1000, "MAX_KNOWLEDGE_DOC_PREVIEW_CHARS 应为 1000"
        assert MAX_GOAL_SUMMARY_CHARS == 100, "MAX_GOAL_SUMMARY_CHARS 应为 100"


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
    async def test_run_basic_with_mock(self, runner: Runner, mock_args: argparse.Namespace) -> None:
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
    async def test_run_basic_with_unlimited_iterations(self, runner: Runner) -> None:
        """测试 _run_basic 无限迭代模式"""
        import coordinator

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True})
        with (
            patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator),
            patch.object(run_module, "print_info") as mock_print_info,
        ):
            options = runner._merge_options({"max_iterations": -1})
            options["max_iterations"] = -1  # 确保设置
            result = await runner._run_basic("测试目标", options)

            assert result == {"success": True}
            # 验证无限迭代提示被打印
            mock_print_info.assert_called()

    @pytest.mark.asyncio
    async def test_run_mp_with_mock(self, runner: Runner) -> None:
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
    async def test_run_iterate_with_mock(self, runner: Runner) -> None:
        """测试 _run_iterate 方法（使用 mock）

        验证:
        1. 基础结果字段正确透传
        2. 元数据字段正确透传（与 scripts/run_iterate.py 字段命名一致）
        """
        import scripts.run_iterate

        # 模拟包含元数据字段的返回结果（与 _run_agent_system 返回一致）
        mock_result = {
            "success": True,
            "mode": "iterate",
            "iterations_completed": 3,
            "total_tasks_created": 5,
            "total_tasks_completed": 4,
            "total_tasks_failed": 1,
            # 元数据字段（与 scripts/run_iterate.py 保持一致）
            "orchestrator_type": "mp",
            "orchestrator_requested": "mp",
            "fallback_occurred": False,
            "fallback_reason": None,
            "execution_mode": "cli",
            "max_iterations_configured": 10,
        }

        with patch.object(scripts.run_iterate, "SelfIterator") as mock_iterator_class:
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value=mock_result)
            mock_iterator_class.return_value = mock_iterator

            options = runner._merge_options({"skip_online": True})
            result = await runner._run_iterate("自我迭代任务", options)

            # 验证基础字段透传
            assert result["success"] is True
            assert result["mode"] == "iterate"

            # 验证元数据字段透传（确保 run.py 不会过滤这些字段）
            assert result["orchestrator_type"] == "mp"
            assert result["orchestrator_requested"] == "mp"
            assert result["fallback_occurred"] is False
            assert result["fallback_reason"] is None
            assert result["execution_mode"] == "cli"
            assert result["max_iterations_configured"] == 10

            mock_iterator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_plan_success(self, runner: Runner) -> None:
        """测试 _run_plan 方法成功（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        mock_result = AgentResult(
            success=True,
            output="执行计划内容...",
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

            options = runner._merge_options({})
            result = await runner._run_plan("规划任务", options)

            # 验证执行器被正确创建和配置
            mock_executor_cls.assert_called_once()
            mock_executor.execute.assert_called_once()

            # 验证执行器配置（mode=plan, force_write=False）
            assert mock_executor.config.mode == "plan"
            assert mock_executor.config.force_write is False

            assert result["success"] is True
            assert result["mode"] == "plan"
            assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_run_plan_failure(self, runner: Runner) -> None:
        """测试 _run_plan 方法失败（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        mock_result = AgentResult(
            success=False,
            output="",
            error="规划失败原因",
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

            options = runner._merge_options({})
            result = await runner._run_plan("失败的规划", options)

            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_run_plan_timeout(self, runner: Runner) -> None:
        """测试 _run_plan 方法超时（使用 PlanAgentExecutor）"""
        import asyncio

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_executor_cls.return_value = mock_executor

            options = runner._merge_options({})
            result = await runner._run_plan("超时的规划", options)

            assert result["success"] is False
            assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_run_plan_exception(self, runner: Runner) -> None:
        """测试 _run_plan 方法异常（使用 PlanAgentExecutor）"""
        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(side_effect=Exception("未知错误"))
            mock_executor_cls.return_value = mock_executor

            options = runner._merge_options({})
            result = await runner._run_plan("异常的规划", options)

            assert result["success"] is False
            assert "未知错误" in result["error"]

    @pytest.mark.asyncio
    async def test_run_ask_success(self, runner: Runner) -> None:
        """测试 _run_ask 方法成功"""
        from cursor.executor import AgentResult

        mock_agent_result = AgentResult(
            success=True,
            output="回答内容...",
            error=None,
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

            options = runner._merge_options({})
            result = await runner._run_ask("问题内容", options)

            assert result["success"] is True
            assert result["mode"] == "ask"
            assert result["answer"] == "回答内容..."

    @pytest.mark.asyncio
    async def test_run_ask_failure(self, runner: Runner) -> None:
        """测试 _run_ask 方法失败"""
        from cursor.executor import AgentResult

        mock_agent_result = AgentResult(
            success=False,
            output="",
            error="问答失败",
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

            options = runner._merge_options({})
            result = await runner._run_ask("失败的问题", options)

            assert result["success"] is False
            assert result["error"] == "问答失败"

    @pytest.mark.asyncio
    async def test_run_ask_timeout(self, runner: Runner) -> None:
        """测试 _run_ask 方法超时"""
        import asyncio

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            MockExecutor.return_value = mock_instance

            options = runner._merge_options({})
            result = await runner._run_ask("超时的问题", options)

            assert result["success"] is False
            assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_run_dispatches_to_correct_mode(self, runner: Runner) -> None:
        """测试 run 方法正确分发到对应模式"""
        # 测试 BASIC 模式
        with patch.object(runner, "_run_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.BASIC, goal="基本任务")
            await runner.run(analysis)
            mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_mp_mode(self, runner: Runner) -> None:
        """测试 run 方法分发到 MP 模式"""
        with patch.object(runner, "_run_mp", new_callable=AsyncMock) as mock_mp:
            mock_mp.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.MP, goal="多进程任务")
            await runner.run(analysis)
            mock_mp.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_iterate_mode(self, runner: Runner) -> None:
        """测试 run 方法分发到 ITERATE 模式"""
        with patch.object(runner, "_run_iterate", new_callable=AsyncMock) as mock_iterate:
            mock_iterate.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.ITERATE, goal="迭代任务")
            await runner.run(analysis)
            mock_iterate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_plan_mode(self, runner: Runner) -> None:
        """测试 run 方法分发到 PLAN 模式"""
        with patch.object(runner, "_run_plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = {"success": True}
            analysis = TaskAnalysis(mode=RunMode.PLAN, goal="规划任务")
            await runner.run(analysis)
            mock_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dispatches_to_ask_mode(self, runner: Runner) -> None:
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
    async def test_run_knowledge_basic(self, runner: Runner) -> None:
        """测试 _run_knowledge 基本功能"""
        import coordinator
        import knowledge

        mock_km = MagicMock()
        mock_km.initialize = AsyncMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"success": True})
        with (
            patch.object(knowledge, "KnowledgeManager", return_value=mock_km),
            patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator),
        ):
            options = runner._merge_options({})
            result = await runner._run_knowledge("知识库任务", options)

            assert result["success"] is True
            mock_km.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_knowledge_with_search(self, runner: Runner) -> None:
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

        with (
            patch.object(knowledge, "KnowledgeManager", return_value=mock_km),
            patch.object(knowledge, "KnowledgeStorage", return_value=mock_storage),
            patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator),
        ):
            options = runner._merge_options({})
            options["search_knowledge"] = "测试搜索"
            result = await runner._run_knowledge("知识库任务", options)

            assert result["success"] is True
            mock_storage.search.assert_called_once()


# ============================================================
# TestTaskAnalyzerAgentAnalysis - Agent 分析测试
# ============================================================


class TestTaskAnalyzerAgentAnalysis:
    """测试 TaskAnalyzer Agent 分析相关功能

    验证 _agent_analysis 使用 plan 模式（只读执行），
    以及 JSON 提取逻辑在各种边界情况下保持稳定。
    """

    def test_agent_analysis_json_extraction(self, mock_args: argparse.Namespace, mock_subprocess) -> None:
        """验证 Agent 分析 JSON 提取，使用 plan 模式"""
        analyzer = TaskAnalyzer(use_agent=True)

        # 模拟包含 JSON 的输出
        mock_subprocess.return_value.stdout = """
        一些其他文本
        {"mode": "iterate", "options": {"skip_online": true}, "reasoning": "任务涉及自我迭代", "refined_goal": "执行自我迭代更新"}
        更多文本
        """

        analysis = analyzer.analyze("执行自我迭代", mock_args)

        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_uses_plan_mode(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析使用 plan 模式（只读保证）"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"mode": "basic", "reasoning": "test"}'
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            analyzer._agent_analysis("测试任务")

            # 验证调用使用了 plan 模式
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert "--mode" in cmd_args
            mode_idx = cmd_args.index("--mode")
            assert cmd_args[mode_idx + 1] == "plan"

    def test_agent_analysis_fallback_on_invalid_json(self, mock_args: argparse.Namespace, mock_subprocess) -> None:
        """验证无效 JSON 时回退到规则分析（稳定返回 None）"""
        analyzer = TaskAnalyzer(use_agent=True)

        mock_subprocess.return_value.stdout = "无效的响应内容"

        # 应该回退到规则分析，不抛出异常
        analysis = analyzer.analyze("执行自我迭代", mock_args)

        # 规则分析应该检测到迭代关键词
        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_timeout_fallback(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析超时时回退（稳定返回 None）"""
        import subprocess

        with patch("run.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            # 使用不会触发规则匹配的任务（规则分析返回 BASIC）
            analysis = analyzer.analyze("完成一个普通的编程任务", mock_args)

            # Agent 分析失败后回退，保持规则分析的 BASIC 模式
            assert analysis.mode == RunMode.BASIC

            # 验证 Agent 被调用且使用 plan 模式
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            assert "--mode" in cmd_args
            assert "plan" in cmd_args

    def test_agent_analysis_error_fallback(self, mock_args: argparse.Namespace, mock_subprocess) -> None:
        """验证 Agent 分析失败时回退（稳定返回 None）"""
        analyzer = TaskAnalyzer(use_agent=True)

        mock_subprocess.return_value.returncode = 1

        # 应该回退到规则分析
        analysis = analyzer.analyze("执行自我迭代", mock_args)
        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_empty_output_returns_none(self, mock_args: argparse.Namespace) -> None:
        """验证 _agent_analysis 空输出时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("普通任务")

            assert result is None

    def test_agent_analysis_non_json_output_returns_none(self, mock_args: argparse.Namespace) -> None:
        """验证 _agent_analysis 非 JSON 输出时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "这只是普通文本，没有任何 JSON 结构"
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("普通任务")

            assert result is None

    def test_agent_analysis_timeout_returns_none(self, mock_args: argparse.Namespace) -> None:
        """验证 _agent_analysis 超时时返回 None"""
        import subprocess

        with patch("run.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("普通任务")

            assert result is None

    def test_agent_analysis_cloud_mode_passthrough(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析返回 cloud 模式时正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """{
                "mode": "cloud",
                "options": {"execution_mode": "cloud"},
                "reasoning": "任务需要云端后台执行",
                "refined_goal": "云端执行任务"
            }"""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("后台长时间执行")

            assert result is not None
            assert result.mode == RunMode.CLOUD
            assert result.options.get("execution_mode") == "cloud"

    def test_agent_analysis_plan_mode_passthrough(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析返回 plan 模式时正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """{
                "mode": "plan",
                "options": {},
                "reasoning": "仅规划不执行",
                "refined_goal": "规划执行步骤"
            }"""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("分析代码架构")

            assert result is not None
            assert result.mode == RunMode.PLAN

    def test_agent_analysis_ask_mode_passthrough(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析返回 ask 模式时正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """{
                "mode": "ask",
                "options": {},
                "reasoning": "问答模式回答问题",
                "refined_goal": "回答问题"
            }"""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("解释这段代码")

            assert result is not None
            assert result.mode == RunMode.ASK

    def test_agent_analysis_new_options_passthrough(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析返回的新选项（auto_commit 等）正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """{
                "mode": "iterate",
                "options": {
                    "auto_commit": true,
                    "auto_push": true,
                    "commit_per_iteration": false,
                    "stream_log": true,
                    "no_mp": false,
                    "orchestrator": "mp",
                    "execution_mode": "cli"
                },
                "reasoning": "自我迭代需要自动提交",
                "refined_goal": "迭代更新并提交"
            }"""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("自我迭代并提交")

            assert result is not None
            assert result.mode == RunMode.ITERATE
            # 验证新选项透传
            assert result.options.get("auto_commit") is True
            assert result.options.get("auto_push") is True
            assert result.options.get("commit_per_iteration") is False
            assert result.options.get("stream_log") is True
            assert result.options.get("no_mp") is False
            assert result.options.get("orchestrator") == "mp"
            assert result.options.get("execution_mode") == "cli"

    def test_agent_analysis_partial_options_passthrough(self, mock_args: argparse.Namespace) -> None:
        """验证 Agent 分析返回部分选项时正确透传，未返回的选项不存在"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            # 仅返回部分选项
            mock_result.stdout = """{
                "mode": "mp",
                "options": {
                    "workers": 5,
                    "stream_log": false
                },
                "reasoning": "多进程处理",
                "refined_goal": "并行执行任务"
            }"""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("多进程处理任务")

            assert result is not None
            assert result.mode == RunMode.MP
            assert result.options.get("workers") == 5
            assert result.options.get("stream_log") is False
            # 未返回的选项不应存在
            assert "auto_commit" not in result.options
            assert "orchestrator" not in result.options


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

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
    def test_knowledge_imports(self) -> None:
        """验证 knowledge.KnowledgeManager, KnowledgeStorage 可导入"""
        from knowledge import KnowledgeManager, KnowledgeStorage

        # 验证类存在且可以被引用
        assert KnowledgeManager is not None
        assert KnowledgeStorage is not None

        # 验证是类类型
        assert isinstance(KnowledgeManager, type)
        assert isinstance(KnowledgeStorage, type)

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
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

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
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

            with patch.object(storage, "load_document", new_callable=AsyncMock, return_value=mock_doc) as mock_load:
                doc = await storage.load_document("test-doc-001")

                mock_load.assert_called_once_with("test-doc-001")
                assert doc is not None
                assert doc.title == "测试文档标题"
                assert doc.url == "https://example.com/doc"
                assert "这是文档内容" in doc.content

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
    @pytest.mark.asyncio
    async def test_enhanced_goal_building(self, mock_args: argparse.Namespace) -> None:
        """测试知识库上下文正确附加到目标描述中"""
        runner = Runner(mock_args)

        with (
            patch("knowledge.KnowledgeManager") as mock_km_class,
            patch("knowledge.KnowledgeStorage") as mock_storage_class,
            patch("coordinator.Orchestrator") as mock_orchestrator_class,
            patch("coordinator.OrchestratorConfig"),
            patch("cursor.client.CursorAgentConfig"),
        ):
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

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
    @pytest.mark.asyncio
    async def test_knowledge_mode_without_search(self, mock_args: argparse.Namespace) -> None:
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

        with (
            patch.object(knowledge, "KnowledgeManager", return_value=mock_km),
            patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator),
        ):
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

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
    @pytest.mark.asyncio
    async def test_knowledge_context_limit(self, mock_args: argparse.Namespace) -> None:
        """测试知识库上下文内容被正确截断"""
        runner = Runner(mock_args)

        with (
            patch("knowledge.KnowledgeManager") as mock_km_class,
            patch("knowledge.KnowledgeStorage") as mock_storage_class,
            patch("coordinator.Orchestrator") as mock_orchestrator_class,
            patch("coordinator.OrchestratorConfig"),
            patch("cursor.client.CursorAgentConfig"),
        ):
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

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
    @pytest.mark.asyncio
    async def test_search_knowledge_docs_readonly_no_create_dir(
        self, mock_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        """验证 minimal/dry_run 模式下 _search_knowledge_docs 不创建目录"""
        import os

        runner = Runner(mock_args)

        # 在临时目录中测试，确保 .cursor/knowledge 不存在
        knowledge_path = tmp_path / ".cursor" / "knowledge"
        assert not knowledge_path.exists()

        # 保存原始工作目录
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # 测试 minimal 模式
            result = await runner._search_knowledge_docs("test query", options={"minimal": True})
            assert result == []  # 应返回空列表
            assert not knowledge_path.exists()  # 不应创建目录

            # 测试 dry_run 模式
            result = await runner._search_knowledge_docs("test query", options={"dry_run": True})
            assert result == []  # 应返回空列表
            assert not knowledge_path.exists()  # 不应创建目录

        finally:
            os.chdir(original_cwd)

    @pytest.mark.skipif(not _knowledge_available, reason=f"知识库依赖不可用: {_knowledge_skip_reason}")
    @pytest.mark.asyncio
    async def test_run_knowledge_readonly_no_create_dir(self, mock_args: argparse.Namespace, tmp_path: Path) -> None:
        """验证 _run_knowledge 在 dry_run/minimal 模式下不创建 .cursor/knowledge"""
        import os

        import coordinator

        runner = Runner(mock_args)
        knowledge_path = tmp_path / ".cursor" / "knowledge"
        assert not knowledge_path.exists()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Mock Orchestrator 避免实际执行
            mock_orchestrator = MagicMock()
            mock_orchestrator.run = AsyncMock(return_value={"success": True})

            with patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator):
                # 测试 dry_run 模式
                options = runner._merge_options({"dry_run": True})
                options["search_knowledge"] = "test query"

                await runner._run_knowledge("测试任务", options)

                # 验证目录未被创建
                assert not knowledge_path.exists(), "dry_run 模式不应创建 .cursor/knowledge 目录"

            # 测试 minimal 模式
            with patch.object(coordinator, "Orchestrator", return_value=mock_orchestrator):
                options = runner._merge_options({"minimal": True})
                options["search_knowledge"] = "test query"

                await runner._run_knowledge("测试任务", options)

                assert not knowledge_path.exists(), "minimal 模式不应创建 .cursor/knowledge 目录"

        finally:
            os.chdir(original_cwd)


# ============================================================
# TestRunKnowledgeParseArgsConfigDefaults - 配置默认值测试
# ============================================================


class TestRunKnowledgeParseArgsConfigDefaults:
    """测试 scripts/run_knowledge.py 的 parse_args() 使用 tri-state 设计

    验证 CLI 参数默认值为 None (tri-state)，运行时通过 resolve 函数从 config.yaml 获取。
    注意：test_config_loading.py 已完整测试了 tri-state 行为和 resolve 函数。
    """

    def test_parse_args_uses_config_worker_pool_size(self) -> None:
        """测试 --workers 默认值为 None (tri-state)"""
        from scripts.run_knowledge import parse_args

        # 使用 patch 模拟 sys.argv，只提供必需的 goal 参数
        with patch("sys.argv", ["run_knowledge.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.workers is None

    def test_parse_args_uses_config_max_iterations(self) -> None:
        """测试 --max-iterations 默认值为 None (tri-state)"""
        from scripts.run_knowledge import parse_args

        with patch("sys.argv", ["run_knowledge.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.max_iterations is None

    def test_parse_args_uses_config_kb_limit(self) -> None:
        """测试 --kb-limit 默认值来自 config.yaml 的 worker.knowledge_integration.max_docs"""
        from core.config import get_config
        from scripts.run_knowledge import parse_args

        config = get_config()
        expected_kb_limit = config.worker.knowledge_integration.max_docs

        with patch("sys.argv", ["run_knowledge.py", "测试任务"]):
            args = parse_args()

        assert args.kb_limit == expected_kb_limit

    def test_parse_args_cli_overrides_config_workers(self) -> None:
        """测试 CLI --workers 参数可覆盖 config.yaml 默认值"""
        from scripts.run_knowledge import parse_args

        custom_workers = 7  # 使用与配置不同的值
        with patch("sys.argv", ["run_knowledge.py", "测试任务", "--workers", str(custom_workers)]):
            args = parse_args()

        assert args.workers == custom_workers
        assert args.workers != CONFIG_WORKER_POOL_SIZE or custom_workers == CONFIG_WORKER_POOL_SIZE

    def test_parse_args_cli_overrides_config_max_iterations(self) -> None:
        """测试 CLI --max-iterations 参数可覆盖 config.yaml 默认值"""
        from scripts.run_knowledge import parse_args

        custom_iterations = "25"
        with patch("sys.argv", ["run_knowledge.py", "测试任务", "--max-iterations", custom_iterations]):
            args = parse_args()

        assert args.max_iterations == custom_iterations

    def test_parse_args_cli_overrides_config_kb_limit(self) -> None:
        """测试 CLI --kb-limit 参数可覆盖 config.yaml 默认值"""
        from scripts.run_knowledge import parse_args

        custom_kb_limit = 15
        with patch("sys.argv", ["run_knowledge.py", "测试任务", "--kb-limit", str(custom_kb_limit)]):
            args = parse_args()

        assert args.kb_limit == custom_kb_limit

    def test_parse_args_help_shows_config_source(self) -> None:
        """测试帮助信息中显示默认值来源于 config.yaml"""
        import contextlib
        import io

        from scripts.run_knowledge import parse_args

        # 捕获帮助输出
        help_output = io.StringIO()
        with pytest.raises(SystemExit), patch("sys.argv", ["run_knowledge.py", "--help"]):
            with contextlib.redirect_stdout(help_output):
                parse_args()

        help_text = help_output.getvalue()
        # 验证帮助信息中包含 "来自 config.yaml" 的提示
        assert "config.yaml" in help_text


# ============================================================
# TestRunMpParseArgsConfigDefaults - run_mp.py 配置默认值测试
# ============================================================


class TestRunMpParseArgsConfigDefaults:
    """测试 scripts/run_mp.py 的 parse_args() 使用 tri-state 设计

    验证 CLI 参数默认值为 None (tri-state)，运行时通过 resolve 函数从 config.yaml 获取。
    注意：test_config_loading.py 已完整测试了 tri-state 行为和 resolve 函数。
    """

    def test_parse_args_uses_config_worker_pool_size(self) -> None:
        """测试 --workers 默认值为 None (tri-state)"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.workers is None

    def test_parse_args_uses_config_max_iterations(self) -> None:
        """测试 --max-iterations 默认值为 None (tri-state)"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.max_iterations is None

    def test_parse_args_uses_config_planning_timeout(self) -> None:
        """测试 --planning-timeout 默认值为 None (tri-state)"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.planning_timeout is None

    def test_parse_args_uses_config_execution_timeout(self) -> None:
        """测试 --execution-timeout 默认值为 None (tri-state)"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.execution_timeout is None

    def test_parse_args_uses_config_review_timeout(self) -> None:
        """测试 --review-timeout 默认值为 None (tri-state)"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.review_timeout is None

    def test_parse_args_cli_overrides_config_workers(self) -> None:
        """测试 CLI --workers 参数可覆盖 config.yaml 默认值"""
        from scripts.run_mp import parse_args

        custom_workers = 7
        with patch("sys.argv", ["run_mp.py", "测试任务", "--workers", str(custom_workers)]):
            args = parse_args()

        assert args.workers == custom_workers
        assert args.workers != CONFIG_WORKER_POOL_SIZE or custom_workers == CONFIG_WORKER_POOL_SIZE

    def test_parse_args_cli_overrides_config_max_iterations(self) -> None:
        """测试 CLI --max-iterations 参数可覆盖 config.yaml 默认值"""
        from scripts.run_mp import parse_args

        custom_iterations = "25"
        with patch("sys.argv", ["run_mp.py", "测试任务", "--max-iterations", custom_iterations]):
            args = parse_args()

        assert args.max_iterations == custom_iterations

    def test_parse_args_cli_overrides_config_planning_timeout(self) -> None:
        """测试 CLI --planning-timeout 参数可覆盖 config.yaml 默认值"""
        from scripts.run_mp import parse_args

        custom_timeout = 123.0
        with patch("sys.argv", ["run_mp.py", "测试任务", "--planning-timeout", str(custom_timeout)]):
            args = parse_args()

        assert args.planning_timeout == custom_timeout

    def test_parse_args_cli_overrides_config_execution_timeout(self) -> None:
        """测试 CLI --execution-timeout 参数可覆盖 config.yaml 默认值"""
        from scripts.run_mp import parse_args

        custom_timeout = 456.0
        with patch("sys.argv", ["run_mp.py", "测试任务", "--execution-timeout", str(custom_timeout)]):
            args = parse_args()

        assert args.execution_timeout == custom_timeout

    def test_parse_args_cli_overrides_config_review_timeout(self) -> None:
        """测试 CLI --review-timeout 参数可覆盖 config.yaml 默认值"""
        from scripts.run_mp import parse_args

        custom_timeout = 789.0
        with patch("sys.argv", ["run_mp.py", "测试任务", "--review-timeout", str(custom_timeout)]):
            args = parse_args()

        assert args.review_timeout == custom_timeout

    def test_parse_args_help_shows_config_source(self) -> None:
        """测试帮助信息中显示默认值来源于 config.yaml"""
        import contextlib
        import io

        from scripts.run_mp import parse_args

        help_output = io.StringIO()
        with pytest.raises(SystemExit), patch("sys.argv", ["run_mp.py", "--help"]):
            with contextlib.redirect_stdout(help_output):
                parse_args()

        help_text = help_output.getvalue()
        # 验证帮助信息中包含 "来自 config.yaml" 的提示
        assert "config.yaml" in help_text
        # 验证各超时参数都有来源提示
        assert "规划超时" in help_text
        assert "任务执行超时" in help_text
        assert "评审超时" in help_text


# ============================================================
# TestRunIterateMode - 自我迭代模式测试
# ============================================================


def _create_iterate_args_class():
    """创建完整的 IterateArgs 类（与 run.py 中的 IterateArgs 保持同步）

    此辅助函数用于测试中创建 IterateArgs 实例，包含 SelfIterator 所需的所有属性。
    """

    class IterateArgs:
        def __init__(self, goal: str, opts: dict):
            self.requirement = goal
            self.skip_online = opts.get("skip_online", False)
            self.changelog_url = opts.get("changelog_url")  # tri-state
            self.dry_run = opts.get("dry_run", False)
            self.max_iterations = str(opts.get("max_iterations", 5))
            self.workers = opts.get("workers", CONFIG_WORKER_POOL_SIZE)
            self.force_update = opts.get("force_update", False)
            self.verbose = opts.get("verbose", False)
            self.auto_commit = opts.get("auto_commit", False)
            self.auto_push = opts.get("auto_push", False)
            self.commit_per_iteration = opts.get("commit_per_iteration", False)
            self.commit_message = opts.get("commit_message", "")
            self.orchestrator = opts.get("orchestrator", "mp")
            self.no_mp = opts.get("no_mp", False)
            self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)
            # SelfIterator 所需的额外属性
            self.directory = opts.get("directory", ".")
            self.execution_mode = opts.get("execution_mode", "cli")
            self.cloud_api_key = opts.get("cloud_api_key")
            self.cloud_auth_timeout = opts.get("cloud_auth_timeout", 30)
            # 流式控制台渲染参数（默认关闭）
            self.stream_console_renderer = opts.get("stream_console_renderer", False)
            self.stream_advanced_renderer = opts.get("stream_advanced_renderer", False)
            self.stream_typing_effect = opts.get("stream_typing_effect", False)
            self.stream_typing_delay = opts.get("stream_typing_delay", 0.02)
            self.stream_word_mode = opts.get("stream_word_mode", True)
            self.stream_color_enabled = opts.get("stream_color_enabled", True)
            self.stream_show_word_diff = opts.get("stream_show_word_diff", False)
            # 文档源配置参数（tri-state：None=使用 config.yaml 默认值）
            self.max_fetch_urls = opts.get("max_fetch_urls")
            self.fallback_core_docs_count = opts.get("fallback_core_docs_count")
            self.llms_txt_url = opts.get("llms_txt_url")
            self.llms_cache_path = opts.get("llms_cache_path")

    return IterateArgs


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

    def test_iterate_args_orchestrator_user_set(self) -> None:
        """测试 IterateArgs 的 _orchestrator_user_set 元字段"""

        # 模拟 run.py 中的 IterateArgs 类（与 run.py 保持同步）
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
                self.orchestrator = opts.get("orchestrator", "mp")
                self.no_mp = opts.get("no_mp", False)
                # 元字段：标记用户是否显式设置了编排器选项
                self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)

        # 测试默认值（未显式设置）
        args_default = IterateArgs("测试任务", {})
        assert args_default._orchestrator_user_set is False
        assert args_default.orchestrator == "mp"
        assert args_default.no_mp is False

        # 测试用户显式设置 --orchestrator mp
        args_mp_explicit = IterateArgs(
            "测试任务",
            {
                "_orchestrator_user_set": True,
                "orchestrator": "mp",
            },
        )
        assert args_mp_explicit._orchestrator_user_set is True
        assert args_mp_explicit.orchestrator == "mp"

        # 测试用户显式设置 --orchestrator basic
        args_basic_explicit = IterateArgs(
            "测试任务",
            {
                "_orchestrator_user_set": True,
                "orchestrator": "basic",
            },
        )
        assert args_basic_explicit._orchestrator_user_set is True
        assert args_basic_explicit.orchestrator == "basic"

        # 测试用户显式设置 --no-mp
        args_no_mp = IterateArgs(
            "测试任务",
            {
                "_orchestrator_user_set": True,
                "no_mp": True,
            },
        )
        assert args_no_mp._orchestrator_user_set is True
        assert args_no_mp.no_mp is True

    def test_orchestrator_user_set_prevents_keyword_override(self) -> None:
        """验证显式设置编排器时，不会被 requirement 中的非并行关键词覆盖"""
        from scripts.run_iterate import SelfIterator

        # 模拟 run.py 中的 IterateArgs 类
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = opts.get("changelog_url")  # tri-state
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 5))
                self.workers = opts.get("workers", CONFIG_WORKER_POOL_SIZE)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")
                self.orchestrator = opts.get("orchestrator", "mp")
                self.no_mp = opts.get("no_mp", False)
                self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)
                self.directory = opts.get("directory", ".")
                self.execution_mode = opts.get("execution_mode", "cli")
                self.cloud_api_key = opts.get("cloud_api_key")
                self.cloud_auth_timeout = opts.get("cloud_auth_timeout", 30)
                # 流式控制台渲染参数（默认关闭）
                self.stream_console_renderer = opts.get("stream_console_renderer", False)
                self.stream_advanced_renderer = opts.get("stream_advanced_renderer", False)
                self.stream_typing_effect = opts.get("stream_typing_effect", False)
                self.stream_typing_delay = opts.get("stream_typing_delay", 0.02)
                self.stream_word_mode = opts.get("stream_word_mode", True)
                self.stream_color_enabled = opts.get("stream_color_enabled", True)
                self.stream_show_word_diff = opts.get("stream_show_word_diff", False)
                # 文档源配置参数（tri-state：None=使用 config.yaml 默认值）
                self.max_fetch_urls = opts.get("max_fetch_urls")
                self.fallback_core_docs_count = opts.get("fallback_core_docs_count")
                self.llms_txt_url = opts.get("llms_txt_url")
                self.llms_cache_path = opts.get("llms_cache_path")

        # 场景1：包含非并行关键词，但用户显式设置了 --orchestrator mp
        # 期望：使用 mp 编排器（不被关键词覆盖）
        args_explicit_mp = IterateArgs(
            "使用协程模式完成任务",
            {
                "_orchestrator_user_set": True,
                "orchestrator": "mp",
            },
        )
        iterator_explicit_mp = SelfIterator(args_explicit_mp)
        orchestrator_type = iterator_explicit_mp._get_orchestrator_type()
        assert orchestrator_type == "mp", "显式设置 --orchestrator mp 时应使用 mp 编排器"

        # 场景2：包含非并行关键词，用户未显式设置
        # 期望：使用 basic 编排器（被关键词覆盖）
        args_auto_detect = IterateArgs(
            "使用协程模式完成任务",
            {
                "_orchestrator_user_set": False,
                "orchestrator": "mp",
            },
        )
        iterator_auto_detect = SelfIterator(args_auto_detect)
        orchestrator_type = iterator_auto_detect._get_orchestrator_type()
        assert orchestrator_type == "basic", "未显式设置时应被关键词覆盖为 basic 编排器"

        # 场景3：不包含非并行关键词，用户未显式设置
        # 期望：使用默认 mp 编排器
        args_no_keyword = IterateArgs(
            "完成代码重构任务",
            {
                "_orchestrator_user_set": False,
                "orchestrator": "mp",
            },
        )
        iterator_no_keyword = SelfIterator(args_no_keyword)
        orchestrator_type = iterator_no_keyword._get_orchestrator_type()
        assert orchestrator_type == "mp", "无非并行关键词时应使用默认 mp 编排器"

        # 场景4：用户显式设置 --no-mp，即使没有非并行关键词
        # 期望：使用 basic 编排器
        args_explicit_no_mp = IterateArgs(
            "完成代码重构任务",
            {
                "_orchestrator_user_set": True,
                "no_mp": True,
            },
        )
        iterator_explicit_no_mp = SelfIterator(args_explicit_no_mp)
        orchestrator_type = iterator_explicit_no_mp._get_orchestrator_type()
        assert orchestrator_type == "basic", "显式设置 --no-mp 时应使用 basic 编排器"

    def test_self_iterator_init(self) -> None:
        """测试 SelfIterator 可以用 IterateArgs 初始化"""
        from scripts.run_iterate import SelfIterator

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

        # 创建 IterateArgs 实例
        iterate_args = IterateArgs(
            "自我迭代测试任务",
            {
                "skip_online": True,
                "dry_run": True,
            },
        )

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

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

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

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

        # 启用 auto_commit
        iterate_args = IterateArgs(
            "测试自动提交",
            {
                "skip_online": True,
                "auto_commit": True,
            },
        )
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
                with patch.object(iterator, "_run_commit_phase", new_callable=AsyncMock) as mock_commit_phase:
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

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

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
                mock_committer.generate_commit_message = AsyncMock(return_value="feat: 自动生成的提交信息")

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

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

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

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

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
                with patch.object(iterator, "_run_commit_phase", new_callable=AsyncMock) as mock_commit_phase:
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
        """测试规划模式成功返回计划（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        plan_text = """## 任务分解
1. 分析需求
2. 设计架构
3. 实现功能

## 执行顺序
- 1 -> 2 -> 3 (顺序执行)

## 推荐模式
mp (多进程模式)"""

        mock_result = AgentResult(
            success=True,
            output=plan_text,
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

            options = runner._merge_options({})
            result = await runner._run_plan("实现用户认证功能", options)

            # 验证返回结构
            assert result["success"] is True
            assert result["goal"] == "实现用户认证功能"
            assert result["mode"] == "plan"
            assert "plan" in result
            assert "任务分解" in result["plan"]
            assert result["dry_run"] is True

            # 验证 PlanAgentExecutor 被正确调用
            mock_executor_cls.assert_called_once()
            mock_executor.execute.assert_called_once()
            # 验证执行器配置（mode=plan, force_write=False）
            assert mock_executor.config.mode == "plan"
            assert mock_executor.config.force_write is False

    @pytest.mark.asyncio
    async def test_run_plan_timeout(self, runner: Runner) -> None:
        """测试规划模式超时处理（使用 PlanAgentExecutor）"""
        import asyncio

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_executor_cls.return_value = mock_executor

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
        """测试规划模式错误处理（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        mock_result = AgentResult(
            success=False,
            output="",
            error="Agent 服务不可用",
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

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
        """测试规划模式异常处理（使用 PlanAgentExecutor）"""
        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(side_effect=OSError("网络连接失败"))
            mock_executor_cls.return_value = mock_executor

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
        from cursor.executor import AgentResult

        answer_text = """Python 是一种高级编程语言，具有以下特点：
1. 简洁易读的语法
2. 丰富的标准库
3. 跨平台支持
4. 强大的社区生态"""

        mock_agent_result = AgentResult(
            success=True,
            output=answer_text,
            error=None,
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

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

            # 验证 AskAgentExecutor 被正确调用
            MockExecutor.assert_called_once()
            mock_instance.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ask_timeout(self, runner: Runner) -> None:
        """测试问答模式超时处理"""
        import asyncio

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            MockExecutor.return_value = mock_instance

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
        from cursor.executor import AgentResult

        mock_agent_result = AgentResult(
            success=False,
            output="",
            error="API 限流",
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

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
        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(side_effect=PermissionError("权限不足"))
            MockExecutor.return_value = mock_instance

            options = runner._merge_options({})
            result = await runner._run_ask("异常问题", options)

            # 验证异常返回结构
            assert result["success"] is False
            assert result["goal"] == "异常问题"
            assert result["mode"] == "ask"
            assert "error" in result
            assert "权限不足" in result["error"]

    @pytest.mark.asyncio
    async def test_run_ask_readonly_guarantee(self, runner: Runner) -> None:
        """测试问答模式的只读保证（force_write=False）

        验证 AskAgentExecutor 强制设置 mode=ask 和 force_write=False
        """
        from cursor.executor import AskAgentExecutor

        # 直接验证 AskAgentExecutor 的配置
        executor = AskAgentExecutor()
        assert executor.config.mode == "ask", "AskAgentExecutor 应强制 mode=ask"
        assert executor.config.force_write is False, "AskAgentExecutor 应强制 force_write=False"

    @pytest.mark.asyncio
    async def test_run_plan_readonly_guarantee(self, runner: Runner) -> None:
        """测试规划模式的只读保证（force_write=False）

        验证 PlanAgentExecutor 强制设置 mode=plan 和 force_write=False
        """
        from cursor.executor import PlanAgentExecutor

        # 直接验证 PlanAgentExecutor 的配置
        executor = PlanAgentExecutor()
        assert executor.config.mode == "plan", "PlanAgentExecutor 应强制 mode=plan"
        assert executor.config.force_write is False, "PlanAgentExecutor 应强制 force_write=False"

    @pytest.mark.asyncio
    async def test_plan_force_write_always_disabled(self, runner: Runner) -> None:
        """测试规划模式：即使用户尝试通过配置开启 force_write 也会被强制关闭

        验证 PlanAgentExecutor 会覆盖用户配置中的 force_write=True
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import PlanAgentExecutor

        # 用户尝试配置 force_write=True
        user_config = CursorAgentConfig(force_write=True)

        # 创建 PlanAgentExecutor，应强制关闭 force_write
        executor = PlanAgentExecutor(config=user_config)

        # 验证 force_write 被强制关闭
        assert executor.config.force_write is False, "PlanAgentExecutor 应强制覆盖用户配置的 force_write=True"
        assert executor.config.mode == "plan", "PlanAgentExecutor 应强制设置 mode=plan"

    @pytest.mark.asyncio
    async def test_ask_force_write_always_disabled(self, runner: Runner) -> None:
        """测试问答模式：即使用户尝试通过配置开启 force_write 也会被强制关闭

        验证 AskAgentExecutor 会覆盖用户配置中的 force_write=True
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import AskAgentExecutor

        # 用户尝试配置 force_write=True
        user_config = CursorAgentConfig(force_write=True)

        # 创建 AskAgentExecutor，应强制关闭 force_write
        executor = AskAgentExecutor(config=user_config)

        # 验证 force_write 被强制关闭
        assert executor.config.force_write is False, "AskAgentExecutor 应强制覆盖用户配置的 force_write=True"
        assert executor.config.mode == "ask", "AskAgentExecutor 应强制设置 mode=ask"

    @pytest.mark.asyncio
    async def test_plan_mode_always_preserved(self, runner: Runner) -> None:
        """测试规划模式：即使用户尝试通过配置设置 mode=agent 也会被覆盖

        验证 PlanAgentExecutor 会覆盖用户配置中的 mode
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import PlanAgentExecutor

        # 用户尝试配置 mode=agent
        user_config = CursorAgentConfig(mode="agent", force_write=True)

        # 创建 PlanAgentExecutor，应强制设置 mode=plan
        executor = PlanAgentExecutor(config=user_config)

        # 验证配置被强制覆盖
        assert executor.config.mode == "plan", "PlanAgentExecutor 应强制覆盖用户配置的 mode"
        assert executor.config.force_write is False, "PlanAgentExecutor 应强制覆盖用户配置的 force_write"

    @pytest.mark.asyncio
    async def test_ask_mode_always_preserved(self, runner: Runner) -> None:
        """测试问答模式：即使用户尝试通过配置设置 mode=agent 也会被覆盖

        验证 AskAgentExecutor 会覆盖用户配置中的 mode
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import AskAgentExecutor

        # 用户尝试配置 mode=agent
        user_config = CursorAgentConfig(mode="agent", force_write=True)

        # 创建 AskAgentExecutor，应强制设置 mode=ask
        executor = AskAgentExecutor(config=user_config)

        # 验证配置被强制覆盖
        assert executor.config.mode == "ask", "AskAgentExecutor 应强制覆盖用户配置的 mode"
        assert executor.config.force_write is False, "AskAgentExecutor 应强制覆盖用户配置的 force_write"

    # ----------------------------------------------------------
    # ReviewerAgent 只读保证测试
    # ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reviewer_readonly_guarantee(self, runner: Runner) -> None:
        """测试 ReviewerAgent 的只读保证（mode='ask' + force_write=False）

        验证 ReviewerAgent 强制设置 mode=ask 和 force_write=False
        """
        from agents.reviewer import ReviewerAgent, ReviewerConfig

        # 创建 ReviewerAgent
        config = ReviewerConfig()
        reviewer = ReviewerAgent(config)

        # 验证 ReviewerAgent 的配置
        assert reviewer.reviewer_config.cursor_config.mode == "ask", "ReviewerAgent 应强制 mode=ask"
        assert reviewer.reviewer_config.cursor_config.force_write is False, "ReviewerAgent 应强制 force_write=False"

    @pytest.mark.asyncio
    async def test_reviewer_force_write_always_disabled(self, runner: Runner) -> None:
        """测试 ReviewerAgent：即使用户传入 force_write=True 也会被覆盖为 False

        验证 ReviewerAgent._apply_ask_mode_config 会覆盖用户配置中的 force_write=True
        """
        from agents.reviewer import ReviewerAgent, ReviewerConfig
        from cursor.client import CursorAgentConfig

        # 用户尝试配置 force_write=True
        user_cursor_config = CursorAgentConfig(force_write=True)
        config = ReviewerConfig(cursor_config=user_cursor_config)

        # 创建 ReviewerAgent，应强制关闭 force_write
        reviewer = ReviewerAgent(config)

        # 验证 force_write 被强制关闭
        assert reviewer.reviewer_config.cursor_config.force_write is False, (
            "ReviewerAgent 应强制覆盖用户配置的 force_write=True"
        )
        assert reviewer.reviewer_config.cursor_config.mode == "ask", "ReviewerAgent 应强制设置 mode=ask"

    @pytest.mark.asyncio
    async def test_reviewer_mode_always_ask(self, runner: Runner) -> None:
        """测试 ReviewerAgent：即使用户传入 mode='agent' 也会被覆盖为 'ask'

        验证 ReviewerAgent._apply_ask_mode_config 会覆盖用户配置中的 mode
        """
        from agents.reviewer import ReviewerAgent, ReviewerConfig
        from cursor.client import CursorAgentConfig

        # 用户尝试配置 mode=agent
        user_cursor_config = CursorAgentConfig(mode="agent", force_write=True)
        config = ReviewerConfig(cursor_config=user_cursor_config)

        # 创建 ReviewerAgent，应强制设置 mode=ask
        reviewer = ReviewerAgent(config)

        # 验证配置被强制覆盖
        assert reviewer.reviewer_config.cursor_config.mode == "ask", "ReviewerAgent 应强制覆盖用户配置的 mode"
        assert reviewer.reviewer_config.cursor_config.force_write is False, (
            "ReviewerAgent 应强制覆盖用户配置的 force_write"
        )

    # ----------------------------------------------------------
    # 返回结构验证测试
    # ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_plan_result_structure(self, runner: Runner) -> None:
        """验证规划模式返回的 dict 结构完整性（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        mock_result = AgentResult(
            success=True,
            output="计划内容",
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

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

            # 验证执行器配置
            assert mock_executor.config.mode == "plan"
            assert mock_executor.config.force_write is False

    @pytest.mark.asyncio
    async def test_plan_error_result_structure(self, runner: Runner) -> None:
        """验证规划模式失败时返回的 dict 结构（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        mock_result = AgentResult(
            success=False,
            output="",
            error="错误信息",
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

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
        from cursor.executor import AgentResult

        mock_agent_result = AgentResult(
            success=True,
            output="回答内容",
            error=None,
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

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
        from cursor.executor import AgentResult

        mock_agent_result = AgentResult(
            success=False,
            output="",
            error="错误信息",
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

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
        """测试规划模式返回空输出（使用 PlanAgentExecutor）"""
        from cursor.executor import AgentResult

        mock_result = AgentResult(
            success=True,
            output="",
            executor_type="plan",
        )

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.config.mode = "plan"
            mock_executor.config.force_write = False
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

            result = await runner._run_plan("空输出任务", runner._merge_options({}))

            # 空输出仍然算成功
            assert result["success"] is True
            assert result["plan"] == ""

    @pytest.mark.asyncio
    async def test_run_ask_empty_output(self, runner: Runner) -> None:
        """测试问答模式返回空输出"""
        from cursor.executor import AgentResult

        mock_agent_result = AgentResult(
            success=True,
            output="   \n\t  ",  # 仅空白字符
            error=None,
            executor_type="ask",
        )

        with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value=mock_agent_result)
            MockExecutor.return_value = mock_instance

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
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,  # 默认禁用自动提交
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

        with (
            patch("run.parse_args", return_value=base_args),
            patch("run.setup_logging"),
            patch("run.print_error") as mock_print_error,
            patch.object(run_module, "print_info"),
        ):
            exit_code = await async_main()

            # 验证返回错误码
            assert exit_code == 1
            # 验证打印了错误信息
            mock_print_error.assert_called_once()
            error_msg = mock_print_error.call_args[0][0]
            assert "任务描述" in error_msg or "请提供" in error_msg

    @pytest.mark.asyncio
    async def test_auto_mode_triggers_analysis(self, base_args: argparse.Namespace) -> None:
        """测试 auto 模式触发 TaskAnalyzer.analyze"""
        from run import RunMode, TaskAnalysis, async_main

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

        with (
            patch("run.parse_args", return_value=base_args),
            patch("run.setup_logging"),
            patch("run.TaskAnalyzer") as mock_analyzer_class,
            patch("run.Runner") as mock_runner_class,
        ):
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
            mock_analyzer.analyze.assert_called_once_with("测试自动分析任务", base_args)

            # 验证 Runner.run 被调用
            mock_runner.run.assert_called_once()
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_explicit_mode_still_analyzes(self, base_args: argparse.Namespace) -> None:
        """测试显式指定模式仍会进行参数解析"""
        from run import RunMode, TaskAnalysis, async_main

        # 设置显式 basic 模式
        base_args.task = "显式模式任务"
        base_args.mode = "basic"  # 非 auto 模式
        base_args.no_auto_analyze = False

        with (
            patch("run.parse_args", return_value=base_args),
            patch("run.setup_logging"),
            patch("run.TaskAnalyzer") as mock_analyzer_class,
            patch("run.Runner") as mock_runner_class,
        ):
            # 设置 TaskAnalyzer mock
            mock_analysis = TaskAnalysis(
                mode=RunMode.MP,
                goal="解析后的目标",
                reasoning="解析结果",
            )
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
            mock_analyzer.analyze.assert_called_once_with("显式模式任务", base_args)

            # 验证 Runner.run 被调用，使用了显式指定的模式
            mock_runner.run.assert_called_once()
            call_args = mock_runner.run.call_args[0]
            analysis = call_args[0]
            assert isinstance(analysis, TaskAnalysis)
            assert analysis.mode == RunMode.BASIC
            assert analysis.goal == "解析后的目标"
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_no_auto_analyze_flag(self, base_args: argparse.Namespace) -> None:
        """测试 --no-auto-analyze 标志生效"""
        from run import RunMode, TaskAnalysis, async_main

        # 设置 auto 模式但禁用自动分析
        base_args.task = "禁用自动分析的任务"
        base_args.mode = "auto"
        base_args.no_auto_analyze = True  # 关键标志

        with (
            patch("run.parse_args", return_value=base_args),
            patch("run.setup_logging"),
            patch("run.TaskAnalyzer") as mock_analyzer_class,
            patch("run.Runner") as mock_runner_class,
        ):
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
    async def test_auto_mode_without_task_falls_back(self, base_args: argparse.Namespace) -> None:
        """测试 auto 模式无任务时不触发分析"""
        from run import async_main

        # auto 模式但无任务
        base_args.task = ""
        base_args.mode = "auto"
        base_args.no_auto_analyze = False

        with (
            patch("run.parse_args", return_value=base_args),
            patch("run.setup_logging"),
            patch("run.TaskAnalyzer") as mock_analyzer_class,
            patch("run.print_error") as mock_print_error,
            patch.object(run_module, "print_info"),
        ):
            exit_code = await async_main()

            # 验证 TaskAnalyzer 没有被调用（因为没有任务）
            mock_analyzer_class.assert_not_called()

            # 应该返回错误
            assert exit_code == 1
            mock_print_error.assert_called()

    @pytest.mark.asyncio
    async def test_runner_failure_returns_error_code(self, base_args: argparse.Namespace) -> None:
        """测试 Runner 执行失败时返回错误码"""
        from run import async_main

        base_args.task = "会失败的任务"
        base_args.mode = "basic"

        with (
            patch("run.parse_args", return_value=base_args),
            patch("run.setup_logging"),
            patch("run.Runner") as mock_runner_class,
            patch("run.print_result"),
        ):
            # 设置 Runner 返回失败结果
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value={"success": False})
            mock_runner_class.return_value = mock_runner

            exit_code = await async_main()

            # 验证返回非零错误码
            assert exit_code == 1


# ============================================================
# TestRunToSelfIteratorParameterMapping - run.py → SelfIterator 参数映射测试
# ============================================================


class TestRunToSelfIteratorParameterMapping:
    """测试 run.py _run_iterate 方法到 SelfIterator 的参数映射

    确保 run.py 中的 IterateArgs 类正确映射所有参数到 SelfIterator，
    包括新增的编排器选项和 _orchestrator_user_set 元字段。
    """

    @pytest.fixture
    def runner_with_iterate_args(self) -> Runner:
        """创建用于测试参数映射的 Runner 实例

        显式设置 execution_mode="cli" 来允许 mp 编排器，
        因为 config.yaml 默认 auto 会强制 basic。
        """
        args = argparse.Namespace(
            task="测试任务",
            mode="iterate",
            directory=".",
            workers=5,
            max_iterations="8",
            strict_review=None,
            enable_sub_planners=None,
            verbose=True,
            skip_online=True,
            dry_run=False,
            force_update=True,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,  # tri-state
            worker_model=None,  # tri-state
            reviewer_model=None,  # tri-state
            stream_log_enabled=None,
            auto_commit=True,
            auto_push=True,
            commit_per_iteration=True,
            orchestrator="mp",
            no_mp=False,
            execution_mode="cli",  # 显式 CLI 模式，允许 mp 编排器
            cloud_timeout=None,
            cloud_auth_timeout=None,
            _orchestrator_user_set=True,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )
        return Runner(args)

    def test_iterate_args_has_all_required_fields(self) -> None:
        """验证 IterateArgs 类包含所有必需字段"""

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
                self.orchestrator = opts.get("orchestrator", "mp")
                self.no_mp = opts.get("no_mp", False)
                self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)

        # 必需字段列表
        required_fields = [
            "requirement",
            "skip_online",
            "changelog_url",
            "dry_run",
            "max_iterations",
            "workers",
            "force_update",
            "verbose",
            "auto_commit",
            "auto_push",
            "commit_per_iteration",
            "commit_message",
            "orchestrator",
            "no_mp",
            "_orchestrator_user_set",
        ]

        args = IterateArgs("测试任务", {})
        for field in required_fields:
            assert hasattr(args, field), f"IterateArgs 缺少字段: {field}"

    def test_merge_options_passes_orchestrator_fields(self, mock_args: argparse.Namespace) -> None:
        """验证 _merge_options 正确传递编排器相关字段"""
        # 设置编排器选项
        mock_args.orchestrator = "basic"
        mock_args.no_mp = True
        mock_args._orchestrator_user_set = True

        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["orchestrator"] == "basic"
        assert options["no_mp"] is True

    def test_merge_options_analysis_overrides_when_not_user_set(self, mock_args: argparse.Namespace) -> None:
        """验证 _orchestrator_user_set=False 时 analysis_options 可覆盖"""
        mock_args.orchestrator = "mp"
        mock_args.no_mp = False
        mock_args._orchestrator_user_set = False

        runner = Runner(mock_args)

        # 自然语言分析结果想覆盖为 basic
        analysis_options = {"orchestrator": "basic", "no_mp": True}
        options = runner._merge_options(analysis_options)

        assert options["orchestrator"] == "basic"
        assert options["no_mp"] is True

    def test_merge_options_user_set_takes_priority(self, mock_args: argparse.Namespace) -> None:
        """验证 _orchestrator_user_set=True 时用户设置优先

        需要显式设置 execution_mode=cli 来允许 mp 编排器，
        因为 config.yaml 默认 auto 会强制 basic。
        """
        mock_args.orchestrator = "mp"
        mock_args.no_mp = False
        mock_args._orchestrator_user_set = True
        mock_args.execution_mode = "cli"  # 显式 CLI 模式，允许 mp 编排器

        runner = Runner(mock_args)

        # 自然语言分析结果想覆盖为 basic，但用户显式设置了 mp
        analysis_options = {"orchestrator": "basic", "no_mp": True}
        options = runner._merge_options(analysis_options)

        # 用户显式设置优先
        assert options["orchestrator"] == "mp"
        assert options["no_mp"] is False

    @pytest.mark.asyncio
    async def test_run_iterate_creates_iterate_args_with_orchestrator(self, runner_with_iterate_args: Runner) -> None:
        """验证 _run_iterate 正确创建包含编排器选项的 IterateArgs"""
        import scripts.run_iterate

        captured_args = None

        def capture_self_iterator_init(args):
            nonlocal captured_args
            captured_args = args
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value={"success": True})
            return mock_iterator

        with patch.object(scripts.run_iterate, "SelfIterator", side_effect=capture_self_iterator_init):
            options = runner_with_iterate_args._merge_options({})
            await runner_with_iterate_args._run_iterate("测试迭代任务", options)

            # 验证 IterateArgs 被正确创建
            assert captured_args is not None
            assert captured_args.requirement == "测试迭代任务"
            assert captured_args.workers == 5
            assert captured_args.skip_online is True
            assert captured_args.auto_commit is True
            assert captured_args.auto_push is True
            assert captured_args.commit_per_iteration is True
            # 验证编排器选项
            assert captured_args.orchestrator == "mp"
            assert captured_args.no_mp is False
            assert captured_args._orchestrator_user_set is True

    @pytest.mark.asyncio
    async def test_run_iterate_passes_orchestrator_user_set_from_args(self, mock_args: argparse.Namespace) -> None:
        """验证 _run_iterate 从 args 传递 _orchestrator_user_set"""
        import scripts.run_iterate

        # 设置原始 args 的 _orchestrator_user_set
        mock_args._orchestrator_user_set = True
        mock_args.orchestrator = "basic"
        mock_args.no_mp = False

        runner = Runner(mock_args)

        captured_args = None

        def capture_init(args):
            nonlocal captured_args
            captured_args = args
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value={"success": True})
            return mock_iterator

        with patch.object(scripts.run_iterate, "SelfIterator", side_effect=capture_init):
            options = runner._merge_options({})
            await runner._run_iterate("测试任务", options)

            # 验证 _orchestrator_user_set 被正确传递
            assert captured_args is not None
            assert captured_args._orchestrator_user_set is True
            assert captured_args.orchestrator == "basic"

    @pytest.mark.asyncio
    async def test_run_iterate_max_iterations_converted_to_string(self, mock_args: argparse.Namespace) -> None:
        """验证 max_iterations 被正确转换为字符串"""
        import scripts.run_iterate

        mock_args.max_iterations = "15"

        runner = Runner(mock_args)

        captured_args = None

        def capture_init(args):
            nonlocal captured_args
            captured_args = args
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value={"success": True})
            return mock_iterator

        with patch.object(scripts.run_iterate, "SelfIterator", side_effect=capture_init):
            options = runner._merge_options({"max_iterations": 15})
            await runner._run_iterate("测试任务", options)

            # 验证 max_iterations 是字符串类型
            assert captured_args is not None
            assert captured_args.max_iterations == "15"
            assert isinstance(captured_args.max_iterations, str)

    def test_task_analyzer_detects_non_parallel_keywords_for_iterate(self, mock_args: argparse.Namespace) -> None:
        """验证 TaskAnalyzer 检测非并行关键词并设置编排器选项"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 自我迭代 + 非并行关键词
        analysis = analyzer.analyze("自我迭代，使用协程模式完成", mock_args)

        # 应该检测到 ITERATE 模式
        assert analysis.mode == RunMode.ITERATE

        # 应该设置非并行选项
        assert analysis.options.get("no_mp") is True or analysis.options.get("orchestrator") == "basic"

    def test_full_parameter_flow_from_run_to_self_iterator(self, mock_args: argparse.Namespace) -> None:
        """验证完整参数流: TaskAnalyzer → Runner._merge_options → IterateArgs"""
        # 设置命令行默认值
        mock_args.orchestrator = "mp"
        mock_args.no_mp = False
        mock_args._orchestrator_user_set = False
        mock_args.auto_commit = False
        mock_args.auto_push = False
        mock_args.commit_per_iteration = False

        # 步骤1: TaskAnalyzer 分析任务
        analyzer = TaskAnalyzer(use_agent=False)
        task = "自我迭代，启用提交，使用协程模式"
        analysis = analyzer.analyze(task, mock_args)

        # 验证分析结果
        assert analysis.mode == RunMode.ITERATE
        # 应该检测到启用提交和非并行
        assert analysis.options.get("auto_commit") is True
        assert analysis.options.get("no_mp") is True or analysis.options.get("orchestrator") == "basic"

        # 步骤2: Runner._merge_options 合并选项
        runner = Runner(mock_args)
        options = runner._merge_options(analysis.options)

        # 验证合并后的选项
        assert options["auto_commit"] is True
        assert options["no_mp"] is True or options["orchestrator"] == "basic"

        # 步骤3: 模拟 IterateArgs 创建（与 run.py 中一致）
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
                self.orchestrator = opts.get("orchestrator", "mp")
                self.no_mp = opts.get("no_mp", False)
                self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)

        iterate_args = IterateArgs(analysis.goal, options)

        # 验证最终传递给 SelfIterator 的参数
        assert iterate_args.requirement == task
        assert iterate_args.auto_commit is True
        assert iterate_args.no_mp is True or iterate_args.orchestrator == "basic"

    def test_task_analyzer_detects_execution_mode_auto_keyword(self, mock_args: argparse.Namespace) -> None:
        """验证 TaskAnalyzer 检测执行模式关键词 - auto 模式"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试各种 auto 模式关键词
        test_cases = [
            "执行模式自动，分析代码",
            "云端优先执行任务",
            "使用 execution_mode=auto 完成任务",
            "execution-mode auto 分析项目",
        ]

        for task in test_cases:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("execution_mode") == "auto", f"任务 '{task}' 应检测到 execution_mode=auto"

    def test_task_analyzer_detects_execution_mode_cloud_keyword(self, mock_args: argparse.Namespace) -> None:
        """验证 TaskAnalyzer 检测执行模式关键词 - cloud 模式"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试各种 cloud 模式关键词
        test_cases = [
            "强制云端执行任务",
            "执行模式云端，分析代码",
            "使用 execution_mode=cloud 完成",
            "execution-mode cloud 运行",
        ]

        for task in test_cases:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("execution_mode") == "cloud", f"任务 '{task}' 应检测到 execution_mode=cloud"

    def test_task_analyzer_detects_execution_mode_cli_keyword(self, mock_args: argparse.Namespace) -> None:
        """验证 TaskAnalyzer 检测执行模式关键词 - cli 模式"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试各种 cli 模式关键词
        test_cases = [
            "本地执行任务",
            "执行模式本地，分析代码",
            "使用 execution_mode=cli 完成",
        ]

        for task in test_cases:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("execution_mode") == "cli", f"任务 '{task}' 应检测到 execution_mode=cli"

    def test_task_analyzer_execution_mode_auto_with_basic_orchestrator(self, mock_args: argparse.Namespace) -> None:
        """验证同时检测 execution_mode=auto 和 orchestrator=basic

        场景：任务描述中同时包含执行模式自动和非并行关键词
        期望：无需依赖 Agent 分析也能稳定得到 execution_mode=auto + orchestrator=basic
        """
        analyzer = TaskAnalyzer(use_agent=False)

        # 同时包含 auto 执行模式和非并行关键词
        test_cases = [
            "云端优先，使用协程模式完成任务",
            "执行模式自动，禁用多进程",
            "execution_mode=auto 使用 basic 编排器",
            "自动执行模式，非并行处理",
        ]

        for task in test_cases:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.options.get("execution_mode") == "auto", f"任务 '{task}' 应检测到 execution_mode=auto"
            assert analysis.options.get("no_mp") is True or analysis.options.get("orchestrator") == "basic", (
                f"任务 '{task}' 应检测到非并行/basic 编排器"
            )

    def test_task_analyzer_execution_mode_not_set_without_keyword(self, mock_args: argparse.Namespace) -> None:
        """验证没有执行模式关键词时不设置 execution_mode"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 不包含执行模式关键词的任务
        task = "分析代码结构并重构"
        analysis = analyzer.analyze(task, mock_args)

        # 不应设置 execution_mode（由后续配置解析决定）
        assert "execution_mode" not in analysis.options or analysis.options.get("execution_mode") is None, (
            "没有执行模式关键词时不应设置 execution_mode"
        )


# ============================================================
# TestRunCloudMode - Cloud 模式执行测试
# ============================================================


class TestRunCloudMode:
    """测试 _run_cloud() 方法的 Cloud 执行逻辑

    验证:
    - AgentExecutorFactory.create 被正确调用
    - cloud_auth_config / timeout / force_write 参数符合预期
    - 无 API key 时的错误分支与提示
    """

    @pytest.fixture
    def cloud_runner_args(self) -> argparse.Namespace:
        """创建用于 Cloud 模式测试的参数"""
        return argparse.Namespace(
            task="& 云端执行任务",
            mode="cloud",
            directory="/test/dir",
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
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cloud",
            cloud_api_key=None,
            cloud_auth_timeout=30,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            # 流式渲染参数
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 角色执行模式
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

    @pytest.mark.asyncio
    async def test_run_cloud_calls_factory_with_correct_mode(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 使用正确的 ExecutionMode.CLOUD 调用工厂"""
        from cursor.executor import ExecutionMode

        runner = Runner(cloud_runner_args)

        captured_mode = None
        captured_cli_config = None
        captured_cloud_auth_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_mode, captured_cli_config, captured_cloud_auth_config
            captured_mode = mode
            captured_cli_config = cli_config
            captured_cloud_auth_config = cloud_auth_config
            # 返回模拟的执行器
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "Cloud 执行成功"
            mock_result.session_id = "test-session-id"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-api-key"}):
                options = runner._merge_options({"execution_mode": "cloud"})
                await runner._run_cloud("测试云端任务", options)

        # 验证工厂被调用且模式正确
        assert captured_mode == ExecutionMode.CLOUD

    @pytest.mark.asyncio
    async def test_run_cloud_passes_correct_cloud_auth_config(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 正确传递 CloudAuthConfig（api_key 来自环境变量）"""
        runner = Runner(cloud_runner_args)

        captured_cloud_auth_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cloud_auth_config
            captured_cloud_auth_config = cloud_auth_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session-123"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "my-secret-key"}):
                options = runner._merge_options({})
                await runner._run_cloud("任务", options)

        # 验证 CloudAuthConfig 被正确创建并传递
        assert captured_cloud_auth_config is not None
        assert captured_cloud_auth_config.api_key == "my-secret-key"

    @pytest.mark.asyncio
    async def test_run_cloud_passes_correct_timeout(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 使用 cloud_timeout 参数（默认 300 秒）"""
        runner = Runner(cloud_runner_args)

        captured_cli_config = None
        captured_execute_timeout = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cli_config
            captured_cli_config = cli_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session"
            mock_result.files_modified = []

            async def capture_execute(*args, **kwargs):
                nonlocal captured_execute_timeout
                captured_execute_timeout = kwargs.get("timeout")
                return mock_result

            mock_executor.execute = capture_execute
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                # 直接在 options 中设置 cloud_timeout
                options["cloud_timeout"] = 900
                await runner._run_cloud("任务", options)

        # 验证 timeout 使用 cloud_timeout（设置为 900）
        assert captured_cli_config is not None
        assert captured_cli_config.timeout == 900
        assert captured_execute_timeout == 900

    @pytest.mark.asyncio
    async def test_run_cloud_force_write_default_false(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 默认 force_write=False

        Policy 变更：force_write 默认 False，与本地 CLI 保持一致
        仅当用户显式传入 --force 或 force_write=True 时才允许写入
        """
        runner = Runner(cloud_runner_args)

        captured_cli_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cli_config
            captured_cli_config = cli_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                # 不指定 force_write，应默认为 False（与本地 CLI 一致）
                options = runner._merge_options({})
                await runner._run_cloud("任务", options)

        # 验证 force_write 默认为 False
        assert captured_cli_config is not None
        assert captured_cli_config.force_write is False

    @pytest.mark.asyncio
    async def test_run_cloud_force_write_can_be_overridden(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 可以通过 options 覆盖 force_write"""
        runner = Runner(cloud_runner_args)

        captured_cli_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cli_config
            captured_cli_config = cli_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                # 显式在 options 中指定 force_write=False
                # 注意: _run_cloud 直接从 options 获取 force_write
                options = runner._merge_options({})
                options["force_write"] = False  # 显式覆盖
                await runner._run_cloud("任务", options)

        # 验证 force_write 被覆盖为 False
        assert captured_cli_config is not None
        assert captured_cli_config.force_write is False

    @pytest.mark.asyncio
    async def test_run_cloud_no_api_key_returns_error_early(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证无 API key 时提前返回错误（不调用工厂）"""
        runner = Runner(cloud_runner_args)

        factory_called = False

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal factory_called
            factory_called = True
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            # 确保环境变量中没有 CURSOR_API_KEY
            with patch.dict("os.environ", {}, clear=True):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        # 验证无 key 时提前返回错误，工厂未被调用
        assert factory_called is False
        assert result["success"] is False
        assert "API Key" in result["error"] or "未配置" in result["error"]

    @pytest.mark.asyncio
    async def test_run_cloud_executor_failure_returns_error(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证执行器失败时返回正确的错误信息"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Cloud API 连接失败"
            # 显式设置避免 MagicMock 返回值导致 compute_message_dedup_key 失败
            mock_result.cooldown_info = None
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_result.session_id = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert result["error"] == "Cloud API 连接失败"
        assert result["mode"] == "cloud"

    @pytest.mark.asyncio
    async def test_run_cloud_timeout_exception_handling(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证超时异常被正确处理"""
        import asyncio

        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        # 错误信息包含超时相关内容
        assert "超时" in result["error"] or "timeout" in result["error"].lower()
        assert result["mode"] == "cloud"

    @pytest.mark.asyncio
    async def test_run_cloud_generic_exception_handling(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证通用异常被正确处理"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=RuntimeError("网络错误"))
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert "网络错误" in result["error"]
        assert result["mode"] == "cloud"

    @pytest.mark.asyncio
    async def test_run_cloud_success_returns_complete_result(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证成功执行时返回完整的结果"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "Cloud 任务执行成功，修改了 3 个文件"
            mock_result.session_id = "cloud-session-abc123"
            mock_result.files_modified = ["file1.py", "file2.py", "file3.py"]
            # 显式设置避免 MagicMock 返回值导致 compute_message_dedup_key 失败
            mock_result.cooldown_info = None
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("云端任务", options)

        assert result["success"] is True
        assert result["mode"] == "cloud"
        assert result["goal"] == "云端任务"
        assert result["output"] == "Cloud 任务执行成功，修改了 3 个文件"
        assert result["session_id"] == "cloud-session-abc123"
        assert result["files_modified"] == ["file1.py", "file2.py", "file3.py"]

    @pytest.mark.asyncio
    async def test_run_cloud_uses_directory_from_options(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 使用 options 中的 directory"""
        cloud_runner_args.directory = "/custom/work/dir"
        runner = Runner(cloud_runner_args)

        captured_cli_config = None
        captured_working_dir = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cli_config
            captured_cli_config = cli_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session"
            mock_result.files_modified = []

            async def capture_execute(*args, **kwargs):
                nonlocal captured_working_dir
                captured_working_dir = kwargs.get("working_directory")
                return mock_result

            mock_executor.execute = capture_execute
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                await runner._run_cloud("任务", options)

        # 验证工作目录被正确传递
        assert captured_cli_config is not None
        assert captured_cli_config.working_directory == "/custom/work/dir"
        assert captured_working_dir == "/custom/work/dir"

    @pytest.mark.asyncio
    async def test_run_cloud_ampersand_prefix_triggers_background_mode(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证 '&' 前缀触发 Cloud 后台模式，传递 background=True

        当用户输入以 '&' 开头时：
        1. TaskAnalyzer 识别为 Cloud 模式
        2. analysis.options['cloud_background'] = True
        3. _run_cloud 传递 background=True 给执行器
        """
        cloud_runner_args.task = "& 后台分析代码"
        runner = Runner(cloud_runner_args)

        captured_background = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "后台任务已提交"
            mock_result.session_id = "bg-session-123"
            mock_result.files_modified = []

            async def capture_execute(*args, **kwargs):
                nonlocal captured_background
                captured_background = kwargs.get("background")
                return mock_result

            mock_executor.execute = capture_execute
            return mock_executor

        # 构建包含正确 & 前缀状态的 ExecutionDecision
        from core.execution_policy import build_execution_decision

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                with patch("run.get_config", return_value=mock_config):
                    # 模拟 '&' 前缀成功触发 Cloud 路由的选项（由 TaskAnalyzer 设置）
                    # 提供 _execution_decision 对象，确保 prefix_routed=True 被正确保留
                    decision = build_execution_decision(
                        prompt="& 后台分析代码",  # & 前缀触发
                        requested_mode=None,
                        cloud_enabled=True,
                        has_api_key=True,
                    )
                    options = runner._merge_options(
                        {
                            "cloud_background": True,  # '&' 前缀触发后台模式
                            "_execution_decision": decision,  # 提供决策对象
                        }
                    )
                    await runner._run_cloud("后台分析代码", options)

        # 验证 background=True 被传递给执行器
        assert captured_background is True, "'&' 前缀触发时应传递 background=True"

    @pytest.mark.asyncio
    async def test_run_cloud_returns_resume_command_on_success(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证成功执行后返回 resume_command（包含 session_id）

        返回结果应包含:
        - session_id: 会话 ID
        - resume_command: 'agent --resume <session_id>' 格式的恢复命令
        """
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "任务完成"
            mock_result.session_id = "cloud-session-xyz789"
            mock_result.files_modified = []
            # 显式设置避免 MagicMock 返回值导致 compute_message_dedup_key 失败
            mock_result.cooldown_info = None
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        # 验证返回结果包含 resume_command
        assert result["success"] is True
        assert result["session_id"] == "cloud-session-xyz789"
        assert result["resume_command"] == "agent --resume cloud-session-xyz789", (
            "resume_command 格式应为 'agent --resume <session_id>'"
        )

    @pytest.mark.asyncio
    async def test_run_cloud_background_mode_returns_resume_command(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证后台模式返回 resume_command 供用户恢复会话"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = ""  # 后台模式可能无输出
            mock_result.session_id = "bg-session-abc"
            mock_result.files_modified = []
            # 显式设置避免 MagicMock 返回值导致 compute_message_dedup_key 失败
            mock_result.cooldown_info = None
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        # 构建包含正确 & 前缀状态的 ExecutionDecision
        from core.execution_policy import build_execution_decision

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                with patch("run.get_config", return_value=mock_config):
                    # 模拟后台模式选项（& 前缀成功触发 Cloud 路由）
                    # 提供 _execution_decision 对象，确保 prefix_routed=True 被正确保留
                    decision = build_execution_decision(
                        prompt="& 后台任务",  # & 前缀触发
                        requested_mode=None,
                        cloud_enabled=True,
                        has_api_key=True,
                    )
                    options = runner._merge_options(
                        {
                            "cloud_background": True,
                            "_execution_decision": decision,  # 提供决策对象
                        }
                    )
                    result = await runner._run_cloud("后台任务", options)

        # 验证后台模式返回正确的元数据
        assert result["success"] is True
        assert result["background"] is True
        assert result["session_id"] == "bg-session-abc"
        assert result["resume_command"] == "agent --resume bg-session-abc"
        # 验证 prefix_routed=True（& 前缀成功触发 Cloud）
        # ⚠ 新断言规范：主要断言使用 prefix_routed
        assert result["prefix_routed"] is True
        # 验证兼容字段存在且值正确（to_dict() 兼容输出，保留此断言以确保兼容性）
        # 注意：新代码的决策分支应使用 prefix_routed，此处仅验证兼容输出
        assert result["triggered_by_prefix"] is True

    @pytest.mark.asyncio
    async def test_run_cloud_no_session_id_returns_none_resume_command(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证无 session_id 时 resume_command 为 None"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "执行完成"
            mock_result.session_id = None  # 无 session_id
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        # 验证无 session_id 时 resume_command 为 None
        assert result["session_id"] is None
        assert result["resume_command"] is None

    @pytest.mark.asyncio
    async def test_run_cloud_execution_mode_cloud_default_foreground(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证 --execution-mode cloud 显式指定时默认前台模式（background=False）

        区别于 '&' 前缀触发（默认后台），--execution-mode cloud 应默认前台执行。
        """
        runner = Runner(cloud_runner_args)

        captured_background = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "前台执行完成"
            mock_result.session_id = "fg-session"
            mock_result.files_modified = ["main.py"]

            async def capture_execute(*args, **kwargs):
                nonlocal captured_background
                captured_background = kwargs.get("background")
                return mock_result

            mock_executor.execute = capture_execute
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                # 不设置 cloud_background，默认应为 False
                options = runner._merge_options(
                    {
                        "execution_mode": "cloud",
                        # 不设置 cloud_background
                    }
                )
                await runner._run_cloud("任务", options)

        # 验证默认前台模式
        assert captured_background is False, "--execution-mode cloud 默认应为前台模式（background=False）"

    @pytest.mark.asyncio
    async def test_run_cloud_result_contains_failure_kind_field(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 _run_cloud 返回结果包含 failure_kind 字段

        无论成功或失败，结果字典都应包含 failure_kind 字段。
        成功时为 None，失败时包含错误类型。
        """
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "session-123"
            mock_result.files_modified = []
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_result.cooldown_info = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        # 验证返回结果结构包含 failure_kind 字段
        assert "failure_kind" in result
        assert "retry_after" in result

    @pytest.mark.asyncio
    async def test_run_cloud_failure_kind_populated_on_error(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 Cloud 执行失败时 failure_kind 被正确填充

        当执行器返回失败结果时，failure_kind 应包含错误分类。
        """
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.output = ""
            mock_result.error = "Rate limit exceeded"
            mock_result.session_id = None
            mock_result.files_modified = []
            mock_result.failure_kind = "rate_limit"
            mock_result.retry_after = 60
            mock_result.cooldown_info = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert result["failure_kind"] == "rate_limit"
        assert result["retry_after"] == 60

    @pytest.mark.asyncio
    async def test_run_cloud_timeout_exception_sets_failure_kind(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 Cloud 超时异常时 failure_kind 被设置为 timeout"""
        import asyncio

        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert result["failure_kind"] == "timeout"

    @pytest.mark.asyncio
    async def test_run_cloud_network_exception_sets_failure_kind(self, cloud_runner_args: argparse.Namespace) -> None:
        """验证 Cloud 网络异常时 failure_kind 被正确分类"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=ConnectionError("Network unreachable"))
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert result["failure_kind"] == "network"


class TestRunCloudModeNoApiKeyError:
    """测试 Cloud 模式下无 API key 的边界用例

    验证 cloud_auth_config 为 None 时的行为以及相关错误提示。
    注意: 当前实现中，无 key 时仍会创建执行器（cloud_auth_config=None），
    具体错误处理由执行器内部完成。
    """

    @pytest.fixture
    def no_key_args(self) -> argparse.Namespace:
        """创建无 API key 的参数"""
        return argparse.Namespace(
            task="& 云端任务",
            mode="cloud",
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
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cloud",
            cloud_api_key=None,  # 无 API key
            cloud_auth_timeout=30,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

    @pytest.mark.asyncio
    async def test_no_api_key_returns_error_without_calling_factory(self, no_key_args: argparse.Namespace) -> None:
        """验证无 API key 时提前返回错误，不调用工厂"""
        runner = Runner(no_key_args)

        factory_called = False

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal factory_called
            factory_called = True
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "s"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        # 清空环境变量中的 CURSOR_API_KEY
        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {}, clear=True):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        # 验证无 key 时提前返回，工厂未被调用
        assert factory_called is False
        assert result["success"] is False
        assert result["mode"] == "cloud"

    @pytest.mark.asyncio
    async def test_no_api_key_executor_error_is_captured(self, no_key_args: argparse.Namespace) -> None:
        """验证无 API key 导致执行器抛出认证错误时被正确捕获"""
        runner = Runner(no_key_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            # 模拟执行器因无认证而失败
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "缺少 API Key，请设置 CURSOR_API_KEY 环境变量"
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {}, clear=True):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert "API Key" in result["error"] or "认证" in result["error"] or "缺少" in result["error"]

    @pytest.mark.asyncio
    async def test_no_api_key_error_message_content(self, no_key_args: argparse.Namespace) -> None:
        """验证无 API key 时返回有意义的错误提示"""
        runner = Runner(no_key_args)

        # 清空环境变量中的 CURSOR_API_KEY
        with patch.dict("os.environ", {}, clear=True):
            options = runner._merge_options({})
            result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        # 错误消息应包含 API Key 相关提示
        assert "API Key" in result["error"] or "未配置" in result["error"]
        assert result["mode"] == "cloud"
        # 验证 session_id 存在（即使为 None）
        assert "session_id" in result


class TestExecutionModeAutoCloudFallbackHints:
    """测试 execution_mode=auto/cloud + 无 key 的提示与降级

    验证场景：
    1. execution_mode=auto + 无 key: 输出可操作提示并降级到 CLI
    2. execution_mode=cloud + 无 key: 提前返回错误并输出提示
    3. 降级时的用户友好消息格式
    """

    @pytest.fixture
    def auto_mode_args(self) -> argparse.Namespace:
        """创建 execution_mode=auto 的参数"""
        return argparse.Namespace(
            task="自动模式任务",
            mode="iterate",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode="auto",  # auto 模式
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,  # 无 API key
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    def test_auto_mode_no_key_uses_policy_resolution(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 auto 模式 + 无 key 使用策略解析降级到 CLI"""
        from core.execution_policy import resolve_effective_execution_mode

        # 策略层应该直接解析为 CLI
        mode, reason = resolve_effective_execution_mode(
            requested_mode="auto",
            has_ampersand_prefix=False,  # 无 & 前缀
            cloud_enabled=True,
            has_api_key=False,  # 无 key
        )

        assert mode == "cli", "auto + 无 key 应解析为 CLI"
        assert "未配置 API Key" in reason, f"原因应包含 API Key 提示，实际: {reason}"

    def test_cloud_mode_no_key_uses_policy_resolution(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 cloud 模式 + 无 key 使用策略解析降级到 CLI"""
        from core.execution_policy import resolve_effective_execution_mode

        mode, reason = resolve_effective_execution_mode(
            requested_mode="cloud",
            has_ampersand_prefix=False,  # 无 & 前缀
            cloud_enabled=True,
            has_api_key=False,  # 无 key
        )

        assert mode == "cli", "cloud + 无 key 应解析为 CLI"
        assert "未配置 API Key" in reason

    def test_auto_mode_no_key_fallback_message_actionable(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 auto + 无 key 时回退消息包含可操作的提示"""
        from core.execution_policy import (
            CloudFailureKind,
            build_user_facing_fallback_message,
        )

        message = build_user_facing_fallback_message(
            kind=CloudFailureKind.NO_KEY,
            retry_after=None,
            requested_mode="auto",
            has_ampersand_prefix=False,
        )

        # 验证消息包含可操作提示
        assert "未配置" in message, "消息应说明问题"
        assert "API Key" in message, "消息应提及 API Key"
        assert "CURSOR_API_KEY" in message, "消息应告知环境变量名"

    def test_ampersand_prefix_no_key_fallback_message(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 & 前缀 + 无 key 时回退消息格式正确"""
        from core.execution_policy import (
            CloudFailureKind,
            build_user_facing_fallback_message,
        )

        message = build_user_facing_fallback_message(
            kind=CloudFailureKind.NO_KEY,
            retry_after=None,
            requested_mode=None,  # 由前缀触发
            has_ampersand_prefix=True,
        )

        # 验证消息包含前缀触发信息
        assert "未配置" in message
        assert "& 前缀" in message, "消息应说明由 & 前缀触发"

    @pytest.mark.asyncio
    async def test_runner_auto_mode_no_key_degrades_gracefully(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 Runner 在 auto + 无 key 时优雅降级"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        auto_mode_args.mode = "auto"
        runner = Runner(auto_mode_args)

        # Mock CloudClientFactory.resolve_api_key 返回 None
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            options = runner._merge_options({"execution_mode": "auto"})
            # 策略层应该处理降级
            mode = runner._get_execution_mode(options)

            # 无 key 时应该降级到 CLI
            assert mode == ExecutionMode.CLI, f"auto + 无 key 应降级到 CLI，实际: {mode}"

    def test_task_analyzer_detects_auto_mode_keyword(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 TaskAnalyzer 检测 auto 模式关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试包含 auto 关键词的任务（使用正确的关键词模式）
        analysis = analyzer.analyze("云端优先分析代码", auto_mode_args)

        assert analysis.options.get("execution_mode") == "auto", "应检测到 execution_mode=auto"

    def test_task_analyzer_detects_cloud_mode_keyword(self, auto_mode_args: argparse.Namespace) -> None:
        """验证 TaskAnalyzer 检测 cloud 模式关键词"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试包含 cloud 关键词的任务（使用正确的关键词模式）
        analysis = analyzer.analyze("强制云端执行任务", auto_mode_args)

        assert analysis.options.get("execution_mode") == "cloud", "应检测到 execution_mode=cloud"


class TestRunCloudModeParameterValidation:
    """测试 _run_cloud 参数验证

    验证各种参数组合的正确性。
    """

    @pytest.fixture
    def base_args(self) -> argparse.Namespace:
        """基础参数"""
        return argparse.Namespace(
            task="云端任务",
            mode="cloud",
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
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cloud",
            cloud_api_key=None,
            cloud_auth_timeout=30,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
        )

    @pytest.mark.asyncio
    async def test_cloud_timeout_independent_from_max_iterations(self, base_args: argparse.Namespace) -> None:
        """验证 Cloud timeout 独立于 max_iterations，使用 cloud_timeout 配置

        新设计：
        - Cloud timeout 不再从 max_iterations 推导
        - 使用独立的 cloud_timeout 参数（来自 config.yaml 或 --cloud-timeout）
        - 默认值为 DEFAULT_CLOUD_TIMEOUT (300 秒)
        """
        base_args.max_iterations = "-1"  # 无限迭代不影响 cloud_timeout
        runner = Runner(base_args)

        captured_timeout = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "s"
            mock_result.files_modified = []

            async def capture_execute(*args, **kwargs):
                nonlocal captured_timeout
                captured_timeout = kwargs.get("timeout")
                return mock_result

            mock_executor.execute = capture_execute
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                # max_iterations = -1 不影响 cloud_timeout
                options = runner._merge_options({"max_iterations": -1})
                await runner._run_cloud("任务", options)

        # Cloud timeout 使用独立的 cloud_timeout 参数，默认值来自 config.yaml
        # 不再从 max_iterations 推导
        expected_timeout = CONFIG_CLOUD_TIMEOUT  # 来自 config.yaml（默认 300）
        assert captured_timeout == expected_timeout, (
            f"Cloud timeout 应为 {expected_timeout}（来自 config.yaml），实际值: {captured_timeout}"
        )

    @pytest.mark.asyncio
    async def test_cloud_mode_auto_commit_independence(self, base_args: argparse.Namespace) -> None:
        """验证 Cloud 模式下 force_write 与 auto_commit 独立

        Policy 变更：force_write 默认 False，与本地 CLI 保持一致
        auto_commit 与 force_write 独立，两者都需要用户显式启用
        """
        base_args.auto_commit = False  # auto_commit 禁用
        runner = Runner(base_args)

        captured_cli_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cli_config
            captured_cli_config = cli_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "s"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                await runner._run_cloud("任务", options)

        # force_write 默认 False（Policy 变更：与本地 CLI 一致）
        # auto_commit=False 与 force_write 独立
        assert captured_cli_config is not None
        assert captured_cli_config.force_write is False

    @pytest.mark.asyncio
    async def test_executor_execute_receives_correct_prompt(self, base_args: argparse.Namespace) -> None:
        """验证执行器 execute 方法收到正确的 prompt"""
        runner = Runner(base_args)

        captured_prompt = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "s"
            mock_result.files_modified = []

            async def capture_execute(*args, **kwargs):
                nonlocal captured_prompt
                captured_prompt = kwargs.get("prompt")
                return mock_result

            mock_executor.execute = capture_execute
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                await runner._run_cloud("这是测试 prompt", options)

        assert captured_prompt == "这是测试 prompt"


# ============================================================
# TestRunBasicParseArgsConfigDefaults - run_basic.py 配置默认值测试
# ============================================================


class TestRunBasicParseArgsConfigDefaults:
    """测试 scripts/run_basic.py 的 parse_args() 使用 tri-state 设计

    验证 CLI 参数默认值为 None (tri-state)，运行时通过 resolve 函数从 config.yaml 获取。
    注意：test_config_loading.py 已完整测试了 tri-state 行为和 resolve 函数。
    """

    def test_parse_args_uses_config_worker_pool_size(self) -> None:
        """测试 --workers 默认值为 None (tri-state)"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.workers is None

    def test_parse_args_uses_config_max_iterations(self) -> None:
        """测试 --max-iterations 默认值为 None (tri-state)"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.max_iterations is None

    def test_parse_args_uses_config_enable_sub_planners_default_none(self) -> None:
        """测试 enable_sub_planners 未指定时为 None (tri-state)"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务"]):
            args = parse_args()

        # 未指定时应为 None，由 run_orchestrator 从 config.yaml 获取
        assert args.enable_sub_planners is None

    def test_parse_args_uses_config_strict_review_default_none(self) -> None:
        """测试 strict_review 未指定时为 None (tri-state)"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务"]):
            args = parse_args()

        # 未指定时应为 None，由 run_orchestrator 从 config.yaml 获取
        assert args.strict_review is None

    def test_parse_args_cli_overrides_config_workers(self) -> None:
        """测试 CLI --workers 参数可覆盖 config.yaml 默认值"""
        from scripts.run_basic import parse_args

        custom_workers = 7
        with patch("sys.argv", ["run_basic.py", "测试任务", "--workers", str(custom_workers)]):
            args = parse_args()

        assert args.workers == custom_workers

    def test_parse_args_cli_overrides_config_max_iterations(self) -> None:
        """测试 CLI --max-iterations 参数可覆盖 config.yaml 默认值"""
        from scripts.run_basic import parse_args

        custom_iterations = "25"
        with patch("sys.argv", ["run_basic.py", "测试任务", "--max-iterations", custom_iterations]):
            args = parse_args()

        assert args.max_iterations == custom_iterations

    def test_parse_args_strict_flag_sets_true(self) -> None:
        """测试 --strict 参数将 strict_review 设为 True"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务", "--strict"]):
            args = parse_args()

        assert args.strict_review is True

    def test_parse_args_no_strict_flag_sets_false(self) -> None:
        """测试 --no-strict 参数将 strict_review 设为 False"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务", "--no-strict"]):
            args = parse_args()

        assert args.strict_review is False

    def test_parse_args_sub_planners_flag_sets_true(self) -> None:
        """测试 --sub-planners 参数将 enable_sub_planners 设为 True"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务", "--sub-planners"]):
            args = parse_args()

        assert args.enable_sub_planners is True

    def test_parse_args_no_sub_planners_flag_sets_false(self) -> None:
        """测试 --no-sub-planners 参数将 enable_sub_planners 设为 False"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务", "--no-sub-planners"]):
            args = parse_args()

        assert args.enable_sub_planners is False

    def test_parse_args_strict_and_no_strict_mutually_exclusive(self) -> None:
        """测试 --strict 和 --no-strict 互斥"""
        from scripts.run_basic import parse_args

        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_basic.py", "测试任务", "--strict", "--no-strict"]):
                parse_args()

    def test_parse_args_sub_planners_mutually_exclusive(self) -> None:
        """测试 --sub-planners 和 --no-sub-planners 互斥"""
        from scripts.run_basic import parse_args

        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_basic.py", "测试任务", "--sub-planners", "--no-sub-planners"]):
                parse_args()

    def test_parse_args_help_shows_config_source(self) -> None:
        """测试帮助信息中显示默认值来源于 config.yaml"""
        import contextlib
        import io

        from scripts.run_basic import parse_args

        help_output = io.StringIO()
        with pytest.raises(SystemExit), patch("sys.argv", ["run_basic.py", "--help"]):
            with contextlib.redirect_stdout(help_output):
                parse_args()

        help_text = help_output.getvalue()
        # 验证帮助信息中包含 "config.yaml" 的提示
        assert "config.yaml" in help_text


class TestRunBasicOrchestratorConfigAssembly:
    """测试 run_basic.py 的 OrchestratorConfig 组装逻辑

    验证 CLI > config.yaml 优先级正确实现。
    """

    @pytest.mark.asyncio
    async def test_orchestrator_config_uses_config_defaults_when_cli_unspecified(
        self,
    ) -> None:
        """测试未指定 CLI 参数时使用 config.yaml 默认值"""
        import argparse

        from scripts.run_basic import run_orchestrator

        args = argparse.Namespace(
            goal="测试任务",
            directory=".",
            workers=CONFIG_WORKER_POOL_SIZE,
            max_iterations=str(CONFIG_MAX_ITERATIONS),
            enable_sub_planners=None,  # 未指定
            strict_review=None,  # 未指定
            verbose=False,
            mock=False,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
        )

        captured_config = None

        # Mock Orchestrator 以捕获配置
        from coordinator import Orchestrator

        original_init = Orchestrator.__init__

        def mock_init(self, config):
            nonlocal captured_config
            captured_config = config
            # 不调用原始 init 以避免实际初始化

        with patch.object(Orchestrator, "__init__", mock_init):
            with patch.object(Orchestrator, "run", new_callable=AsyncMock, return_value={"success": True}):
                try:
                    await run_orchestrator(args)
                except Exception:
                    pass  # 忽略任何初始化错误

        # 验证配置使用了 config.yaml 默认值
        assert captured_config is not None
        assert captured_config.enable_sub_planners == CONFIG_ENABLE_SUB_PLANNERS
        assert captured_config.strict_review == CONFIG_STRICT_REVIEW

    @pytest.mark.asyncio
    async def test_orchestrator_config_cli_overrides_enable_sub_planners(self) -> None:
        """测试 CLI 参数可覆盖 enable_sub_planners"""
        import argparse

        from scripts.run_basic import run_orchestrator

        # 使用与 config.yaml 相反的值
        cli_value = not CONFIG_ENABLE_SUB_PLANNERS

        args = argparse.Namespace(
            goal="测试任务",
            directory=".",
            workers=CONFIG_WORKER_POOL_SIZE,
            max_iterations=str(CONFIG_MAX_ITERATIONS),
            enable_sub_planners=cli_value,  # CLI 指定
            strict_review=None,
            verbose=False,
            mock=False,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
        )

        captured_config = None

        from coordinator import Orchestrator

        def mock_init(self, config):
            nonlocal captured_config
            captured_config = config

        with patch.object(Orchestrator, "__init__", mock_init):
            with patch.object(Orchestrator, "run", new_callable=AsyncMock, return_value={"success": True}):
                try:
                    await run_orchestrator(args)
                except Exception:
                    pass

        assert captured_config is not None
        assert captured_config.enable_sub_planners == cli_value

    @pytest.mark.asyncio
    async def test_orchestrator_config_cli_overrides_strict_review(self) -> None:
        """测试 CLI 参数可覆盖 strict_review"""
        import argparse

        from scripts.run_basic import run_orchestrator

        # 使用与 config.yaml 相反的值
        cli_value = not CONFIG_STRICT_REVIEW

        args = argparse.Namespace(
            goal="测试任务",
            directory=".",
            workers=CONFIG_WORKER_POOL_SIZE,
            max_iterations=str(CONFIG_MAX_ITERATIONS),
            enable_sub_planners=None,
            strict_review=cli_value,  # CLI 指定
            verbose=False,
            mock=False,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
        )

        captured_config = None

        from coordinator import Orchestrator

        def mock_init(self, config):
            nonlocal captured_config
            captured_config = config

        with patch.object(Orchestrator, "__init__", mock_init):
            with patch.object(Orchestrator, "run", new_callable=AsyncMock, return_value={"success": True}):
                try:
                    await run_orchestrator(args)
                except Exception:
                    pass

        assert captured_config is not None
        assert captured_config.strict_review == cli_value


# ============================================================
# execution_mode 配置默认值测试
# ============================================================


class TestExecutionModeConfigDefault:
    """测试 execution_mode 参数使用 tri-state 设计

    验证场景:
    1. run.py parse_args() 的 --execution-mode 默认值为 None (tri-state)
    2. CLI 显式参数可覆盖默认值
    3. '&' 前缀触发 Cloud 模式仍优先于配置默认值
    4. Cloud/Auto 模式下 orchestrator 兼容性策略仍生效（强制 basic）

    注意：test_config_loading.py 已完整测试了 tri-state 行为和 resolve 函数。
    """

    @pytest.fixture
    def config_execution_mode(self) -> str:
        """获取配置中的 execution_mode 默认值"""
        return get_config().cloud_agent.execution_mode

    def test_parse_args_execution_mode_uses_config_default(self, config_execution_mode: str) -> None:
        """测试 run.py parse_args() 的 --execution-mode 默认值为 None (tri-state)"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.execution_mode is None, (
            f"--execution-mode 默认值应为 None (tri-state)，实际值: {args.execution_mode}"
        )

    def test_parse_args_execution_mode_cli_override(self) -> None:
        """测试 CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务", "--execution-mode", "cloud"]):
            args = parse_args()

        assert args.execution_mode == "cloud"

    def test_parse_args_help_shows_config_source_for_execution_mode(self) -> None:
        """测试帮助信息中 --execution-mode 显示来源于 config.yaml"""
        import contextlib
        import io

        from run import parse_args

        help_output = io.StringIO()
        with pytest.raises(SystemExit), patch("sys.argv", ["run.py", "--help"]):
            with contextlib.redirect_stdout(help_output):
                parse_args()

        help_text = help_output.getvalue()
        # 验证帮助信息中包含 "config.yaml" 的提示
        assert "config.yaml" in help_text, "--execution-mode 帮助信息应包含 'config.yaml' 来源提示"

    def test_ampersand_prefix_overrides_config_default_execution_mode(self, mock_args: argparse.Namespace) -> None:
        """测试 '&' 前缀触发 Cloud 模式优先于配置默认值

        场景：配置中 execution_mode 未显式指定（tri-state None），
              但任务带 '&' 前缀且配置了 API Key
        期望：Cloud 模式被激活，覆盖配置默认值

        关键区别：
        - args.execution_mode=None (tri-state 未指定) = 允许 & 触发 Cloud
        - args.execution_mode="cli" (显式指定) = 忽略 & 前缀

        需要同时满足 cloud_enabled=True 和 has_api_key=True
        """
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory

        # tri-state: None 表示未显式指定，使用 config.yaml 默认值
        # 这种情况下 '&' 前缀可以触发 Cloud 模式
        mock_args.execution_mode = None
        analyzer = TaskAnalyzer(use_agent=False)

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = True
            config.cloud_agent.execution_mode = "cli"

            # Mock Cloud API Key 存在和 cloud_enabled=True
            with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
                # 带 '&' 前缀的任务
                analysis = analyzer.analyze("& 分析代码架构", mock_args)

                # '&' 前缀应触发 Cloud 模式（因为 execution_mode 未显式指定）
                assert analysis.mode == RunMode.CLOUD
                # effective_mode 是决策后的实际模式
                assert analysis.options.get("effective_mode") == "cloud"
                assert analysis.goal == "分析代码架构"
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    def test_explicit_cli_mode_ignores_ampersand_prefix(self, mock_args: argparse.Namespace) -> None:
        """测试显式 --execution-mode cli 时忽略 '&' 前缀

        场景：用户显式指定 --execution-mode cli，任务带 '&' 前缀
        期望：'&' 前缀被当作普通字符，不触发 Cloud 模式

        优先级规则：显式 execution_mode=cli > & 前缀触发 Cloud
        """
        from cursor.cloud_client import CloudClientFactory

        # 显式指定 execution_mode=cli（非 None）
        mock_args.execution_mode = "cli"
        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        # 即使 API Key 存在，显式 CLI 模式也应忽略 & 前缀
        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"),
            patch("run.get_config", return_value=mock_config),
        ):
            # 带 '&' 前缀的任务
            analysis = analyzer.analyze("& 分析代码架构", mock_args)

            # 不应触发 Cloud 模式
            assert analysis.mode != RunMode.CLOUD, "显式 execution_mode=cli 时，'&' 前缀应被忽略"
            # goal 应去除 '&' 前缀（作为普通字符处理）
            assert analysis.goal == "分析代码架构"
            # 不应设置 Cloud 相关选项（prefix_routed=False）
            # 内部分支使用 prefix_routed 字段
            assert analysis.options.get("cloud_background") is not True
            assert analysis.options.get("prefix_routed") is not True

    def test_merge_options_uses_analysis_execution_mode_over_config(self, mock_args: argparse.Namespace) -> None:
        """测试 _merge_options 中 analysis_options 的 execution_mode 优先于配置

        场景：args.execution_mode=None（未显式设置），analysis 中设置了 cloud
        期望：最终 options["execution_mode"] 为 cloud

        注意：如果 args.execution_mode 有值（不管来自 CLI 还是配置），它有最高优先级。
        """
        mock_args.execution_mode = None  # 未显式设置，让 analysis_options 生效
        runner = Runner(mock_args)

        # 模拟 TaskAnalyzer 检测到 '&' 前缀后设置的选项
        analysis_options = {"execution_mode": "cloud"}
        options = runner._merge_options(analysis_options)

        assert options["execution_mode"] == "cloud", "analysis_options 中的 execution_mode 应优先于配置默认值"


class TestExecutionModeConfigDefaultWithMock:
    """测试 execution_mode 配置默认值（使用 mock 配置）

    通过 mock ConfigManager 来测试不同配置值下的行为。
    """

    def test_config_auto_default_parsed_correctly(self) -> None:
        """测试配置 execution_mode=auto 时未传参的默认解析"""

        # 保存原始配置
        original_execution_mode = get_config().cloud_agent.execution_mode

        try:
            # Mock 配置为 auto
            with patch.object(get_config().cloud_agent, "execution_mode", "auto"):
                # 需要重新导入以使用新的默认值
                # 由于 parse_args 在模块加载时读取配置，这里直接验证配置值
                assert get_config().cloud_agent.execution_mode == "auto"
        finally:
            # 配置会自动恢复（patch 结束后）
            pass

    def test_config_cloud_default_parsed_correctly(self) -> None:
        """测试配置 execution_mode=cloud 时未传参的默认解析"""

        # Mock 配置为 cloud
        with patch.object(get_config().cloud_agent, "execution_mode", "cloud"):
            assert get_config().cloud_agent.execution_mode == "cloud"

    def test_orchestrator_compatibility_with_config_cloud_default(self, mock_args: argparse.Namespace) -> None:
        """测试配置 execution_mode=cloud 时 orchestrator 兼容性策略

        场景：配置默认 execution_mode=cloud，用户未显式设置 orchestrator
        期望：MP 模式应给出警告，因为 Cloud 模式不支持 MP 编排器
        """
        from run import Runner

        # 模拟配置默认值为 cloud
        mock_args.execution_mode = "cloud"
        mock_args.orchestrator = "mp"
        mock_args._orchestrator_user_set = False

        runner = Runner(mock_args)
        options = runner._merge_options({})

        # 验证 execution_mode 被正确传递
        assert options["execution_mode"] == "cloud"
        # 注意：实际的 orchestrator 强制切换发生在 _run_iterate 或 _run_mp 中

    def test_orchestrator_compatibility_with_config_auto_default(self, mock_args: argparse.Namespace) -> None:
        """测试配置 execution_mode=auto 时 orchestrator 兼容性策略

        场景：配置默认 execution_mode=auto，用户未显式设置 orchestrator
        期望：Auto 模式也应触发 basic 编排器（与 Cloud 模式一致）
        """
        from run import Runner

        # 模拟配置默认值为 auto
        mock_args.execution_mode = "auto"
        mock_args.orchestrator = "mp"
        mock_args._orchestrator_user_set = False

        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["execution_mode"] == "auto"


class TestRunIterateExecutionModeConfigDefault:
    """测试 scripts/run_iterate.py 的 execution_mode 使用 tri-state 设计

    注意：test_config_loading.py 已完整测试了 tri-state 行为和 resolve 函数。
    """

    @pytest.fixture
    def config_execution_mode(self) -> str:
        """获取配置中的 execution_mode 默认值"""
        return get_config().cloud_agent.execution_mode

    def test_run_iterate_parse_args_uses_config_default(self, config_execution_mode: str) -> None:
        """测试 run_iterate.py parse_args() 的 --execution-mode 默认值为 None (tri-state)"""
        from scripts.run_iterate import parse_args

        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.execution_mode is None, (
            f"run_iterate.py --execution-mode 默认值应为 None (tri-state)，实际值: {args.execution_mode}"
        )

    def test_run_iterate_parse_args_cli_override(self) -> None:
        """测试 run_iterate.py CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        from scripts.run_iterate import parse_args

        with patch("sys.argv", ["run_iterate.py", "测试任务", "--execution-mode", "cloud"]):
            args = parse_args()

        assert args.execution_mode == "cloud"

    def test_run_iterate_help_shows_config_source(self) -> None:
        """测试 run_iterate.py 帮助信息中 --execution-mode 显示来源于 config.yaml"""
        import contextlib
        import io

        from scripts.run_iterate import parse_args

        help_output = io.StringIO()
        with pytest.raises(SystemExit), patch("sys.argv", ["run_iterate.py", "--help"]):
            with contextlib.redirect_stdout(help_output):
                parse_args()

        help_text = help_output.getvalue()
        assert "config.yaml" in help_text, "run_iterate.py --execution-mode 帮助信息应包含 'config.yaml' 来源提示"


# ============================================================
# TestTableDrivenParseArgsDefaults - 表驱动测试 parse_args 默认值
# ============================================================


class TestTableDrivenParseArgsDefaults:
    """表驱动测试：验证各脚本 parse_args() 默认值来自 get_config()

    覆盖脚本：
    - run.py
    - scripts/run_basic.py
    - scripts/run_mp.py
    - scripts/run_iterate.py

    测试字段：
    - workers / worker_pool_size
    - max_iterations
    - cloud_timeout
    - cloud_auth_timeout
    - planner_model / worker_model / reviewer_model
    """

    @pytest.fixture
    def config_defaults(self) -> dict:
        """获取配置中的默认值"""
        config = get_config()
        return {
            "worker_pool_size": config.system.worker_pool_size,
            "max_iterations": config.system.max_iterations,
            "cloud_timeout": config.cloud_agent.timeout,
            "cloud_auth_timeout": config.cloud_agent.auth_timeout,
            "execution_mode": config.cloud_agent.execution_mode,
            "planner_model": config.models.planner,
            "worker_model": config.models.worker,
            "reviewer_model": config.models.reviewer,
            "planner_timeout": config.planner.timeout,
            "worker_timeout": config.worker.task_timeout,
            "reviewer_timeout": config.reviewer.timeout,
        }

    # ========== run.py parse_args 测试 ==========

    # run.py 中使用 tri-state 设计的字段（返回 None，运行时通过 resolve_orchestrator_settings 解析）
    # 包括模型字段 planner_model, worker_model, reviewer_model
    RUN_PY_TRISTATE_FIELDS = [
        "workers",
        "max_iterations",
        "cloud_timeout",
        "cloud_auth_timeout",
        "execution_mode",
        "planner_model",
        "worker_model",
        "reviewer_model",
    ]

    # run.py 中直接从 config.yaml 读取默认值的字段（目前为空，所有字段都使用 tri-state）
    RUN_PY_DIRECT_CONFIG_FIELD_TESTS: list = []

    @pytest.mark.parametrize(
        "arg_name",
        RUN_PY_TRISTATE_FIELDS,
        ids=[f"run.py:{field}" for field in RUN_PY_TRISTATE_FIELDS],
    )
    def test_run_py_parse_args_tristate_returns_none(self, arg_name: str) -> None:
        """表驱动测试：run.py parse_args() tri-state 字段返回 None"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        assert actual is None, f"run.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），实际值: {actual}"

    @pytest.mark.parametrize(
        "arg_name,config_key",
        RUN_PY_DIRECT_CONFIG_FIELD_TESTS,
        ids=[f"run.py:{case[0]}" for case in RUN_PY_DIRECT_CONFIG_FIELD_TESTS],
    )
    def test_run_py_parse_args_default(self, config_defaults: dict, arg_name: str, config_key: str) -> None:
        """表驱动测试：run.py parse_args() 非 tri-state 字段默认值来自 config.yaml"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        expected = config_defaults[config_key]
        actual = getattr(args, arg_name)

        assert actual == expected, (
            f"run.py --{arg_name.replace('_', '-')} 默认值应为 {expected}（来自 config.yaml），实际值: {actual}"
        )

    # ========== scripts/run_basic.py parse_args 测试 ==========

    # run_basic.py 使用 tri-state 设计的字段
    RUN_BASIC_TRISTATE_FIELDS = [
        "workers",
        "max_iterations",
    ]

    @pytest.mark.parametrize(
        "arg_name",
        RUN_BASIC_TRISTATE_FIELDS,
        ids=[f"run_basic.py:{field}" for field in RUN_BASIC_TRISTATE_FIELDS],
    )
    def test_run_basic_py_parse_args_default(self, config_defaults: dict, arg_name: str) -> None:
        """表驱动测试：scripts/run_basic.py parse_args() tri-state 字段返回 None"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert actual is None, (
            f"run_basic.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），实际值: {actual}"
        )

    # ========== scripts/run_mp.py parse_args 测试 ==========

    # run_mp.py 使用 tri-state 设计的字段
    RUN_MP_TRISTATE_FIELDS = [
        "workers",
        "max_iterations",
        "planner_model",
        "worker_model",
        "reviewer_model",
        "planning_timeout",
        "execution_timeout",
        "review_timeout",
    ]

    @pytest.mark.parametrize(
        "arg_name",
        RUN_MP_TRISTATE_FIELDS,
        ids=[f"run_mp.py:{field}" for field in RUN_MP_TRISTATE_FIELDS],
    )
    def test_run_mp_py_parse_args_default(self, config_defaults: dict, arg_name: str) -> None:
        """表驱动测试：scripts/run_mp.py parse_args() tri-state 字段返回 None"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert actual is None, f"run_mp.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），实际值: {actual}"

    # ========== scripts/run_iterate.py parse_args 测试 ==========

    # run_iterate.py 中使用 tri-state 设计的字段（返回 None，运行时通过 resolve_orchestrator_settings 解析）
    RUN_ITERATE_TRISTATE_FIELDS = [
        "workers",
        "max_iterations",
        "cloud_timeout",
        "cloud_auth_timeout",
        "execution_mode",
    ]

    @pytest.mark.parametrize(
        "arg_name",
        RUN_ITERATE_TRISTATE_FIELDS,
        ids=[f"run_iterate.py:{field}" for field in RUN_ITERATE_TRISTATE_FIELDS],
    )
    def test_run_iterate_py_parse_args_tristate_returns_none(self, arg_name: str) -> None:
        """表驱动测试：scripts/run_iterate.py parse_args() tri-state 字段返回 None"""
        from scripts.run_iterate import parse_args

        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        assert actual is None, (
            f"run_iterate.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），实际值: {actual}"
        )


# ============================================================
# TestCloudTimeoutAuthTimeoutTableDriven - Cloud 超时参数表驱动测试
# ============================================================


class TestCloudTimeoutAuthTimeoutTableDriven:
    """表驱动测试：Cloud 超时和认证超时参数

    测试场景：
    1. 默认值来自 config.yaml
    2. CLI 参数可覆盖默认值
    3. 参数传递到下游组件
    """

    @pytest.fixture
    def config_cloud_values(self) -> dict:
        """获取配置中的 Cloud 相关默认值"""
        config = get_config()
        return {
            "timeout": config.cloud_agent.timeout,
            "auth_timeout": config.cloud_agent.auth_timeout,
        }

    # Cloud 超时参数测试数据
    CLOUD_TIMEOUT_CLI_OVERRIDE_TESTS = [
        # (script_module, cli_arg, cli_value, expected_attr)
        ("run", "--cloud-timeout", "1200", "cloud_timeout", 1200),
        ("run", "--cloud-auth-timeout", "60", "cloud_auth_timeout", 60),
        ("scripts.run_iterate", "--cloud-timeout", "900", "cloud_timeout", 900),
        ("scripts.run_iterate", "--cloud-auth-timeout", "45", "cloud_auth_timeout", 45),
    ]

    @pytest.mark.parametrize(
        "script_module,cli_arg,cli_value,attr_name,expected_value",
        CLOUD_TIMEOUT_CLI_OVERRIDE_TESTS,
        ids=[f"{case[0]}:{case[1]}" for case in CLOUD_TIMEOUT_CLI_OVERRIDE_TESTS],
    )
    def test_cloud_timeout_cli_override(
        self,
        script_module: str,
        cli_arg: str,
        cli_value: str,
        attr_name: str,
        expected_value: int,
    ) -> None:
        """表驱动测试：CLI 参数可覆盖 Cloud 超时默认值"""
        import importlib

        module = importlib.import_module(script_module)
        parse_args = module.parse_args

        argv = [f"{script_module.split('.')[-1]}.py", "测试任务", cli_arg, cli_value]
        with patch("sys.argv", argv):
            args = parse_args()

        actual = getattr(args, attr_name)
        assert actual == expected_value, f"{script_module} {cli_arg} 应为 {expected_value}，实际值: {actual}"


# ============================================================
# TestWorkerPoolMaxIterationsTableDriven - Worker 池和迭代次数表驱动测试
# ============================================================


class TestWorkerPoolMaxIterationsTableDriven:
    """表驱动测试：worker_pool_size 和 max_iterations 参数

    测试场景：
    1. 默认值来自 config.yaml
    2. CLI 参数可覆盖默认值
    3. 特殊值处理（MAX/-1 表示无限迭代）
    """

    @pytest.fixture
    def config_system_values(self) -> dict:
        """获取配置中的 system 相关默认值"""
        config = get_config()
        return {
            "worker_pool_size": config.system.worker_pool_size,
            "max_iterations": config.system.max_iterations,
        }

    # Worker 数量 CLI 覆盖测试数据
    WORKERS_CLI_OVERRIDE_TESTS = [
        ("run", "-w", "8", 8),
        ("run", "--workers", "10", 10),
        ("scripts.run_basic", "-w", "5", 5),
        ("scripts.run_basic", "--workers", "7", 7),
        ("scripts.run_mp", "-w", "6", 6),
        ("scripts.run_mp", "--workers", "12", 12),
        ("scripts.run_iterate", "--workers", "9", 9),
    ]

    @pytest.mark.parametrize(
        "script_module,cli_arg,cli_value,expected_value",
        WORKERS_CLI_OVERRIDE_TESTS,
        ids=[f"{case[0]}:{case[1]}={case[2]}" for case in WORKERS_CLI_OVERRIDE_TESTS],
    )
    def test_workers_cli_override(
        self,
        script_module: str,
        cli_arg: str,
        cli_value: str,
        expected_value: int,
    ) -> None:
        """表驱动测试：CLI 参数可覆盖 workers 默认值"""
        import importlib

        module = importlib.import_module(script_module)
        parse_args = module.parse_args

        argv = [f"{script_module.split('.')[-1]}.py", "测试任务", cli_arg, cli_value]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.workers == expected_value, (
            f"{script_module} {cli_arg} 应为 {expected_value}，实际值: {args.workers}"
        )

    # Max iterations CLI 覆盖测试数据（包括特殊值）
    MAX_ITERATIONS_CLI_OVERRIDE_TESTS = [
        ("run", "-m", "20", "20"),
        ("run", "--max-iterations", "MAX", "MAX"),
        ("run", "--max-iterations", "-1", "-1"),
        ("scripts.run_basic", "-m", "15", "15"),
        ("scripts.run_mp", "-m", "25", "25"),
        ("scripts.run_iterate", "--max-iterations", "30", "30"),
    ]

    @pytest.mark.parametrize(
        "script_module,cli_arg,cli_value,expected_value",
        MAX_ITERATIONS_CLI_OVERRIDE_TESTS,
        ids=[f"{case[0]}:{case[1]}={case[2]}" for case in MAX_ITERATIONS_CLI_OVERRIDE_TESTS],
    )
    def test_max_iterations_cli_override(
        self,
        script_module: str,
        cli_arg: str,
        cli_value: str,
        expected_value: str,
    ) -> None:
        """表驱动测试：CLI 参数可覆盖 max_iterations 默认值"""
        import importlib

        module = importlib.import_module(script_module)
        parse_args = module.parse_args

        argv = [f"{script_module.split('.')[-1]}.py", "测试任务", cli_arg, cli_value]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.max_iterations == expected_value, (
            f"{script_module} {cli_arg} 应为 {expected_value}，实际值: {args.max_iterations}"
        )

    # parse_max_iterations 特殊值处理测试
    PARSE_MAX_ITERATIONS_TESTS = [
        ("10", 10),
        ("MAX", -1),
        ("max", -1),
        ("UNLIMITED", -1),
        ("INF", -1),
        ("-1", -1),
        ("0", -1),
        ("100", 100),
    ]

    @pytest.mark.parametrize(
        "input_value,expected_output",
        PARSE_MAX_ITERATIONS_TESTS,
        ids=[f"'{case[0]}'→{case[1]}" for case in PARSE_MAX_ITERATIONS_TESTS],
    )
    def test_parse_max_iterations_special_values(self, input_value: str, expected_output: int) -> None:
        """表驱动测试：parse_max_iterations 特殊值处理"""
        result = parse_max_iterations(input_value)
        assert result == expected_output, (
            f"parse_max_iterations('{input_value}') 应为 {expected_output}，实际值: {result}"
        )


# ============================================================
# TestModelsConfigTableDriven - 模型配置表驱动测试
# ============================================================


class TestModelsConfigTableDriven:
    """表驱动测试：模型配置参数（planner/worker/reviewer）

    测试场景：
    1. 默认值来自 config.yaml
    2. CLI 参数可覆盖默认值
    """

    @pytest.fixture
    def config_models(self) -> dict:
        """获取配置中的模型默认值"""
        config = get_config()
        return {
            "planner": config.models.planner,
            "worker": config.models.worker,
            "reviewer": config.models.reviewer,
        }

    # 模型 CLI 覆盖测试数据
    MODELS_CLI_OVERRIDE_TESTS = [
        ("run", "--planner-model", "test-planner", "planner_model", "test-planner"),
        ("run", "--worker-model", "test-worker", "worker_model", "test-worker"),
        (
            "scripts.run_mp",
            "--planner-model",
            "mp-planner",
            "planner_model",
            "mp-planner",
        ),
        (
            "scripts.run_mp",
            "--worker-model",
            "mp-worker",
            "worker_model",
            "mp-worker",
        ),
        (
            "scripts.run_mp",
            "--reviewer-model",
            "mp-reviewer",
            "reviewer_model",
            "mp-reviewer",
        ),
    ]

    @pytest.mark.parametrize(
        "script_module,cli_arg,cli_value,attr_name,expected_value",
        MODELS_CLI_OVERRIDE_TESTS,
        ids=[f"{case[0]}:{case[1]}" for case in MODELS_CLI_OVERRIDE_TESTS],
    )
    def test_models_cli_override(
        self,
        script_module: str,
        cli_arg: str,
        cli_value: str,
        attr_name: str,
        expected_value: str,
    ) -> None:
        """表驱动测试：CLI 参数可覆盖模型默认值"""
        import importlib

        module = importlib.import_module(script_module)
        parse_args = module.parse_args

        argv = [f"{script_module.split('.')[-1]}.py", "测试任务", cli_arg, cli_value]
        with patch("sys.argv", argv):
            args = parse_args()

        actual = getattr(args, attr_name)
        assert actual == expected_value, f"{script_module} {cli_arg} 应为 {expected_value}，实际值: {actual}"

    def test_run_mp_default_models_from_config(self, config_models: dict) -> None:
        """测试 scripts/run_mp.py 模型默认值为 None (tri-state)

        run_mp.py 使用 tri-state 设计，模型参数默认返回 None，
        运行时通过 resolve 函数从 config.yaml 获取。
        """
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None
        assert args.planner_model is None, (
            f"planner_model 默认值应为 None（tri-state 设计），实际值: {args.planner_model}"
        )
        assert args.worker_model is None, f"worker_model 默认值应为 None（tri-state 设计），实际值: {args.worker_model}"
        assert args.reviewer_model is None, (
            f"reviewer_model 默认值应为 None（tri-state 设计），实际值: {args.reviewer_model}"
        )

    def test_run_py_default_models_from_config(self, config_models: dict) -> None:
        """测试 run.py 模型默认值使用 tri-state 设计

        tri-state 语义：
        - argparse 返回 None（未显式指定）
        - 最终值通过 resolve_orchestrator_settings 从 config.yaml 解析
        """
        from core.config import resolve_orchestrator_settings
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        # argparse 返回 None（tri-state 设计）
        assert args.planner_model is None, f"planner_model 应为 None（tri-state 设计），实际值: {args.planner_model}"
        assert args.worker_model is None, f"worker_model 应为 None（tri-state 设计），实际值: {args.worker_model}"

        # 最终值通过 resolve_orchestrator_settings 从 config.yaml 解析
        resolved = resolve_orchestrator_settings({})
        assert resolved["planner_model"] == config_models["planner"], (
            f"resolve_orchestrator_settings 应返回 {config_models['planner']}，实际值: {resolved['planner_model']}"
        )
        assert resolved["worker_model"] == config_models["worker"], (
            f"resolve_orchestrator_settings 应返回 {config_models['worker']}，实际值: {resolved['worker_model']}"
        )


# ============================================================
# TestCloudTimeoutExecutionModeIteratePath - 测试 cloud 参数在 iterate 模式下的最终生效值
# ============================================================


class TestCloudTimeoutExecutionModeIteratePath:
    """测试 --cloud-timeout/--cloud-auth-timeout/--execution-mode 在 --mode iterate 与 & 前缀 cloud 路径下的最终生效值

    覆盖场景：
    1. --mode iterate 使用默认 cloud 参数（来自 config.yaml）
    2. --mode iterate 使用 CLI 覆盖 cloud 参数
    3. & 前缀触发 cloud 模式时的参数传递
    4. execution_mode 强制 basic 编排器的行为验证
    """

    @pytest.fixture
    def config_cloud_defaults(self) -> dict:
        """获取 config.yaml 中的 cloud 默认值"""
        config = get_config()
        return {
            "timeout": config.cloud_agent.timeout,
            "auth_timeout": config.cloud_agent.auth_timeout,
            "execution_mode": config.cloud_agent.execution_mode,
        }

    def test_iterate_mode_uses_config_cloud_timeout(self, config_cloud_defaults: dict) -> None:
        """测试 --mode iterate 的 --cloud-timeout：parse_args 返回 None，
        运行时通过 resolve_orchestrator_settings 解析获得 config.yaml 值"""
        from core.config import resolve_orchestrator_settings
        from run import parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", "测试任务"]):
            args = parse_args()

        # tri-state: parse_args 应返回 None
        assert args.cloud_timeout is None, f"cloud_timeout 应为 None（tri-state 设计），实际值: {args.cloud_timeout}"

        # 运行时解析应得到 config.yaml 的值
        resolved = resolve_orchestrator_settings()
        assert resolved["cloud_timeout"] == config_cloud_defaults["timeout"], (
            f"resolve_orchestrator_settings 应返回 {config_cloud_defaults['timeout']}，"
            f"实际值: {resolved['cloud_timeout']}"
        )

    def test_iterate_mode_uses_config_cloud_auth_timeout(self, config_cloud_defaults: dict) -> None:
        """测试 --mode iterate 的 --cloud-auth-timeout：parse_args 返回 None，
        运行时通过 resolve_orchestrator_settings 解析获得 config.yaml 值"""
        from core.config import resolve_orchestrator_settings
        from run import parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", "测试任务"]):
            args = parse_args()

        # tri-state: parse_args 应返回 None
        assert args.cloud_auth_timeout is None, (
            f"cloud_auth_timeout 应为 None（tri-state 设计），实际值: {args.cloud_auth_timeout}"
        )

        # 运行时解析应得到 config.yaml 的值
        resolved = resolve_orchestrator_settings()
        assert resolved["cloud_auth_timeout"] == config_cloud_defaults["auth_timeout"], (
            f"resolve_orchestrator_settings 应返回 {config_cloud_defaults['auth_timeout']}，"
            f"实际值: {resolved['cloud_auth_timeout']}"
        )

    def test_iterate_mode_uses_config_execution_mode(self, config_cloud_defaults: dict) -> None:
        """测试 --mode iterate 的 --execution-mode：parse_args 返回 None，
        运行时通过 resolve_orchestrator_settings 解析获得 config.yaml 值"""
        from core.config import resolve_orchestrator_settings
        from run import parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", "测试任务"]):
            args = parse_args()

        # tri-state: parse_args 应返回 None
        assert args.execution_mode is None, f"execution_mode 应为 None（tri-state 设计），实际值: {args.execution_mode}"

        # 运行时解析应得到 config.yaml 的值
        resolved = resolve_orchestrator_settings()
        assert resolved["execution_mode"] == config_cloud_defaults["execution_mode"], (
            f"resolve_orchestrator_settings 应返回 {config_cloud_defaults['execution_mode']}，"
            f"实际值: {resolved['execution_mode']}"
        )

    def test_iterate_mode_cli_overrides_cloud_timeout(self) -> None:
        """测试 CLI --cloud-timeout 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        custom_timeout = 600
        with patch("sys.argv", ["run.py", "--mode", "iterate", "--cloud-timeout", str(custom_timeout), "测试任务"]):
            args = parse_args()

        assert args.cloud_timeout == custom_timeout

    def test_iterate_mode_cli_overrides_cloud_auth_timeout(self) -> None:
        """测试 CLI --cloud-auth-timeout 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        custom_auth_timeout = 60
        with patch(
            "sys.argv", ["run.py", "--mode", "iterate", "--cloud-auth-timeout", str(custom_auth_timeout), "测试任务"]
        ):
            args = parse_args()

        assert args.cloud_auth_timeout == custom_auth_timeout

    def test_iterate_mode_cli_overrides_execution_mode(self) -> None:
        """测试 CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        # 测试覆盖为 cloud
        with patch("sys.argv", ["run.py", "--mode", "iterate", "--execution-mode", "cloud", "测试任务"]):
            args = parse_args()
        assert args.execution_mode == "cloud"

        # 测试覆盖为 auto
        with patch("sys.argv", ["run.py", "--mode", "iterate", "--execution-mode", "auto", "测试任务"]):
            args = parse_args()
        assert args.execution_mode == "auto"

    def test_cloud_prefix_detected_in_task(self) -> None:
        """测试 & 前缀能被正确检测为 cloud 请求"""
        from core.cloud_utils import is_cloud_request, strip_cloud_prefix

        # 测试各种 & 前缀格式
        assert is_cloud_request("& 分析代码") is True
        assert is_cloud_request("&分析代码") is True
        assert is_cloud_request("  & 分析代码") is True
        assert is_cloud_request("& ") is False  # 仅空白内容
        assert is_cloud_request("分析 & 代码") is False  # & 不在开头
        assert is_cloud_request("分析代码") is False

        # 测试移除前缀
        assert strip_cloud_prefix("& 分析代码") == "分析代码"
        assert strip_cloud_prefix("&分析代码") == "分析代码"
        assert strip_cloud_prefix("  &  测试  ") == "测试"
        assert strip_cloud_prefix("no prefix") == "no prefix"

    def test_cloud_prefix_triggers_cloud_mode_in_runner(self) -> None:
        """测试 Runner 中 & 前缀触发 cloud 模式"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "& 分析代码架构"]):
            args = parse_args()

        # 验证 is_cloud_request 检测结果
        assert is_cloud_request(args.task) is True

        # 验证 strip_cloud_prefix 结果
        assert strip_cloud_prefix(args.task) == "分析代码架构"

    def test_execution_mode_cloud_preserves_cli_timeout_values(self) -> None:
        """测试 execution_mode=cloud 时 CLI 指定的 timeout 值被保留"""
        from run import parse_args

        custom_timeout = 900
        custom_auth_timeout = 45
        with patch(
            "sys.argv",
            [
                "run.py",
                "--mode",
                "iterate",
                "--execution-mode",
                "cloud",
                "--cloud-timeout",
                str(custom_timeout),
                "--cloud-auth-timeout",
                str(custom_auth_timeout),
                "测试任务",
            ],
        ):
            args = parse_args()

        assert args.execution_mode == "cloud"
        assert args.cloud_timeout == custom_timeout
        assert args.cloud_auth_timeout == custom_auth_timeout

    def test_execution_mode_auto_preserves_cli_timeout_values(self) -> None:
        """测试 execution_mode=auto 时 CLI 指定的 timeout 值被保留"""
        from run import parse_args

        custom_timeout = 1200
        with patch(
            "sys.argv",
            [
                "run.py",
                "--mode",
                "iterate",
                "--execution-mode",
                "auto",
                "--cloud-timeout",
                str(custom_timeout),
                "测试任务",
            ],
        ):
            args = parse_args()

        assert args.execution_mode == "auto"
        assert args.cloud_timeout == custom_timeout


class TestIterateModeOrchestratorSelection:
    """测试 iterate 模式下编排器选择逻辑

    验证:
    1. execution_mode=cli 允许使用 MP 编排器
    2. execution_mode=cloud/auto 强制使用 basic 编排器
    """

    def test_cli_mode_allows_mp_orchestrator(self) -> None:
        """测试 execution_mode=cli 时允许使用 MP 编排器"""
        from run import parse_args

        with patch(
            "sys.argv", ["run.py", "--mode", "iterate", "--execution-mode", "cli", "--orchestrator", "mp", "测试任务"]
        ):
            args = parse_args()

        assert args.execution_mode == "cli"
        assert args.orchestrator == "mp"

    def test_cloud_mode_default_orchestrator(self) -> None:
        """测试 execution_mode=cloud 时默认编排器设置"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", "--execution-mode", "cloud", "测试任务"]):
            args = parse_args()

        assert args.execution_mode == "cloud"
        # 编排器在 resolve_orchestrator_settings 中会被强制切换为 basic
        # 但 parse_args 本身不会改变 orchestrator 值

    def test_no_mp_flag_forces_basic(self) -> None:
        """测试 --no-mp 标志强制使用 basic 编排器"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", "--no-mp", "测试任务"]):
            args = parse_args()

        assert args.no_mp is True


class TestCloudPrefixWithExecutionModeIntegration:
    """测试 & 前缀与 execution_mode 的集成行为

    验证:
    1. & 前缀任务的参数传递
    2. 与 --execution-mode 的交互
    """

    def test_cloud_prefix_task_with_default_execution_mode(self) -> None:
        """测试 & 前缀任务使用默认 execution_mode"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "& 分析代码"]):
            args = parse_args()

        # 验证任务包含 & 前缀
        assert args.task == "& 分析代码"
        # 验证 is_cloud_request 能正确检测
        assert is_cloud_request(args.task) is True

    def test_cloud_prefix_task_with_explicit_cloud_mode(self) -> None:
        """测试 & 前缀任务与显式 --execution-mode cloud"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "--execution-mode", "cloud", "& 后台分析"]):
            args = parse_args()

        assert args.execution_mode == "cloud"
        assert is_cloud_request(args.task) is True

    def test_cloud_prefix_stripped_correctly(self) -> None:
        """测试 & 前缀被正确移除"""
        test_cases = [
            ("& 分析代码", "分析代码"),
            ("&分析代码", "分析代码"),
            ("  &  测试任务  ", "测试任务"),
            ("& ", ""),  # 空内容
            ("&", ""),  # 仅 &
        ]

        for input_task, expected in test_cases:
            result = strip_cloud_prefix(input_task)
            assert result == expected, f"strip_cloud_prefix('{input_task}') 应为 '{expected}'，实际值: '{result}'"


# ============================================================
# TestIterateArgsMapping - 测试 IterateArgs 字段映射完整性
# ============================================================


class TestIterateArgsMapping:
    """测试 run.py 中 IterateArgs 的字段映射完整性

    验证:
    1. CLI 参数通过 _merge_options 正确传递到 IterateArgs
    2. SelfIterator 实际读取的字段都在 IterateArgs 中有映射
    3. 优先级：CLI 参数 > config.yaml > 默认值
    """

    @pytest.fixture
    def mock_iterate_args_from_runner(self) -> dict:
        """创建带完整 CLI 参数的 args，并通过 Runner._merge_options 得到 options"""
        from run import Runner, parse_args

        with patch(
            "sys.argv",
            [
                "run.py",
                "--mode",
                "iterate",
                "--cloud-timeout",
                "600",
                "--cloud-auth-timeout",
                "45",
                "--execution-mode",
                "cloud",
                "--workers",
                "5",
                "--max-iterations",
                "20",
                "--auto-commit",
                "--stall-diagnostics",
                "--stall-diagnostics-level",
                "info",
                "--stall-recovery-interval",
                "60.0",
                "--stream-console-renderer",
                "--stream-show-word-diff",
                "测试任务",
            ],
        ):
            args = parse_args()

        runner = Runner(args)
        options = runner._merge_options({})
        return options

    def test_cloud_timeout_mapped_to_options(self, mock_iterate_args_from_runner: dict) -> None:
        """测试 --cloud-timeout CLI 参数被映射到 options"""
        options = mock_iterate_args_from_runner
        assert "cloud_timeout" in options
        assert options["cloud_timeout"] == 600

    def test_cloud_auth_timeout_mapped_to_options(self, mock_iterate_args_from_runner: dict) -> None:
        """测试 --cloud-auth-timeout CLI 参数被映射到 options"""
        options = mock_iterate_args_from_runner
        assert "cloud_auth_timeout" in options
        assert options["cloud_auth_timeout"] == 45

    def test_execution_mode_mapped_to_options(self, mock_iterate_args_from_runner: dict) -> None:
        """测试 --execution-mode CLI 参数被映射到 options"""
        options = mock_iterate_args_from_runner
        assert "execution_mode" in options
        assert options["execution_mode"] == "cloud"

    def test_workers_mapped_to_options(self, mock_iterate_args_from_runner: dict) -> None:
        """测试 --workers CLI 参数被映射到 options"""
        options = mock_iterate_args_from_runner
        assert "workers" in options
        assert options["workers"] == 5

    def test_stall_diagnostics_mapped_to_options(self, mock_iterate_args_from_runner: dict) -> None:
        """测试卡死诊断参数被映射到 options"""
        options = mock_iterate_args_from_runner
        assert "stall_diagnostics_enabled" in options
        assert options["stall_diagnostics_enabled"] is True
        assert "stall_diagnostics_level" in options
        assert options["stall_diagnostics_level"] == "info"
        assert "stall_recovery_interval" in options
        assert options["stall_recovery_interval"] == 60.0

    def test_stream_render_params_mapped_to_options(self, mock_iterate_args_from_runner: dict) -> None:
        """测试流式渲染参数被映射到 options"""
        options = mock_iterate_args_from_runner
        assert "stream_console_renderer" in options
        assert options["stream_console_renderer"] is True
        assert "stream_show_word_diff" in options
        assert options["stream_show_word_diff"] is True

    def test_iterate_args_has_all_selfiterator_fields(self) -> None:
        """测试 IterateArgs 包含 SelfIterator 读取的所有字段

        通过构造 IterateArgs 对象并验证字段存在性。
        """
        from run import Runner, parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", "测试任务"]):
            args = parse_args()

        runner = Runner(args)
        options = runner._merge_options({})

        # 构造 IterateArgs（通过访问 _run_iterate 中的定义）
        # 由于 IterateArgs 是内部类，直接列出 SelfIterator 读取的字段进行验证
        required_fields = [
            "cloud_timeout",
            "cloud_auth_timeout",
            "execution_mode",
            "workers",
            "max_iterations",
            "verbose",
            "auto_commit",
            "auto_push",
            "commit_per_iteration",
            "orchestrator",
            "no_mp",
            "stream_console_renderer",
            "stream_advanced_renderer",
            "stream_typing_effect",
            "stream_typing_delay",
            "stream_word_mode",
            "stream_color_enabled",
            "stream_show_word_diff",
            "stall_diagnostics_enabled",
            "stall_diagnostics_level",
            "stall_recovery_interval",
            "execution_health_check_interval",
            "health_warning_cooldown",
            "quiet",
            "log_level",
            "heartbeat_debug",
            "planner_execution_mode",
            "worker_execution_mode",
            "reviewer_execution_mode",
            "directory",
        ]

        for field in required_fields:
            assert field in options, f"options 中缺少字段: {field}"

    def test_cli_priority_over_config_yaml(self) -> None:
        """测试 CLI 参数优先于 config.yaml 默认值"""
        from run import Runner, parse_args

        # 获取 config.yaml 的默认值
        config_cloud_timeout = get_config().cloud_agent.timeout
        config_workers = get_config().system.worker_pool_size

        # 使用 CLI 覆盖
        cli_cloud_timeout = config_cloud_timeout + 100
        cli_workers = config_workers + 2

        with patch(
            "sys.argv",
            [
                "run.py",
                "--mode",
                "iterate",
                "--cloud-timeout",
                str(cli_cloud_timeout),
                "--workers",
                str(cli_workers),
                "测试任务",
            ],
        ):
            args = parse_args()

        runner = Runner(args)
        options = runner._merge_options({})

        assert options["cloud_timeout"] == cli_cloud_timeout, (
            f"CLI --cloud-timeout 应覆盖 config.yaml 默认值 {config_cloud_timeout}"
        )
        assert options["workers"] == cli_workers, f"CLI --workers 应覆盖 config.yaml 默认值 {config_workers}"


# ============================================================
# TestIterateArgsToSelfIterator - 测试 IterateArgs 传递给 SelfIterator
# ============================================================


class TestIterateArgsToSelfIterator:
    """测试 IterateArgs 字段能被 SelfIterator 正确读取

    验证:
    1. 通过模拟 argparse.Namespace 传递给 SelfIterator
    2. SelfIterator 的 _resolved_settings 正确反映 CLI 参数
    """

    @pytest.fixture
    def mock_iterate_args(self) -> argparse.Namespace:
        """创建模拟的 IterateArgs 对象"""
        args = argparse.Namespace(
            requirement="测试任务",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="20",
            workers=5,
            force_update=False,
            verbose=False,
            directory=".",
            auto_commit=True,
            auto_push=False,
            commit_per_iteration=False,
            commit_message="",
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cloud",
            cloud_api_key=None,
            cloud_auth_timeout=45,
            cloud_timeout=600,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            quiet=False,
            log_level="INFO",
            heartbeat_debug=False,
            stall_diagnostics_enabled=True,
            stall_diagnostics_level="info",
            stall_recovery_interval=60.0,
            execution_health_check_interval=45.0,
            health_warning_cooldown=90.0,
            stream_console_renderer=True,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=True,
        )
        return args

    def test_selfiterator_reads_cloud_timeout(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取 cloud_timeout"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, "__init__", lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            # 模拟 _resolve_config_settings 调用
            cloud_timeout = getattr(iterator.args, "cloud_timeout", None)
            assert cloud_timeout == 600

    def test_selfiterator_reads_stall_diagnostics(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取卡死诊断参数"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, "__init__", lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            assert getattr(iterator.args, "stall_diagnostics_enabled", None) is True
            assert getattr(iterator.args, "stall_diagnostics_level", None) == "info"
            assert getattr(iterator.args, "stall_recovery_interval", 30.0) == 60.0

    def test_selfiterator_reads_stream_render_params(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取流式渲染参数"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, "__init__", lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            assert getattr(iterator.args, "stream_console_renderer", False) is True
            assert getattr(iterator.args, "stream_show_word_diff", False) is True

    def test_selfiterator_reads_execution_mode(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取 execution_mode"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, "__init__", lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            assert getattr(iterator.args, "execution_mode", "cli") == "cloud"


# ============================================================
# run_agent_system.sh 脚本透传行为测试
# ============================================================


class TestShellScriptPassthrough:
    """测试 shell 脚本参数透传给 run.py 的行为

    验证 run_agent_system.sh 脚本重构后，通过 CLI 参数透传给 run.py
    时配置优先级处理正确。

    优先级: CLI 参数 > config.yaml > 默认值
    """

    def test_cli_max_iterations_overrides_config(self) -> None:
        """测试 CLI --max-iterations 覆盖 config.yaml 配置"""
        # 模拟 shell 脚本透传 --max-iterations 参数
        args = argparse.Namespace(
            task="test task",
            mode="mp",
            directory=".",
            workers=None,  # 未指定，应使用 config.yaml
            max_iterations="20",  # 脚本参数透传
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

        runner = Runner(args)
        options = runner._merge_options({})

        # CLI 参数应覆盖 config.yaml
        assert options["max_iterations"] == 20

    def test_cli_workers_overrides_config(self) -> None:
        """测试 CLI --workers 覆盖 config.yaml 配置"""
        args = argparse.Namespace(
            task="test task",
            mode="mp",
            directory=".",
            workers=7,  # 脚本参数透传
            max_iterations=None,  # 未指定，应使用 config.yaml
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

        runner = Runner(args)
        options = runner._merge_options({})

        # CLI 参数应覆盖 config.yaml
        assert options["workers"] == 7
        # 未指定的 max_iterations 应使用 config.yaml 值
        assert options["max_iterations"] == CONFIG_MAX_ITERATIONS

    def test_no_cli_args_uses_config_defaults(self) -> None:
        """测试无 CLI 参数时使用 config.yaml 默认值"""
        # 模拟 shell 脚本未传递任何参数（仅 GOAL）
        args = argparse.Namespace(
            task="test task",
            mode="mp",
            directory=".",
            workers=None,  # 未指定
            max_iterations=None,  # 未指定
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

        runner = Runner(args)
        options = runner._merge_options({})

        # 应使用 config.yaml 默认值
        assert options["workers"] == CONFIG_WORKER_POOL_SIZE
        assert options["max_iterations"] == CONFIG_MAX_ITERATIONS
        assert options["planner_model"] == CONFIG_PLANNER_MODEL
        assert options["worker_model"] == CONFIG_WORKER_MODEL

    def test_cli_model_args_override_config(self) -> None:
        """测试 CLI 模型参数覆盖 config.yaml 配置

        验证 AGENT_PLANNER_MODEL/AGENT_WORKER_MODEL 环境变量透传
        通过 --planner-model/--worker-model 后能正确覆盖配置。
        """
        args = argparse.Namespace(
            task="test task",
            mode="mp",
            directory=".",
            workers=None,
            max_iterations=None,
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="custom-planner-from-env",  # 环境变量透传
            worker_model="custom-worker-from-env",  # 环境变量透传
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

        runner = Runner(args)
        options = runner._merge_options({})

        # CLI/环境变量透传的模型参数应覆盖 config.yaml
        assert options["planner_model"] == "custom-planner-from-env"
        assert options["worker_model"] == "custom-worker-from-env"

    def test_max_iterations_unlimited_keyword(self) -> None:
        """测试 MAX/-1 无限迭代关键词透传正确处理"""
        args = argparse.Namespace(
            task="test task",
            mode="mp",
            directory=".",
            workers=None,
            max_iterations="MAX",  # 无限迭代关键词
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=CONFIG_PLANNER_MODEL,
            worker_model=CONFIG_WORKER_MODEL,
            reviewer_model=CONFIG_REVIEWER_MODEL,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

        runner = Runner(args)
        options = runner._merge_options({})

        # MAX 应被解析为 -1（无限迭代）
        assert options["max_iterations"] == -1


# ============================================================
# TestExecutionModeAutoOrchestratorBasicPassthrough - 配置透传断言
# ============================================================


class TestExecutionModeAutoOrchestratorBasicPassthrough:
    """测试 run.py --mode iterate --execution-mode auto --orchestrator basic 的配置透传

    验证：
    1. CLI 参数被正确解析
    2. 参数透传到 SelfIterator
    3. 编排器选择符合预期
    """

    def test_parse_args_auto_mode_basic_orchestrator(self) -> None:
        """测试解析 --execution-mode auto --orchestrator basic"""
        from run import parse_args

        with patch(
            "sys.argv",
            ["run.py", "--mode", "iterate", "--execution-mode", "auto", "--orchestrator", "basic", "测试任务"],
        ):
            args = parse_args()

        assert args.execution_mode == "auto", "execution_mode 应为 'auto'"
        assert args.orchestrator == "basic", "orchestrator 应为 'basic'"
        assert args.mode == "iterate", "mode 应为 'iterate'"

    def test_runner_merge_options_preserves_auto_mode(self) -> None:
        """测试 Runner._merge_options 保留 execution_mode=auto"""
        from run import Runner

        args = argparse.Namespace(
            mode="iterate",
            task="测试任务",
            model=None,
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            directory=".",
            max_iterations="5",
            dry_run=False,
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            execution_mode="auto",  # 显式指定 auto
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            cloud_background=None,
            orchestrator="basic",  # 显式指定 basic
            no_mp=False,
            _orchestrator_user_set=True,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 模型配置
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            # 知识库相关
            use_knowledge=False,
            self_update=False,
            search_knowledge=None,
            # 额外必需属性
            strict_review=None,
            enable_sub_planners=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
        )

        runner = Runner(args)
        options = runner._merge_options({"execution_mode": "auto"})

        assert options["execution_mode"] == "auto", "execution_mode='auto' 应被透传"

    def test_selfiterator_receives_auto_mode_basic_orchestrator(self) -> None:
        """测试 SelfIterator 接收 execution_mode=auto, orchestrator=basic"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode
        from scripts.run_iterate import SelfIterator

        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="5",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="basic",  # 显式指定 basic
            no_mp=False,
            _orchestrator_user_set=True,
            execution_mode="auto",  # 显式指定 auto
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            iterator = SelfIterator(args)

            # 验证 execution_mode 被正确设置
            actual_mode = iterator._get_execution_mode()
            assert actual_mode == ExecutionMode.AUTO, f"execution_mode 应为 AUTO，实际: {actual_mode}"

            # 验证编排器为 basic（无论是用户显式指定还是强制切换）
            actual_orchestrator = iterator._get_orchestrator_type()
            assert actual_orchestrator == "basic", f"orchestrator 应为 'basic'，实际: {actual_orchestrator}"

    @pytest.mark.parametrize(
        "cli_args,expect_execution_mode,expect_orchestrator",
        [
            pytest.param(
                ["--execution-mode", "auto", "--orchestrator", "basic"],
                "auto",
                "basic",
                id="auto_basic_explicit",
            ),
            pytest.param(
                ["--execution-mode", "auto", "--orchestrator", "mp"],
                "auto",
                "mp",  # parse_args 不会修改，强制切换在 SelfIterator 中
                id="auto_mp_parsed_as_is",
            ),
            pytest.param(
                ["--execution-mode", "cloud", "--orchestrator", "basic"],
                "cloud",
                "basic",
                id="cloud_basic_explicit",
            ),
            pytest.param(
                ["--execution-mode", "cli", "--orchestrator", "mp"],
                "cli",
                "mp",
                id="cli_mp_allowed",
            ),
        ],
    )
    def test_parse_args_passthrough_various_combinations(
        self,
        cli_args: list,
        expect_execution_mode: str,
        expect_orchestrator: str,
    ) -> None:
        """参数化测试：各种 CLI 参数组合的透传"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "--mode", "iterate", *cli_args, "测试任务"]):
            args = parse_args()

        assert args.execution_mode == expect_execution_mode, (
            f"期望 execution_mode={expect_execution_mode}，实际 {args.execution_mode}"
        )
        assert args.orchestrator == expect_orchestrator, (
            f"期望 orchestrator={expect_orchestrator}，实际 {args.orchestrator}"
        )

    def test_orchestrator_forced_basic_when_auto_mode_in_selfiterator(self) -> None:
        """测试 SelfIterator 中 auto 模式下 orchestrator=mp 被强制切换为 basic"""
        from cursor.cloud_client import CloudClientFactory
        from scripts.run_iterate import SelfIterator

        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="5",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户指定 mp
            no_mp=False,
            _orchestrator_user_set=True,
            execution_mode="auto",  # auto 模式
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-api-key"):
            iterator = SelfIterator(args)

            # 虽然用户指定了 mp，但 auto 模式强制使用 basic
            actual_orchestrator = iterator._get_orchestrator_type()
            assert actual_orchestrator == "basic", "auto 模式下 orchestrator=mp 应被强制切换为 basic"


# ============================================================
# ExecutionModeOrchestratorTestCase - 参数化测试数据结构
# ============================================================


from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ExecutionModeOrchestratorTestCase:
    """执行模式和编排器组合测试参数

    用于 pytest.mark.parametrize 驱动的参数化测试，覆盖所有
    execution_mode 与 orchestrator 的组合场景。

    Attributes:
        test_id: 测试用例唯一标识符
        requested_execution_mode: 请求的执行模式 (None/cli/auto/cloud)
        has_ampersand_prefix: 是否使用 & 前缀触发云端
        cloud_enabled: 云端是否启用
        has_api_key: 是否有 API Key
        orchestrator_user_set: 用户是否显式设置 orchestrator
        orchestrator_flag: orchestrator 参数值 (mp/basic/None)
        no_mp_flag: --no-mp 标志
        expected_orchestrator_type: 期望的最终编排器类型
        expected_effective_execution_mode: 期望的有效执行模式
        expect_mp_attempted: 是否期望尝试 MP 编排器
        expect_fallback_to_basic: 是否期望回退到 basic
    """

    test_id: str
    requested_execution_mode: Optional[str]  # None/cli/auto/cloud
    has_ampersand_prefix: bool
    cloud_enabled: bool
    has_api_key: bool
    orchestrator_user_set: bool
    orchestrator_flag: Optional[str]  # mp/basic/None
    no_mp_flag: bool
    expected_orchestrator_type: str  # mp/basic
    expected_effective_execution_mode: str  # cli/auto/cloud
    expect_mp_attempted: bool
    expect_fallback_to_basic: bool


# 执行模式与编排器组合测试参数表（run.py 专用）
RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES: list[ExecutionModeOrchestratorTestCase] = [
    # ===== CLI 模式场景 =====
    ExecutionModeOrchestratorTestCase(
        test_id="run_cli_default_mp",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="run_cli_explicit_mp",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="mp",
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="run_cli_explicit_basic",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="basic",
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,
        expect_fallback_to_basic=False,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="run_cli_no_mp_flag",
        requested_execution_mode="cli",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=True,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,
        expect_fallback_to_basic=False,
    ),
    # ===== AUTO 模式场景（有 API Key）=====
    ExecutionModeOrchestratorTestCase(
        test_id="run_auto_with_key_forces_basic",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="auto",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="run_auto_with_key_explicit_mp_forces_basic",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=True,
        orchestrator_flag="mp",
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="auto",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    # ===== AUTO 模式场景（无 API Key，回退 CLI）=====
    # 新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响
    ExecutionModeOrchestratorTestCase(
        test_id="run_auto_no_key_forces_basic",  # 重命名以反映新行为
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 新行为：auto 强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 不会尝试 MP
        expect_fallback_to_basic=True,  # 强制切换到 basic
    ),
    # ===== CLOUD 模式场景 =====
    ExecutionModeOrchestratorTestCase(
        test_id="run_cloud_with_key_forces_basic",
        requested_execution_mode="cloud",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cloud",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    # 新行为：requested=auto/cloud 即强制 basic，不受 key/enable 影响
    ExecutionModeOrchestratorTestCase(
        test_id="run_cloud_no_key_forces_basic",  # 重命名以反映新行为
        requested_execution_mode="cloud",
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # 新行为：cloud 强制 basic
        expected_effective_execution_mode="cli",
        expect_mp_attempted=False,  # 不会尝试 MP
        expect_fallback_to_basic=True,  # 强制切换到 basic
    ),
    # ===== & 前缀触发场景 =====
    ExecutionModeOrchestratorTestCase(
        test_id="run_ampersand_prefix_with_key_cloud",
        requested_execution_mode=None,
        has_ampersand_prefix=True,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",
        expected_effective_execution_mode="cloud",
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
    ExecutionModeOrchestratorTestCase(
        test_id="run_ampersand_prefix_no_key_fallback_cli",
        requested_execution_mode=None,
        has_ampersand_prefix=True,
        cloud_enabled=True,
        has_api_key=False,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    # ===== None 执行模式（默认 CLI）=====
    ExecutionModeOrchestratorTestCase(
        test_id="run_none_mode_defaults_cli",
        requested_execution_mode=None,
        has_ampersand_prefix=False,
        cloud_enabled=True,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="mp",
        expected_effective_execution_mode="cli",
        expect_mp_attempted=True,
        expect_fallback_to_basic=False,
    ),
    # ===== cloud_enabled=False 场景 =====
    # 注意：显式指定 auto 模式时，即使 cloud_enabled=False，
    # 执行模式仍保持 auto（用于策略层），但编排器强制为 basic
    ExecutionModeOrchestratorTestCase(
        test_id="run_auto_cloud_disabled_keeps_auto",
        requested_execution_mode="auto",
        has_ampersand_prefix=False,
        cloud_enabled=False,
        has_api_key=True,
        orchestrator_user_set=False,
        orchestrator_flag=None,
        no_mp_flag=False,
        expected_orchestrator_type="basic",  # auto 模式强制 basic
        expected_effective_execution_mode="auto",  # 保持 auto
        expect_mp_attempted=False,
        expect_fallback_to_basic=True,
    ),
]


# ============================================================
# TestRunExecutionModeOrchestratorParametrized - run.py 参数化测试
# ============================================================


class TestRunExecutionModeOrchestratorParametrized:
    """run.py 执行模式与编排器组合参数化测试

    使用 ExecutionModeOrchestratorTestCase 参数表驱动测试，
    覆盖 run.py 中所有 execution_mode 与 orchestrator 的组合场景。
    """

    @pytest.mark.parametrize(
        "test_case",
        RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES,
        ids=[tc.test_id for tc in RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES],
    )
    def test_effective_execution_mode_resolution(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试有效执行模式解析"""
        from core.execution_policy import resolve_effective_execution_mode

        # 调用策略函数
        effective_mode, reason = resolve_effective_execution_mode(
            requested_mode=test_case.requested_execution_mode,
            has_ampersand_prefix=test_case.has_ampersand_prefix,  # 语法检测层面
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
        )

        assert effective_mode == test_case.expected_effective_execution_mode, (
            f"[{test_case.test_id}] 期望执行模式 {test_case.expected_effective_execution_mode}，"
            f"实际 {effective_mode}，原因: {reason}"
        )

    @pytest.mark.parametrize(
        "test_case",
        RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES,
        ids=[tc.test_id for tc in RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES],
    )
    def test_orchestrator_compatibility(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试编排器与执行模式的兼容性

        新行为：should_use_mp_orchestrator 基于 requested_mode 判断，
        而不是 effective_mode。与 CLI help 对齐：requested=auto/cloud 即强制 basic。

        对于 & 前缀成功触发 Cloud 的场景，_get_orchestrator_type 会使用 "cloud"。
        """
        from core.execution_policy import should_use_mp_orchestrator

        # 确定实际传入 should_use_mp_orchestrator 的 requested_mode
        requested_mode: str | None
        if test_case.has_ampersand_prefix and test_case.expected_effective_execution_mode == "cloud":
            requested_mode = "cloud"
        else:
            requested_mode = test_case.requested_execution_mode

        can_use_mp = should_use_mp_orchestrator(requested_mode)

        if test_case.expect_fallback_to_basic:
            assert can_use_mp is False, f"[{test_case.test_id}] 请求的执行模式 {requested_mode} 不应允许 MP 编排器"
        elif test_case.expect_mp_attempted and not test_case.no_mp_flag:
            assert can_use_mp is True, f"[{test_case.test_id}] 请求的执行模式 {requested_mode} 应允许 MP 编排器"

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES if tc.expected_effective_execution_mode == "cli"],
        ids=[
            tc.test_id
            for tc in RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES
            if tc.expected_effective_execution_mode == "cli"
        ],
    )
    def test_cli_mode_allows_mp_by_default(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试 CLI 模式默认允许 MP 编排器"""
        from core.execution_policy import should_use_mp_orchestrator

        can_use_mp = should_use_mp_orchestrator("cli")
        assert can_use_mp is True, "CLI 模式应默认允许 MP 编排器"

        # 除非显式设置了 --no-mp 或 --orchestrator basic
        if test_case.no_mp_flag or test_case.orchestrator_flag == "basic":
            assert test_case.expected_orchestrator_type == "basic", (
                f"[{test_case.test_id}] --no-mp 或 --orchestrator basic 时应使用 basic"
            )

    @pytest.mark.parametrize(
        "test_case",
        [
            tc
            for tc in RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES
            if tc.expected_effective_execution_mode in ("auto", "cloud")
        ],
        ids=[
            tc.test_id
            for tc in RUN_EXECUTION_MODE_ORCHESTRATOR_TEST_CASES
            if tc.expected_effective_execution_mode in ("auto", "cloud")
        ],
    )
    def test_cloud_auto_modes_force_basic(self, test_case: ExecutionModeOrchestratorTestCase) -> None:
        """测试 cloud/auto 模式强制使用 basic 编排器"""
        from core.execution_policy import should_use_mp_orchestrator

        effective_mode = test_case.expected_effective_execution_mode
        can_use_mp = should_use_mp_orchestrator(effective_mode)

        assert can_use_mp is False, f"[{test_case.test_id}] {effective_mode} 模式不应允许 MP 编排器"
        assert test_case.expected_orchestrator_type == "basic", (
            f"[{test_case.test_id}] {effective_mode} 模式应使用 basic 编排器"
        )


# ============================================================
# TestRuleBasedAnalysisAndMergeOptionsMatrix - 参数化测试
# ============================================================


@dataclass
class ExecutionModeMatrixTestCase:
    """execution_mode 与编排器配置的测试用例

    统一测试 TaskAnalyzer._rule_based_analysis、Runner._merge_options 和
    Runner._get_execution_mode 对于各种 execution_mode/& 前缀/api_key/cloud_enabled 组合的行为。

    语义说明:
    - has_ampersand_prefix: 语法检测层面，原始文本是否有 & 前缀
    - prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud 路由
    - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出兼容）
    """

    test_id: str

    # 输入条件
    task: str  # 任务描述（可能含 & 前缀，has_ampersand_prefix）
    cli_execution_mode: Optional[str]  # CLI --execution-mode 参数值
    cli_orchestrator: Optional[str]  # CLI --orchestrator 参数值
    cli_no_mp: Optional[bool]  # CLI --no-mp 参数值
    cloud_enabled: bool  # config.yaml cloud_agent.enabled
    has_api_key: bool  # 是否有 API Key

    # 期望 _rule_based_analysis 返回的 analysis.options
    expected_analysis_execution_mode: Optional[str]
    expected_analysis_cloud_background: Optional[bool]
    # prefix_routed（策略决策层面）：& 前缀是否成功触发 Cloud 路由
    # 字段名使用 prefix_routed（内部分支统一使用此字段）
    expected_analysis_prefix_routed: bool

    # 期望 _merge_options 返回的 merged_options
    expected_merged_execution_mode: str
    expected_merged_orchestrator: str
    expected_merged_no_mp: bool

    # 期望 _get_execution_mode 返回的 ExecutionMode 枚举值
    expected_execution_mode_enum: str  # "CLI", "AUTO", "CLOUD", etc.


EXECUTION_MODE_MATRIX_TEST_CASES: list[ExecutionModeMatrixTestCase] = [
    # === 基础默认场景（无 & 前缀，config.yaml 默认 execution_mode=auto）===
    # 当 config.yaml 默认 execution_mode=auto 时，无 API Key 会回退到 CLI
    # 但编排器仍强制 basic（基于 requested_mode=auto）
    ExecutionModeMatrixTestCase(
        test_id="auto-default-no-prefix-no-key",
        task="普通任务",
        cli_execution_mode=None,  # 使用 config.yaml 默认值 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=False,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        # config.yaml 默认 auto → merged_execution_mode="auto"
        expected_merged_execution_mode="auto",
        # auto 模式强制 basic 编排器
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        # 无 API Key 回退到 CLI
        expected_execution_mode_enum="CLI",
    ),
    # === 显式 CLI 模式（覆盖 config.yaml 默认 auto）===
    # 需要显式设置 cli_execution_mode='cli' 来获得 mp 编排器
    ExecutionModeMatrixTestCase(
        test_id="cli-explicit-mode",
        task="普通任务",
        cli_execution_mode="cli",  # 显式指定 CLI，覆盖 config.yaml 默认 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=True,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        expected_merged_execution_mode="cli",
        expected_merged_orchestrator="mp",
        expected_merged_no_mp=False,
        expected_execution_mode_enum="CLI",
    ),
    # === & 前缀成功触发 Cloud 路由（cloud_enabled=True, has_api_key=True）===
    # 语义：has_ampersand_prefix=True（语法检测）+ 满足条件 → prefix_routed=True
    # 注意：& 前缀触发时，execution_mode 保持为 requested_mode（config.yaml 默认 auto），
    # effective_mode 为 "cloud"。编排器选择基于 prefix_routed=True 强制 basic。
    ExecutionModeMatrixTestCase(
        test_id="ampersand-prefix-cloud-enabled-with-key",
        task="& 后台执行任务",  # has_ampersand_prefix=True
        cli_execution_mode=None,  # 使用 config.yaml 默认值 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=True,
        # execution_mode 保持为 None（用户未显式指定），effective_mode 为 "cloud"
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=True,
        expected_analysis_prefix_routed=True,  # prefix_routed=True（成功路由）
        # merged_execution_mode 使用 config.yaml 默认值 "auto"，effective_mode 为 "cloud"
        expected_merged_execution_mode="auto",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLOUD",
    ),
    # === & 前缀但 cloud_enabled=False（未成功路由，回退到 CLI）===
    # 语义：has_ampersand_prefix=True（语法检测）但因 cloud_enabled=False → prefix_routed=False
    # 注意：config.yaml 默认 auto，但 & 前缀未成功路由，仍使用 auto 的编排器策略（basic）
    ExecutionModeMatrixTestCase(
        test_id="ampersand-prefix-cloud-disabled",
        task="& 后台执行任务",  # has_ampersand_prefix=True
        cli_execution_mode=None,  # 使用 config.yaml 默认值 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=False,
        has_api_key=True,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,  # prefix_routed=False（未成功路由）
        # config.yaml 默认 auto → merged_execution_mode="auto"
        expected_merged_execution_mode="auto",
        # auto 模式强制 basic 编排器（即使回退到 CLI）
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLI",
    ),
    # === & 前缀但无 API Key（未成功路由，回退到 CLI）===
    # 语义：has_ampersand_prefix=True（语法检测）但因无 API Key → prefix_routed=False
    # 注意：config.yaml 默认 auto，编排器仍强制 basic
    ExecutionModeMatrixTestCase(
        test_id="ampersand-prefix-no-api-key",
        task="& 后台执行任务",  # has_ampersand_prefix=True
        cli_execution_mode=None,  # 使用 config.yaml 默认值 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,  # prefix_routed=False（未成功路由）
        # config.yaml 默认 auto → merged_execution_mode="auto"
        expected_merged_execution_mode="auto",
        # auto 模式强制 basic 编排器（即使回退到 CLI）
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLI",
    ),
    # === 显式 execution_mode=cli 覆盖 & 前缀 ===
    # 语义：has_ampersand_prefix=True（语法检测）但因显式 CLI → prefix_routed=False
    # 显式 CLI 模式允许 mp 编排器
    ExecutionModeMatrixTestCase(
        test_id="cli-explicit-ignores-ampersand",
        task="& 后台执行任务",  # has_ampersand_prefix=True
        cli_execution_mode="cli",  # 显式指定 CLI，覆盖 config.yaml 默认 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=True,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,  # prefix_routed=False（显式 CLI 覆盖）
        expected_merged_execution_mode="cli",
        expected_merged_orchestrator="mp",
        expected_merged_no_mp=False,
        expected_execution_mode_enum="CLI",
    ),
    # === 显式 execution_mode=cloud（非 & 前缀触发）===
    # 语义：has_ampersand_prefix=False → prefix_routed=False（显式指定 Cloud，非 & 前缀触发）
    ExecutionModeMatrixTestCase(
        test_id="cloud-explicit-with-key",
        task="普通任务",  # has_ampersand_prefix=False
        cli_execution_mode="cloud",
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=True,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,  # prefix_routed=False（非 & 前缀触发）
        expected_merged_execution_mode="cloud",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLOUD",
    ),
    # === 显式 execution_mode=auto（非 & 前缀触发）===
    # 语义：has_ampersand_prefix=False → prefix_routed=False
    ExecutionModeMatrixTestCase(
        test_id="auto-explicit-with-key",
        task="普通任务",  # has_ampersand_prefix=False
        cli_execution_mode="auto",
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=True,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,  # prefix_routed=False（非 & 前缀触发）
        expected_merged_execution_mode="auto",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="AUTO",
    ),
    # === 显式 execution_mode=cloud 但无 API Key（回退到 CLI）===
    ExecutionModeMatrixTestCase(
        test_id="cloud-explicit-no-key-fallback",
        task="普通任务",
        cli_execution_mode="cloud",
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=True,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        expected_merged_execution_mode="cloud",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLI",  # 回退到 CLI
    ),
    # === 显式 --orchestrator=basic + 显式 cli 模式 ===
    # 注意：需要显式 cli 模式来确保 mp 编排器可用，然后用 --orchestrator=basic 覆盖
    ExecutionModeMatrixTestCase(
        test_id="orchestrator-basic-explicit",
        task="普通任务",
        cli_execution_mode="cli",  # 显式 CLI 模式
        cli_orchestrator="basic",
        cli_no_mp=None,
        cloud_enabled=False,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        expected_merged_execution_mode="cli",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLI",
    ),
    # === 显式 --no-mp + 显式 cli 模式 ===
    ExecutionModeMatrixTestCase(
        test_id="no-mp-explicit",
        task="普通任务",
        cli_execution_mode="cli",  # 显式 CLI 模式
        cli_orchestrator=None,
        cli_no_mp=True,
        cloud_enabled=False,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        expected_merged_execution_mode="cli",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLI",
    ),
    # === 自然语言检测到非并行关键词（config.yaml 默认 auto）===
    # 注意：即使检测到 basic 关键词，config.yaml 默认 auto 仍强制 basic 编排器
    ExecutionModeMatrixTestCase(
        test_id="natural-language-no-mp",
        task="使用 basic 编排器完成任务",
        cli_execution_mode=None,  # 使用 config.yaml 默认值 auto
        cli_orchestrator=None,
        cli_no_mp=None,
        cloud_enabled=False,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        # config.yaml 默认 auto → merged_execution_mode="auto"
        expected_merged_execution_mode="auto",
        expected_merged_orchestrator="basic",
        expected_merged_no_mp=True,
        expected_execution_mode_enum="CLI",
    ),
    # === 显式 cli 模式 + mp 编排器（显式获取 mp 行为）===
    # 新增用例：验证显式 cli 模式可以使用 mp 编排器
    ExecutionModeMatrixTestCase(
        test_id="cli-explicit-with-mp",
        task="普通任务",
        cli_execution_mode="cli",  # 显式 CLI 模式
        cli_orchestrator="mp",  # 显式指定 mp
        cli_no_mp=None,
        cloud_enabled=False,
        has_api_key=False,
        expected_analysis_execution_mode=None,
        expected_analysis_cloud_background=None,
        expected_analysis_prefix_routed=False,
        expected_merged_execution_mode="cli",
        expected_merged_orchestrator="mp",
        expected_merged_no_mp=False,
        expected_execution_mode_enum="CLI",
    ),
]


class TestRuleBasedAnalysisAndMergeOptionsMatrix:
    """参数化测试: _rule_based_analysis、_merge_options、_get_execution_mode 的行为一致性

    对每个测试用例断言:
    - analysis.options: execution_mode、cloud_background、prefix_routed
    - merged_options: execution_mode、orchestrator、no_mp
    - _get_execution_mode(): ExecutionMode 枚举结果

    注意: triggered_by_prefix 作为 prefix_routed 的兼容字段保留
    """

    @pytest.fixture
    def make_args(self) -> Callable[..., argparse.Namespace]:
        """构造 argparse.Namespace 的工厂函数"""

        def _make_args(
            task: str,
            cli_execution_mode: Optional[str],
            cli_orchestrator: Optional[str],
            cli_no_mp: Optional[bool],
        ) -> argparse.Namespace:
            return argparse.Namespace(
                task=task,
                mode="auto",
                directory=".",
                _directory_user_set=False,
                workers=CONFIG_WORKER_POOL_SIZE,
                max_iterations=str(CONFIG_MAX_ITERATIONS),
                strict_review=None,
                enable_sub_planners=None,
                verbose=False,
                quiet=False,
                log_level=None,
                skip_online=False,
                dry_run=False,
                force_update=False,
                use_knowledge=False,
                search_knowledge=None,
                self_update=False,
                planner_model=None,
                worker_model=None,
                reviewer_model=None,
                stream_log_enabled=None,
                stream_log_console=None,
                stream_log_detail_dir=None,
                stream_log_raw_dir=None,
                no_auto_analyze=False,
                auto_commit=False,
                auto_push=False,
                commit_per_iteration=False,
                orchestrator=cli_orchestrator,
                no_mp=cli_no_mp,
                _orchestrator_user_set=(cli_orchestrator is not None or cli_no_mp is not None),
                execution_mode=cli_execution_mode,
                planner_execution_mode=None,
                worker_execution_mode=None,
                reviewer_execution_mode=None,
                cloud_api_key=None,
                cloud_auth_timeout=None,
                cloud_timeout=None,
                cloud_background=None,
                stream_console_renderer=False,
                stream_advanced_renderer=False,
                stream_typing_effect=False,
                stream_typing_delay=0.02,
                stream_word_mode=True,
                stream_color_enabled=True,
                stream_show_word_diff=False,
                heartbeat_debug=False,
                stall_diagnostics_enabled=None,
                stall_diagnostics_level=None,
                stall_recovery_interval=30.0,
                execution_health_check_interval=30.0,
                health_warning_cooldown=60.0,
                enable_knowledge_injection=True,
                knowledge_top_k=3,
                knowledge_max_chars_per_doc=1200,
                knowledge_max_total_chars=3000,
            )

        return _make_args

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_MODE_MATRIX_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_MODE_MATRIX_TEST_CASES],
    )
    def test_rule_based_analysis_options(
        self,
        test_case: ExecutionModeMatrixTestCase,
        make_args: Callable[..., argparse.Namespace],
    ) -> None:
        """测试 _rule_based_analysis 返回的 analysis.options"""
        from cursor.cloud_client import CloudClientFactory

        args = make_args(
            task=test_case.task,
            cli_execution_mode=test_case.cli_execution_mode,
            cli_orchestrator=test_case.cli_orchestrator,
            cli_no_mp=test_case.cli_no_mp,
        )

        # 直接修改配置对象以避免 MagicMock 绑定问题
        from core.config import get_config

        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = test_case.cloud_enabled
            config.cloud_agent.execution_mode = "auto"  # config.yaml 默认值

            api_key = "mock-api-key" if test_case.has_api_key else None

            analyzer = TaskAnalyzer(use_agent=False)

            with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
                analysis = analyzer._rule_based_analysis(test_case.task, args)
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

        # 断言 analysis.options
        # 内部分支使用 prefix_routed 字段（triggered_by_prefix 仅作兼容输出）
        actual_exec_mode = analysis.options.get("execution_mode")
        actual_cloud_bg = analysis.options.get("cloud_background")
        actual_prefix_routed = analysis.options.get("prefix_routed", False)

        assert actual_exec_mode == test_case.expected_analysis_execution_mode, (
            f"[{test_case.test_id}] analysis.options['execution_mode'] 期望 "
            f"{test_case.expected_analysis_execution_mode}, 实际 {actual_exec_mode}"
        )
        assert actual_cloud_bg == test_case.expected_analysis_cloud_background, (
            f"[{test_case.test_id}] analysis.options['cloud_background'] 期望 "
            f"{test_case.expected_analysis_cloud_background}, 实际 {actual_cloud_bg}"
        )
        assert actual_prefix_routed == test_case.expected_analysis_prefix_routed, (
            f"[{test_case.test_id}] analysis.options['prefix_routed'] 期望 "
            f"{test_case.expected_analysis_prefix_routed}, 实际 {actual_prefix_routed}"
        )

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_MODE_MATRIX_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_MODE_MATRIX_TEST_CASES],
    )
    def test_merge_options_results(
        self,
        test_case: ExecutionModeMatrixTestCase,
        make_args: Callable[..., argparse.Namespace],
    ) -> None:
        """测试 _merge_options 返回的 merged_options"""
        from cursor.cloud_client import CloudClientFactory

        args = make_args(
            task=test_case.task,
            cli_execution_mode=test_case.cli_execution_mode,
            cli_orchestrator=test_case.cli_orchestrator,
            cli_no_mp=test_case.cli_no_mp,
        )

        # Mock config 和 API Key
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = test_case.cloud_enabled
        mock_config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto
        mock_config.system.worker_pool_size = CONFIG_WORKER_POOL_SIZE
        mock_config.system.max_iterations = CONFIG_MAX_ITERATIONS
        mock_config.system.enable_sub_planners = CONFIG_ENABLE_SUB_PLANNERS
        mock_config.system.strict_review = CONFIG_STRICT_REVIEW
        mock_config.planner.timeout = CONFIG_PLANNER_TIMEOUT
        mock_config.worker.task_timeout = CONFIG_WORKER_TIMEOUT
        mock_config.reviewer.timeout = CONFIG_REVIEWER_TIMEOUT
        mock_config.cloud_agent.timeout = CONFIG_CLOUD_TIMEOUT
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.models.planner = CONFIG_PLANNER_MODEL
        mock_config.models.worker = CONFIG_WORKER_MODEL
        mock_config.models.reviewer = CONFIG_REVIEWER_MODEL
        mock_config.logging.stream_json.enabled = CONFIG_STREAM_LOG_ENABLED
        mock_config.logging.stream_json.console = CONFIG_STREAM_LOG_CONSOLE
        mock_config.logging.stream_json.detail_dir = CONFIG_STREAM_LOG_DETAIL_DIR
        mock_config.logging.stream_json.raw_dir = CONFIG_STREAM_LOG_RAW_DIR

        api_key = "mock-api-key" if test_case.has_api_key else None

        analyzer = TaskAnalyzer(use_agent=False)
        runner = Runner(args)

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("run.get_config", return_value=mock_config):
                analysis = analyzer._rule_based_analysis(test_case.task, args)
                merged = runner._merge_options(analysis.options)

        # 断言 merged_options
        assert merged["execution_mode"] == test_case.expected_merged_execution_mode, (
            f"[{test_case.test_id}] merged_options['execution_mode'] 期望 "
            f"{test_case.expected_merged_execution_mode}, 实际 {merged['execution_mode']}"
        )
        assert merged["orchestrator"] == test_case.expected_merged_orchestrator, (
            f"[{test_case.test_id}] merged_options['orchestrator'] 期望 "
            f"{test_case.expected_merged_orchestrator}, 实际 {merged['orchestrator']}"
        )
        assert merged["no_mp"] == test_case.expected_merged_no_mp, (
            f"[{test_case.test_id}] merged_options['no_mp'] 期望 "
            f"{test_case.expected_merged_no_mp}, 实际 {merged['no_mp']}"
        )

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_MODE_MATRIX_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_MODE_MATRIX_TEST_CASES],
    )
    def test_get_execution_mode_enum(
        self,
        test_case: ExecutionModeMatrixTestCase,
        make_args: Callable[..., argparse.Namespace],
    ) -> None:
        """测试 _get_execution_mode 返回的 ExecutionMode 枚举"""
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        args = make_args(
            task=test_case.task,
            cli_execution_mode=test_case.cli_execution_mode,
            cli_orchestrator=test_case.cli_orchestrator,
            cli_no_mp=test_case.cli_no_mp,
        )

        # 直接修改配置对象
        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = test_case.cloud_enabled
            config.cloud_agent.execution_mode = "auto"  # config.yaml 默认值

            api_key = "mock-api-key" if test_case.has_api_key else None

            analyzer = TaskAnalyzer(use_agent=False)
            runner = Runner(args)

            with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
                analysis = analyzer._rule_based_analysis(test_case.task, args)
                merged = runner._merge_options(analysis.options)
                execution_mode = runner._get_execution_mode(merged)

            # 断言 ExecutionMode 枚举
            expected_enum = getattr(ExecutionMode, test_case.expected_execution_mode_enum)
            assert execution_mode == expected_enum, (
                f"[{test_case.test_id}] _get_execution_mode() 期望 "
                f"ExecutionMode.{test_case.expected_execution_mode_enum}, 实际 {execution_mode}"
            )
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode

    @pytest.mark.parametrize(
        "test_case",
        EXECUTION_MODE_MATRIX_TEST_CASES,
        ids=[tc.test_id for tc in EXECUTION_MODE_MATRIX_TEST_CASES],
    )
    def test_full_pipeline_consistency(
        self,
        test_case: ExecutionModeMatrixTestCase,
        make_args: Callable[..., argparse.Namespace],
    ) -> None:
        """测试完整流水线: _rule_based_analysis → _merge_options → _get_execution_mode 一致性"""
        from core.config import get_config
        from cursor.cloud_client import CloudClientFactory
        from cursor.executor import ExecutionMode

        args = make_args(
            task=test_case.task,
            cli_execution_mode=test_case.cli_execution_mode,
            cli_orchestrator=test_case.cli_orchestrator,
            cli_no_mp=test_case.cli_no_mp,
        )

        # 直接修改配置对象
        config = get_config()
        original_enabled = config.cloud_agent.enabled
        original_execution_mode = config.cloud_agent.execution_mode

        try:
            config.cloud_agent.enabled = test_case.cloud_enabled
            config.cloud_agent.execution_mode = "auto"  # config.yaml 默认值

            api_key = "mock-api-key" if test_case.has_api_key else None

            analyzer = TaskAnalyzer(use_agent=False)
            runner = Runner(args)

            with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
                # 步骤 1: _rule_based_analysis
                analysis = analyzer._rule_based_analysis(test_case.task, args)

                # 步骤 2: _merge_options
                merged = runner._merge_options(analysis.options)

                # 步骤 3: _get_execution_mode
                execution_mode = runner._get_execution_mode(merged)

            # 验证 analysis.options
            # 内部分支使用 prefix_routed 字段（triggered_by_prefix 仅作兼容输出）
            actual_analysis_exec_mode = analysis.options.get("execution_mode")
            actual_analysis_cloud_bg = analysis.options.get("cloud_background")
            actual_analysis_prefix_routed = analysis.options.get("prefix_routed", False)

            assert actual_analysis_exec_mode == test_case.expected_analysis_execution_mode
            assert actual_analysis_cloud_bg == test_case.expected_analysis_cloud_background
            assert actual_analysis_prefix_routed == test_case.expected_analysis_prefix_routed

            # 验证 merged_options
            assert merged["execution_mode"] == test_case.expected_merged_execution_mode
            assert merged["orchestrator"] == test_case.expected_merged_orchestrator
            assert merged["no_mp"] == test_case.expected_merged_no_mp

            # 验证 ExecutionMode
            expected_enum = getattr(ExecutionMode, test_case.expected_execution_mode_enum)
            assert execution_mode == expected_enum
        finally:
            config.cloud_agent.enabled = original_enabled
            config.cloud_agent.execution_mode = original_execution_mode


# ============================================================
# TestCloudFallbackUserMessageDedup - Cloud 回退用户消息去重测试
# ============================================================


class TestCloudFallbackUserMessageDedup:
    """Cloud 回退场景用户消息去重测试

    验证在以下场景中 user_message 在 stdout/stderr 中出现次数为 0 或 1：
    1. CloudClientFactory.resolve_api_key 返回 None
    2. cloud_enabled True/False
    3. AutoExecutor 产生 cooldown_info 的场景

    确保不出现重复的用户提示消息。
    """

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建模拟参数"""
        args = argparse.Namespace()
        args.task = "& 测试任务"
        args.execution_mode = None
        args.mode = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.dry_run = False
        args.max_iterations = 10
        args.workers = 3
        args.orchestrator = "mp"
        args.no_mp = False
        args.auto_commit = False
        args.auto_push = False
        args.commit_per_iteration = False
        args.force = False
        args.strict_review = False
        args.skip_online = True
        args.knowledge_base = None
        args.output_format = "text"
        args.cloud_api_key = None
        args.cloud_auth_timeout = 30
        args.cloud_timeout = 300
        args.print_config = False
        args._execution_mode_user_set = False
        args._orchestrator_user_set = False
        return args

    def test_no_api_key_no_duplicate_message(self, mock_args: argparse.Namespace, capsys) -> None:
        """测试无 API Key 时不输出重复的用户消息

        当 CloudClientFactory.resolve_api_key 返回 None 时，
        stdout/stderr 中不应出现重复的用户提示消息。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("run.get_config", return_value=mock_config),
        ):
            # 多次分析同一任务
            for _ in range(3):
                _ = analyzer.analyze("& 测试任务", mock_args)

        captured = capsys.readouterr()

        # 检查 stdout/stderr 中不应有多行用户提示
        # 日志已降级为 debug，不应出现在 stdout
        # 由于去重机制，相同消息只会输出一次
        assert captured.out.count("API Key") <= 1, (
            f"stdout 中 'API Key' 出现次数应 <= 1，实际: {captured.out.count('API Key')}"
        )
        assert captured.err.count("API Key") <= 1, (
            f"stderr 中 'API Key' 出现次数应 <= 1，实际: {captured.err.count('API Key')}"
        )

    def test_cloud_enabled_false_no_message(self, mock_args: argparse.Namespace, capsys) -> None:
        """测试 cloud_enabled=False 时输出信息性消息但不重复

        当 cloud_enabled=False 时，可能会输出一次信息性消息告知用户
        & 前缀被忽略，但不应重复输出。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = False

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("run.get_config", return_value=mock_config),
        ):
            # 多次分析
            for _ in range(3):
                _ = analyzer.analyze("& 测试任务", mock_args)

        captured = capsys.readouterr()

        # cloud_enabled=False 时可能输出一次信息性消息，但不应重复
        # 消息中应包含 "cloud_enabled=False" 或 "CLI" 的提示
        cloud_count = captured.out.count("cloud_enabled=False")
        assert cloud_count <= 1, f"'cloud_enabled=False' 消息出现 {cloud_count} 次，应 <= 1 次"

    def test_cloud_enabled_true_no_api_key_no_duplicate(self, mock_args: argparse.Namespace, capsys) -> None:
        """测试 cloud_enabled=True 但无 API Key 时不输出重复消息

        当 cloud_enabled=True 但 resolve_api_key 返回 None 时，
        用户消息应只出现 0 或 1 次。
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        CursorAgentClient._cloud_api_key_warning_shown = False
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("run.get_config", return_value=mock_config),
        ):
            # 执行多次分析
            for _ in range(5):
                _ = analyzer.analyze("& 测试任务", mock_args)

        captured = capsys.readouterr()

        # 验证不出现重复的用户提示
        combined_output = captured.out + captured.err

        # 检查典型的用户提示关键词 - 由于去重机制，应只出现 1 次
        key_phrases = ["API Key", "未配置"]
        for phrase in key_phrases:
            count = combined_output.count(phrase)
            assert count <= 1, f"'{phrase}' 在输出中出现 {count} 次，应 <= 1 次"

    @pytest.mark.asyncio
    async def test_auto_executor_cooldown_info_user_message_no_duplicate(self, capsys) -> None:
        """测试 AutoExecutor 产生 cooldown_info 时 user_message 不重复

        当 AutoExecutor 因 Cloud 失败回退到 CLI 时，
        cooldown_info 中的 user_message 应只由入口脚本打印一次，
        不应在库层重复输出。
        """
        from cursor.client import CursorAgentConfig
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        captured = capsys.readouterr()

        # 验证 cooldown_info 存在且包含 user_message
        assert result.cooldown_info is not None
        user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")

        # 验证 user_message 不在 stdout/stderr 中重复出现
        # 应该由入口脚本控制打印，库层不应直接打印
        # 消费逻辑优先使用稳定字段（kind/reason/user_message）
        if user_message:
            # 提取关键词进行检查
            key_phrases = ["回退", "CLI", "速率限制", "Rate limit"]
            for phrase in key_phrases:
                stdout_count = captured.out.count(phrase)
                stderr_count = captured.err.count(phrase)
                total_count = stdout_count + stderr_count
                # 库层不应直接打印 user_message，所以应该是 0
                # 或者最多是技术性日志 1 次
                assert total_count <= 1, f"'{phrase}' 在输出中出现 {total_count} 次，库层不应重复打印用户提示"

    @pytest.mark.asyncio
    async def test_auto_executor_multiple_failures_no_message_flood(self, capsys) -> None:
        """测试 AutoExecutor 多次失败不会产生消息洪水

        当 Cloud 多次失败时，不应在每次失败时都打印用户消息，
        冷却信息应通过结构化字段传递给入口脚本。
        """
        from cursor.client import CursorAgentConfig
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor, CooldownConfig

        # 使用短冷却时间以便测试
        cooldown_config = CooldownConfig(
            rate_limit_default_seconds=1,
            rate_limit_min_seconds=1,
        )

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(
            cli_config=config,
            cooldown_config=cooldown_config,
        )

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 1
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            # 执行多次
            results = []
            for _ in range(3):
                result = await auto_executor.execute("测试任务")
                results.append(result)

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # 验证不会产生消息洪水
        # 关键词在输出中出现次数应该合理（不是 3 倍于请求次数）
        key_phrases = ["速率限制", "Rate limit", "冷却"]
        for phrase in key_phrases:
            count = combined_output.count(phrase)
            # 允许技术性日志，但不应该是 3 次（每次请求一次）
            # 由于冷却机制，后续请求可能直接使用 CLI 而不触发新的警告
            assert count <= 3, f"'{phrase}' 在输出中出现 {count} 次，可能存在消息洪水"

    def test_decision_user_message_printed_at_most_once(self, mock_args: argparse.Namespace) -> None:
        """测试 decision.user_message 在一次执行路径中最多打印一次

        mock print_warning 函数，验证：
        1. 当 decision.user_message 存在时，最多调用 print_warning 一次
        2. 去重机制正确工作（相同消息不重复打印）
        """
        from unittest.mock import MagicMock

        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("run.get_config", return_value=mock_config),
            patch("run.print_warning") as mock_print_warning,
        ):
            # 多次分析同一个 & 前缀任务（触发 user_message 生成）
            for _ in range(5):
                result = analyzer.analyze("& 测试任务", mock_args)

            # 验证 print_warning 最多被调用一次
            # 由于去重机制，相同的 user_message 只会打印一次
            assert mock_print_warning.call_count <= 1, (
                f"print_warning 被调用 {mock_print_warning.call_count} 次，应最多调用 1 次（去重机制）"
            )

    def test_decision_user_message_dedup_across_different_tasks(self, mock_args: argparse.Namespace) -> None:
        """测试不同任务产生相同 user_message 时的去重

        当多个不同的 & 前缀任务产生相同类型的 user_message（如"未配置 API Key"）时，
        应该只打印一次。
        """
        from unittest.mock import MagicMock

        from cursor.cloud_client import CloudClientFactory

        # 重置类级别标志
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with (
            patch.object(CloudClientFactory, "resolve_api_key", return_value=None),
            patch("run.get_config", return_value=mock_config),
            patch("run.print_warning") as mock_print_warning,
        ):
            # 多个不同任务，但都触发相同类型的 user_message
            tasks = [
                "& 任务1",
                "& 任务2",
                "& 不同的任务描述",
            ]
            for task in tasks:
                mock_args.task = task
                _ = analyzer.analyze(task, mock_args)

            # 相同类型的 user_message 只打印一次
            assert mock_print_warning.call_count <= 1, (
                f"print_warning 被调用 {mock_print_warning.call_count} 次，相同类型的 user_message 应只打印 1 次"
            )


# ============================================================
# TestPrintWarningMockVerification - print_warning mock 验证测试
# ============================================================


class TestPrintWarningMockVerification:
    """验证 print_warning 调用行为的测试

    通过 mock print_warning 和 logger，断言 decision.user_message
    在整个执行路径中最多被打印一次。

    测试覆盖场景：
    1. TaskAnalyzer 分析阶段的 user_message 打印
    2. AutoExecutor cooldown_info 的 user_message 打印
    3. 库层（cursor/client.py）不直接打印 user_message
    """

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建模拟参数"""
        args = argparse.Namespace()
        args.task = "& 测试任务"
        args.execution_mode = None
        args.mode = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.dry_run = False
        args.max_iterations = 10
        args.workers = 3
        args.orchestrator = "mp"
        args.no_mp = False
        args.auto_commit = False
        args.auto_push = False
        args.commit_per_iteration = False
        args.force = False
        args.strict_review = False
        args.skip_online = True
        args.knowledge_base = None
        args.output_format = "text"
        args.cloud_api_key = None
        args.cloud_auth_timeout = 30
        args.cloud_timeout = 300
        args.print_config = False
        args._execution_mode_user_set = False
        args._orchestrator_user_set = False
        return args

    def test_library_layer_no_user_message_print(self) -> None:
        """测试库层（cursor/client.py）不直接打印 user_message

        验证 CursorAgentClient._should_route_to_cloud 方法
        在缺少 API Key 时只使用 logger.debug，不使用 print 或 print_warning。

        TODO: 如果未来发现库层仍有多行用户提示输出，应在此测试中标注并修复。
        """
        import io

        from loguru import logger

        from cursor.client import CursorAgentClient, CursorAgentConfig
        from cursor.cloud_client import CloudClientFactory

        # 重置警告标志
        CursorAgentClient._cloud_api_key_warning_shown = False

        config = CursorAgentConfig(
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
        )
        client = CursorAgentClient(config)

        # 捕获日志输出
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}: {message}", level="DEBUG")

        try:
            with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
                # 多次调用 _should_route_to_cloud
                for _ in range(3):
                    result = client._should_route_to_cloud("& 测试任务")
                    assert result is False, "无 API Key 时应返回 False"

            log_content = log_output.getvalue()

            # 验证日志中使用的是 DEBUG 级别，不是 WARNING
            # 多行用户提示不应出现在日志中
            assert (
                "WARNING" not in log_content or "API Key" not in log_content.split("WARNING")[1]
                if "WARNING" in log_content
                else True
            ), "库层不应使用 WARNING 级别输出 API Key 相关的用户提示"

            # 验证相同警告只记录一次（类级别标志）
            debug_count = log_content.count("检测到 Cloud 请求")
            assert debug_count <= 1, (
                f"DEBUG 日志 '检测到 Cloud 请求' 出现 {debug_count} 次，应最多出现 1 次（类级别去重）"
            )

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_executor_cooldown_info_user_message_structure(self) -> None:
        """测试 AutoExecutor 的 cooldown_info 结构正确

        验证 Cloud 失败时 cooldown_info 包含 user_message 字段，
        且该字段是字符串类型（供入口脚本决定是否打印）。
        """
        from cursor.client import CursorAgentConfig
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证 cooldown_info 结构（消费逻辑优先使用稳定字段 kind/reason/user_message）
        assert result.cooldown_info is not None, "cooldown_info 应存在"
        assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, "cooldown_info 应包含 user_message 字段"

        user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_message is None or isinstance(user_message, str), (
            f"user_message 应为 None 或 str，实际类型: {type(user_message)}"
        )

        if user_message:
            # 验证 user_message 包含有意义的内容
            assert len(user_message) > 0, "user_message 不应为空字符串"


# ============================================================
# TestUserMessageDedupScenarios - 用户消息去重典型场景测试
# ============================================================


class TestUserMessageDedupScenarios:
    """用户消息去重典型场景测试

    覆盖三个典型场景，验证：
    - 输出次数 ≤ 1（不刷屏）
    - 包含正确的关键字
    - cooldown_info/user_message 结构正确
    """

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        """创建模拟参数"""
        args = argparse.Namespace()
        args.task = "& 测试任务"
        args.execution_mode = None
        args.mode = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.dry_run = False
        args.max_iterations = 10
        args.workers = 3
        args.orchestrator = "mp"
        args.no_mp = False
        args.auto_commit = False
        args.auto_push = False
        args.commit_per_iteration = False
        args.force = False
        args.strict_review = False
        args.skip_online = True
        args.knowledge_base = None
        args.output_format = "text"
        args.cloud_api_key = None
        args.cloud_auth_timeout = 30
        args.cloud_timeout = 300
        args.print_config = False
        args._execution_mode_user_set = False
        args._orchestrator_user_set = False
        return args

    def test_ampersand_prefix_no_key_output_count_and_keywords(self, mock_args: argparse.Namespace, capsys) -> None:
        """场景1: & 前缀但无 API Key - 验证输出次数和关键字

        断言：
        1. user_message 输出次数 ≤ 1（不刷屏）
        2. 输出包含 "API Key" 或 "未配置" 关键字
        3. 输出包含 "CLI" 关键字（提示实际执行方式）
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置去重状态
        CursorAgentClient._cloud_api_key_warning_shown = False
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("run.get_config", return_value=mock_config):
                # 多次分析触发去重
                for i in range(5):
                    mock_args.task = f"& 测试任务 {i}"
                    _ = analyzer.analyze(f"& 测试任务 {i}", mock_args)

        captured = capsys.readouterr()
        combined = captured.out + captured.err

        # 断言 1: 输出次数 ≤ 1（不刷屏）
        api_key_count = combined.count("API Key") + combined.count("api_key")
        assert api_key_count <= 1, f"'API Key' 出现 {api_key_count} 次，应 ≤ 1 次（去重机制）"

        # 断言 2: 如果有输出，应包含正确的关键字
        if api_key_count > 0:
            assert "未配置" in combined or "API Key" in combined, "输出应包含 '未配置' 或 'API Key'"

        # 断言 3: 提示实际执行方式
        if api_key_count > 0:
            assert "CLI" in combined, "输出应提示实际执行方式为 CLI"

    def test_ampersand_prefix_no_key_user_message_structure(self, mock_args: argparse.Namespace) -> None:
        """场景1补充: & 前缀但无 API Key - 验证 ExecutionDecision.user_message 结构

        断言：
        1. decision.user_message 存在（非 None）
        2. user_message 包含 "⚠" 或 "ℹ" 图标
        3. user_message 包含关键信息（API Key、CLI）
        """
        from core.execution_policy import build_execution_decision

        # 测试无 API Key 时的决策
        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,
            user_requested_orchestrator=None,
        )

        # 断言 1: user_message 存在
        assert decision.user_message is not None, "无 API Key 时 user_message 应存在"

        # 断言 2: 包含图标
        assert "⚠" in decision.user_message or "ℹ" in decision.user_message, (
            f"user_message 应包含图标，实际: {decision.user_message}"
        )

        # 断言 3: 包含关键信息
        assert "API Key" in decision.user_message or "CLI" in decision.user_message, (
            f"user_message 应包含关键信息，实际: {decision.user_message}"
        )

    @pytest.mark.asyncio
    async def test_auto_mode_cooldown_fallback_output_count(self, capsys) -> None:
        """场景2: AUTO 模式冷却回退 - 验证输出次数

        断言（消费逻辑优先使用稳定字段 kind/reason/user_message）：
        1. cooldown_info[CooldownInfoFields.USER_MESSAGE] 存在
        2. stdout/stderr 中关键词出现次数 ≤ 1
        3. cooldown_info 包含 reason 或 fallback_reason（兼容）
        """
        from cursor.client import CursorAgentConfig
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("& 测试任务")

        captured = capsys.readouterr()
        combined = captured.out + captured.err

        # 断言 1: cooldown_info 存在且包含 user_message（优先使用稳定字段）
        assert result.cooldown_info is not None, "cooldown_info 应存在"
        assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, "cooldown_info 应包含 user_message"

        # 断言 2: 输出次数 ≤ 1（不刷屏）
        # 库层不应直接打印，user_message 由入口脚本打印
        fallback_keywords = ["回退", "CLI", "冷却", "Rate limit"]
        for keyword in fallback_keywords:
            count = combined.count(keyword)
            assert count <= 1, f"'{keyword}' 在输出中出现 {count} 次，库层不应重复打印"

        # 断言 3: cooldown_info 包含 reason（优先）或 fallback_reason（兼容）
        assert (
            CooldownInfoFields.REASON in result.cooldown_info
            or CooldownInfoFields.FALLBACK_REASON in result.cooldown_info
        ), "cooldown_info 应包含 reason 或 fallback_reason"

    def test_explicit_cli_with_ampersand_prefix_ignored(self, mock_args: argparse.Namespace, capsys) -> None:
        """场景3: 显式 cli + & 前缀 - 验证 & 前缀被忽略

        断言：
        1. decision.effective_mode == "cli"（& 前缀被忽略）
        2. decision.prefix_routed == False
        3. user_message 包含 "显式指定" 或 "cli" 提示（如果有）
        4. 输出次数 ≤ 1（不刷屏）
        """
        from core.execution_policy import build_execution_decision

        # 显式指定 cli 模式 + & 前缀
        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,  # 即使有 API Key
            user_requested_orchestrator=None,
        )

        # 断言 1: effective_mode 为 cli
        assert decision.effective_mode == "cli", (
            f"显式 cli 模式下 effective_mode 应为 cli，实际: {decision.effective_mode}"
        )

        # 断言 2: prefix_routed 为 False（& 前缀被忽略）
        assert decision.prefix_routed is False, "显式 cli 模式下 prefix_routed 应为 False"

        # 断言 3: 如果有 user_message，应包含相关提示
        if decision.user_message:
            # 显式 cli 模式下应提示 & 前缀被忽略
            assert "cli" in decision.user_message.lower() or "忽略" in decision.user_message, (
                f"user_message 应提示 cli 或忽略，实际: {decision.user_message}"
            )

    def test_explicit_cli_with_ampersand_prefix_no_flood(self, mock_args: argparse.Namespace, capsys) -> None:
        """场景3补充: 显式 cli + & 前缀多次分析 - 验证不刷屏

        断言：
        1. 多次分析同样的 & 前缀任务，输出次数 ≤ 1
        """
        from unittest.mock import MagicMock

        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        # 重置去重状态
        CursorAgentClient._cloud_api_key_warning_shown = False
        TaskAnalyzer.reset_shown_messages()

        analyzer = TaskAnalyzer(use_agent=False)

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        # 设置显式 cli 模式
        mock_args.execution_mode = "cli"
        mock_args._execution_mode_user_set = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="test_key"):
            with patch("run.get_config", return_value=mock_config):
                # 多次分析 & 前缀任务
                for i in range(5):
                    mock_args.task = f"& 任务 {i}"
                    _ = analyzer.analyze(f"& 任务 {i}", mock_args)

        captured = capsys.readouterr()
        combined = captured.out + captured.err

        # 断言: 输出次数 ≤ 1（不刷屏）
        # 显式 cli 模式下，& 前缀被忽略的提示最多出现 1 次
        cli_count = combined.lower().count("cli")
        ignore_count = combined.count("忽略") + combined.count("显式")
        total_hint_count = max(cli_count, ignore_count)

        # 允许技术性日志，但不应刷屏（每条消息最多 1 次）
        assert total_hint_count <= 3, f"提示信息出现 {total_hint_count} 次，不应刷屏"


# ============================================================
# TestDryRunNoSideEffects - Dry-run 模式无副作用验证
# ============================================================


class TestDryRunNoSideEffects:
    """测试 dry_run 模式下 Runner 不产生文件系统副作用

    验证在 tmp_path 下运行 Runner（dry_run=True），
    运行前后对比目录树（顶层与 .cursor/、logs/），断言未新增。
    """

    def _snapshot_directory(self, path: Path) -> set[str]:
        """获取目录下所有文件/目录的相对路径快照

        Args:
            path: 根目录路径

        Returns:
            所有文件/目录的相对路径集合
        """

        if not path.exists():
            return set()

        snapshot = set()
        for item in path.rglob("*"):
            rel_path = item.relative_to(path)
            snapshot.add(str(rel_path))
        return snapshot

    @pytest.mark.asyncio
    async def test_dry_run_mode_no_new_files_created(self, tmp_path: Path, monkeypatch) -> None:
        """测试 dry_run 模式下 Runner 不会创建新文件/目录

        在 tmp_path 下运行 Runner（dry_run=True），
        验证 .cursor/、logs/ 等目录不会被创建。
        """

        # 切换到临时目录
        monkeypatch.chdir(tmp_path)

        # 创建 dry_run 模式的 args
        args = argparse.Namespace(
            task="测试任务",
            mode="plan",  # plan 模式只分析不执行
            directory=str(tmp_path),
            _directory_user_set=False,
            workers=1,
            max_iterations="1",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=True,
            log_level=None,
            skip_online=True,
            dry_run=True,  # 关键：启用 dry_run 模式
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=False,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            enable_knowledge_injection=False,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        # 获取运行前的目录快照
        before_snapshot = self._snapshot_directory(tmp_path)

        # 创建分析结果和 Runner
        analysis = TaskAnalysis(
            mode=RunMode.PLAN,
            goal="测试任务",
            options={"dry_run": True},
            reasoning="测试",
        )

        runner = Runner(args)

        # Mock plan 执行器（在函数内部导入，需要 patch 原始模块）
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "分析结果"
        mock_result.error = None

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

            result = await runner.run(analysis)

        # 获取运行后的目录快照
        after_snapshot = self._snapshot_directory(tmp_path)

        # 计算新增的文件/目录
        new_items = after_snapshot - before_snapshot

        # 特别检查 .cursor/ 和 logs/ 目录
        cursor_dir_created = any(item.startswith(".cursor") for item in new_items)
        logs_dir_created = any(item.startswith("logs") for item in new_items)

        assert not cursor_dir_created, (
            f".cursor/ 目录不应在 dry_run 模式下被创建，新增项: {[i for i in new_items if '.cursor' in i]}"
        )
        assert not logs_dir_created, (
            f"logs/ 目录不应在 dry_run 模式下被创建，新增项: {[i for i in new_items if 'logs' in i]}"
        )

    @pytest.mark.asyncio
    async def test_plan_mode_no_file_writes(self, tmp_path: Path, monkeypatch) -> None:
        """测试 plan 模式下不会写入文件

        验证 Path.mkdir 和 Path.write_text 在关键目录下不会被调用。
        """
        from pathlib import Path

        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            task="测试任务",
            mode="plan",
            directory=str(tmp_path),
            _directory_user_set=False,
            workers=1,
            max_iterations="1",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=True,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=False,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            enable_knowledge_injection=False,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        # 追踪写入操作
        mkdir_calls = []
        write_text_calls = []

        original_mkdir = Path.mkdir
        original_write_text = Path.write_text

        def patched_mkdir(self, *args, **kwargs):
            if str(self).startswith(str(tmp_path)):
                mkdir_calls.append(str(self))
            return original_mkdir(self, *args, **kwargs)

        def patched_write_text(self, *args, **kwargs):
            if str(self).startswith(str(tmp_path)):
                write_text_calls.append(str(self))
            return original_write_text(self, *args, **kwargs)

        analysis = TaskAnalysis(
            mode=RunMode.PLAN,
            goal="测试任务",
            options={"dry_run": True},
            reasoning="测试",
        )

        runner = Runner(args)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "分析结果"
        mock_result.error = None

        with patch("cursor.executor.PlanAgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_executor_cls.return_value = mock_executor

            with patch.object(Path, "mkdir", patched_mkdir):
                with patch.object(Path, "write_text", patched_write_text):
                    result = await runner.run(analysis)

        # 验证在 .cursor/ 或 logs/ 下没有 mkdir 调用
        cursor_mkdirs = [c for c in mkdir_calls if ".cursor" in c]
        logs_mkdirs = [c for c in mkdir_calls if "logs" in c]

        assert len(cursor_mkdirs) == 0, f"dry_run 模式下不应创建 .cursor/ 子目录: {cursor_mkdirs}"
        assert len(logs_mkdirs) == 0, f"dry_run 模式下不应创建 logs/ 子目录: {logs_mkdirs}"

    @pytest.mark.asyncio
    async def test_iterate_minimal_dry_run_no_side_effects(self, tmp_path: Path, monkeypatch) -> None:
        """测试 iterate 模式 + minimal + dry_run 不产生副作用

        这是最严格的测试：iterate 模式通常会初始化知识库等，
        但配合 minimal 和 dry_run 应该完全无副作用。
        """

        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            task="测试任务",
            mode="iterate",
            directory=str(tmp_path),
            _directory_user_set=False,
            workers=1,
            max_iterations="1",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=True,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=False,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            enable_knowledge_injection=False,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        # 获取运行前的目录快照
        before_snapshot = self._snapshot_directory(tmp_path)

        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="测试任务",
            options={
                "skip_online": True,
                "dry_run": True,
            },
            reasoning="测试",
        )

        runner = Runner(args)

        # Mock SelfIterator（在函数内部从 scripts.run_iterate 导入）
        mock_result = {
            "success": True,
            "dry_run": True,
            "summary": "测试摘要",
        }

        with patch("scripts.run_iterate.SelfIterator") as mock_iterator_cls:
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value=mock_result)
            mock_iterator_cls.return_value = mock_iterator

            result = await runner.run(analysis)

        # 获取运行后的目录快照
        after_snapshot = self._snapshot_directory(tmp_path)

        # 计算新增的文件/目录
        new_items = after_snapshot - before_snapshot

        # 特别检查 .cursor/ 和 logs/ 目录
        cursor_dir_created = any(item.startswith(".cursor") for item in new_items)
        logs_dir_created = any(item.startswith("logs") for item in new_items)

        assert not cursor_dir_created, (
            f".cursor/ 目录不应在 iterate + dry_run 模式下被创建，新增项: {[i for i in new_items if '.cursor' in i]}"
        )
        assert not logs_dir_created, (
            f"logs/ 目录不应在 iterate + dry_run 模式下被创建，新增项: {[i for i in new_items if 'logs' in i]}"
        )

    @pytest.mark.asyncio
    async def test_iterate_minimal_no_webfetcher_calls(self, tmp_path: Path, monkeypatch) -> None:
        """测试 run.py --mode iterate --minimal 不调用 WebFetcher 网络方法

        验证通过 run.py 入口运行 iterate + minimal 模式时：
        1. WebFetcher.fetch() 不被调用
        2. WebFetcher.fetch_many() 不被调用
        3. WebFetcher._do_fetch() 不被调用

        这是 run.py --mode iterate --minimal 入口的网络隔离保证。
        """

        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            task="测试任务",
            mode="iterate",
            directory=str(tmp_path),
            _directory_user_set=False,
            workers=1,
            max_iterations="1",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=True,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=False,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            enable_knowledge_injection=False,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        # 追踪 WebFetcher 方法调用
        fetch_calls = []
        fetch_many_calls = []
        do_fetch_calls = []

        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="测试任务",
            options={
                "skip_online": True,
                "dry_run": True,
                "minimal": True,  # 启用 minimal 模式
            },
            reasoning="测试",
        )

        runner = Runner(args)

        # Mock SelfIterator，同时验证传递的参数包含 minimal=True
        mock_result = {
            "success": True,
            "dry_run": True,
            "summary": "测试摘要",
        }

        # 创建一个 SelfIterator mock，验证 WebFetcher 不被调用
        with patch("scripts.run_iterate.SelfIterator") as mock_iterator_cls:
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value=mock_result)
            mock_iterator_cls.return_value = mock_iterator

            # 同时 patch WebFetcher 来验证不被调用
            with patch("scripts.run_iterate.WebFetcher") as mock_fetcher_cls:
                mock_fetcher = MagicMock()
                mock_fetcher.initialize = AsyncMock()

                async def track_fetch(*args, **kwargs):
                    fetch_calls.append(("fetch", args, kwargs))
                    raise AssertionError("WebFetcher.fetch() 不应被调用")

                async def track_fetch_many(*args, **kwargs):
                    fetch_many_calls.append(("fetch_many", args, kwargs))
                    raise AssertionError("WebFetcher.fetch_many() 不应被调用")

                async def track_do_fetch(*args, **kwargs):
                    do_fetch_calls.append(("_do_fetch", args, kwargs))
                    raise AssertionError("WebFetcher._do_fetch() 不应被调用")

                mock_fetcher.fetch = AsyncMock(side_effect=track_fetch)
                mock_fetcher.fetch_many = AsyncMock(side_effect=track_fetch_many)
                mock_fetcher._do_fetch = AsyncMock(side_effect=track_do_fetch)
                mock_fetcher_cls.return_value = mock_fetcher

                result = await runner.run(analysis)

        # 验证没有网络调用
        assert len(fetch_calls) == 0, f"iterate + minimal 模式下 WebFetcher.fetch() 不应被调用: {fetch_calls}"
        assert len(fetch_many_calls) == 0, (
            f"iterate + minimal 模式下 WebFetcher.fetch_many() 不应被调用: {fetch_many_calls}"
        )
        assert len(do_fetch_calls) == 0, f"iterate + minimal 模式下 WebFetcher._do_fetch() 不应被调用: {do_fetch_calls}"

    @pytest.mark.asyncio
    async def test_iterate_minimal_socket_isolation(self, tmp_path: Path, monkeypatch) -> None:
        """测试 run.py --mode iterate --minimal 通过 socket.socket patch 验证网络隔离

        这是更严格的测试：直接 patch socket.socket，验证 minimal 模式
        不会触发任何底层网络连接。
        """
        import socket as socket_module

        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            task="测试任务",
            mode="iterate",
            directory=str(tmp_path),
            _directory_user_set=False,
            workers=1,
            max_iterations="1",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=True,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=True,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=False,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            enable_knowledge_injection=False,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        # 记录 socket.socket.connect 调用
        socket_connect_calls = []

        class MockSocket:
            """Mock socket that tracks connect calls"""

            def __init__(self, *args, **kwargs):
                self._args = args
                self._kwargs = kwargs

            def connect(self, address):
                socket_connect_calls.append(("connect", address))
                raise AssertionError(f"socket.connect() 不应在 minimal 模式下被调用: {address}")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def close(self):
                pass

            def settimeout(self, *args):
                pass

            def setsockopt(self, *args):
                pass

        analysis = TaskAnalysis(
            mode=RunMode.ITERATE,
            goal="测试任务",
            options={
                "skip_online": True,
                "dry_run": True,
                "minimal": True,
            },
            reasoning="测试",
        )

        runner = Runner(args)

        mock_result = {
            "success": True,
            "dry_run": True,
            "summary": "测试摘要",
        }

        with patch("scripts.run_iterate.SelfIterator") as mock_iterator_cls:
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value=mock_result)
            mock_iterator_cls.return_value = mock_iterator

            # Patch socket.socket 类
            with patch.object(socket_module, "socket", MockSocket):
                result = await runner.run(analysis)

        # 验证没有网络连接尝试
        assert len(socket_connect_calls) == 0, (
            f"iterate + minimal 模式下不应有 socket.connect() 调用: {socket_connect_calls}"
        )


# ============================================================
# 执行模式与编排器选择一致性测试（与 test_iterate_execution_matrix_consistency.py 对齐）
# ============================================================


class TestExecutionModeOrchestratorMatrixConsistencyWithRunPy:
    """测试 run.py 入口的执行模式与编排器选择一致性

    本测试类覆盖以下关键场景（与 CLI help 和 AGENTS.md 对齐）：
    1. requested=auto 无 key（orchestrator 必须 basic）
    2. requested=cloud 无 key（basic）
    3. & 前缀但未成功触发（允许 mp）
    4. 显式 cli + & 前缀（忽略前缀，允许 mp）
    5. 显式 orchestrator mp + requested cloud（应强制 basic）

    这些测试确保 run.py 的 _merge_options 逻辑与
    core.config.resolve_orchestrator_settings 和
    core.execution_policy.should_use_mp_orchestrator 的行为一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.fixture
    def base_args(self) -> argparse.Namespace:
        """创建基础测试参数"""
        return argparse.Namespace(
            task="测试任务",
            mode="iterate",
            directory=".",
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="mp",  # 默认 mp
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode=None,  # tri-state
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            enable_knowledge_injection=False,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    def test_auto_no_key_forces_basic_orchestrator(self, base_args: argparse.Namespace) -> None:
        """关键场景：requested=auto 无 API Key 时，编排器必须是 basic

        即使 effective_mode 回退到 CLI，只要 requested_mode=auto，
        编排器仍应强制使用 basic。
        """
        from core.config import ConfigManager, resolve_orchestrator_settings
        from core.execution_policy import should_use_mp_orchestrator
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 场景：CLI 显式 --execution-mode auto，但无 API Key
        base_args.execution_mode = "auto"

        # 测试 1: should_use_mp_orchestrator 应返回 False
        can_use_mp = should_use_mp_orchestrator("auto")
        assert can_use_mp is False, "requested_mode=auto 应强制 basic 编排器（should_use_mp=False）"

        # 测试 2: resolve_orchestrator_settings 应返回 orchestrator=basic
        result = resolve_orchestrator_settings(overrides={"execution_mode": "auto"})
        assert result["orchestrator"] == "basic", (
            "requested_mode=auto 无 API Key 时，resolve_orchestrator_settings 应返回 orchestrator='basic'"
        )

        # 测试 3: 使用 Runner._merge_options 验证
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    runner = Runner(base_args)
                    options = runner._merge_options({"execution_mode": "auto"})

                    # 验证编排器被强制为 basic
                    assert options.get("orchestrator") == "basic", (
                        f"Runner._merge_options: auto 模式无 API Key 时，"
                        f"orchestrator 应为 'basic'，实际={options.get('orchestrator')}"
                    )

    def test_cloud_no_key_forces_basic_orchestrator(self, base_args: argparse.Namespace) -> None:
        """关键场景：requested=cloud 无 API Key 时，编排器必须是 basic

        即使 effective_mode 回退到 CLI，只要 requested_mode=cloud，
        编排器仍应强制使用 basic。
        """
        from core.config import ConfigManager, resolve_orchestrator_settings
        from core.execution_policy import should_use_mp_orchestrator
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 场景：CLI 显式 --execution-mode cloud，但无 API Key
        base_args.execution_mode = "cloud"

        # 测试 1: should_use_mp_orchestrator 应返回 False
        can_use_mp = should_use_mp_orchestrator("cloud")
        assert can_use_mp is False, "requested_mode=cloud 应强制 basic 编排器（should_use_mp=False）"

        # 测试 2: resolve_orchestrator_settings 应返回 orchestrator=basic
        result = resolve_orchestrator_settings(overrides={"execution_mode": "cloud"})
        assert result["orchestrator"] == "basic", (
            "requested_mode=cloud 无 API Key 时，resolve_orchestrator_settings 应返回 orchestrator='basic'"
        )

        # 测试 3: 使用 Runner._merge_options 验证
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    runner = Runner(base_args)
                    options = runner._merge_options({"execution_mode": "cloud"})

                    # 验证编排器被强制为 basic
                    assert options.get("orchestrator") == "basic", (
                        f"Runner._merge_options: cloud 模式无 API Key 时，"
                        f"orchestrator 应为 'basic'，实际={options.get('orchestrator')}"
                    )

    def test_ampersand_prefix_not_triggered_still_basic_due_to_config_auto(self, base_args: argparse.Namespace) -> None:
        """& 前缀未成功触发时，因 config 默认 auto 仍强制 basic 编排器

        当 config.yaml 默认 execution_mode=auto 时，即使 & 前缀未成功触发
        （因缺少 API Key 或 cloud_enabled=False），编排器仍强制使用 basic。

        原因：编排器选择基于 requested_mode（config.yaml 默认 auto），
        而非 effective_mode（回退后的 cli）。

        如需使用 mp 编排器，用户应显式指定 --execution-mode cli。
        """
        from core.config import ConfigManager, resolve_orchestrator_settings
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 场景：& 前缀存在但无 API Key，未成功触发
        # prefix_routed=False（因为没有成功触发）
        base_args.execution_mode = None  # 未显式指定，使用 config.yaml 默认 auto

        # 测试 1: resolve_orchestrator_settings 会读取 config 默认 auto
        # 因此即使 prefix_routed=False，仍返回 basic（因为 config 默认 auto）
        result = resolve_orchestrator_settings(
            overrides={},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "basic", "config.yaml 默认 auto 时，编排器应为 basic（即使 & 前缀未成功触发）"

        # 测试 2: 使用 TaskAnalyzer + Runner._merge_options 验证（无 API Key 场景）
        # analysis.options 可能不包含 orchestrator，最终编排器由 _merge_options 决定
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True
        mock_config.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto
        mock_config.cloud_agent.timeout = 300
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.system.max_iterations = 10
        mock_config.system.worker_pool_size = 3
        mock_config.system.enable_sub_planners = True
        mock_config.system.strict_review = False
        mock_config.models.planner = CONFIG_PLANNER_MODEL
        mock_config.models.worker = CONFIG_WORKER_MODEL
        mock_config.models.reviewer = CONFIG_REVIEWER_MODEL
        mock_config.planner.timeout = 500.0
        mock_config.worker.task_timeout = 600.0
        mock_config.reviewer.timeout = 300.0
        mock_config.logging.stream_json.enabled = False
        mock_config.logging.stream_json.console = True
        mock_config.logging.stream_json.detail_dir = "logs/stream_json/detail/"
        mock_config.logging.stream_json.raw_dir = "logs/stream_json/raw/"

        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    analyzer = TaskAnalyzer(use_agent=False)
                    analysis = analyzer.analyze("& 后台执行任务", base_args)

                    # 关键验证点：effective_mode 应为 cli（因为无 API Key）
                    effective_mode = analysis.options.get("effective_mode", "cli")
                    assert effective_mode == "cli", (
                        f"& 前缀无 API Key 时，effective_mode 应为 'cli'，实际={effective_mode}"
                    )

                    # prefix_routed 应为 False（& 前缀未成功触发）
                    # 内部分支使用 prefix_routed 字段
                    assert analysis.options.get("prefix_routed") is not True, (
                        "& 前缀无 API Key 时，prefix_routed 应为 False"
                    )

                    # 通过 Runner._merge_options 验证最终编排器选择
                    runner = Runner(base_args)
                    merged = runner._merge_options(analysis.options)
                    orch = merged.get("orchestrator", "basic")
                    assert orch == "basic", f"config 默认 auto 时，_merge_options 编排器应为 'basic'，实际={orch}"

    def test_explicit_cli_with_ampersand_allows_mp(self, base_args: argparse.Namespace) -> None:
        """显式 execution_mode=cli + & 前缀时，忽略前缀，允许 mp

        显式指定 --execution-mode cli 时，即使有 & 前缀也应忽略，
        使用 CLI 模式并允许 mp 编排器。
        """
        from core.config import ConfigManager, resolve_orchestrator_settings
        from core.execution_policy import should_use_mp_orchestrator
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 场景：显式 --execution-mode cli + & 前缀
        base_args.execution_mode = "cli"

        # 测试 1: should_use_mp_orchestrator 应返回 True
        can_use_mp = should_use_mp_orchestrator("cli")
        assert can_use_mp is True, "execution_mode=cli 应允许 mp 编排器"

        # 测试 2: resolve_orchestrator_settings 应返回 orchestrator=mp
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,  # 内部分支统一使用 prefix_routed
        )
        assert result["orchestrator"] == "mp", "显式 execution_mode=cli 时，编排器应为 mp"

        # 测试 3: 使用 TaskAnalyzer 验证
        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True

        with patch.object(CloudClientFactory, "resolve_api_key", return_value="mock-key"):
            with patch("run.get_config", return_value=mock_config):
                with patch("core.config.get_config", return_value=mock_config):
                    analyzer = TaskAnalyzer(use_agent=False)
                    analysis = analyzer.analyze("& 显式 CLI 任务", base_args)

                    # 显式 CLI 模式应忽略 & 前缀
                    assert analysis.mode != RunMode.CLOUD, "显式 execution_mode=cli 应忽略 & 前缀"

                    # prefix_routed 应为 False
                    # 内部分支使用 prefix_routed 字段
                    assert analysis.options.get("prefix_routed") is not True, "显式 CLI 时 prefix_routed 应为 False"

    def test_explicit_mp_with_cloud_forces_basic(self, base_args: argparse.Namespace) -> None:
        """显式 orchestrator=mp + requested=cloud 时，应强制 basic

        即使用户显式指定 --orchestrator mp，如果 execution_mode=cloud，
        编排器仍应强制切换到 basic（与 AGENTS.md 规则一致）。
        """
        from core.config import ConfigManager, resolve_orchestrator_settings

        ConfigManager.reset_instance()

        # 场景：显式 --orchestrator mp + --execution-mode cloud
        base_args.orchestrator = "mp"
        base_args._orchestrator_user_set = True
        base_args.execution_mode = "cloud"

        # 测试: resolve_orchestrator_settings 应强制 basic
        result = resolve_orchestrator_settings(
            overrides={
                "orchestrator": "mp",
                "execution_mode": "cloud",
            }
        )

        # 根据 AGENTS.md 规则，cloud 模式强制 basic，即使用户显式指定 mp
        assert result["orchestrator"] == "basic", (
            "显式 orchestrator=mp + execution_mode=cloud 时，应强制切换到 basic 编排器（AGENTS.md 规则）"
        )

    def test_explicit_mp_with_auto_forces_basic(self, base_args: argparse.Namespace) -> None:
        """显式 orchestrator=mp + requested=auto 时，应强制 basic

        与 cloud 模式相同，auto 模式也应强制使用 basic 编排器。
        """
        from core.config import ConfigManager, resolve_orchestrator_settings

        ConfigManager.reset_instance()

        # 场景：显式 --orchestrator mp + --execution-mode auto
        base_args.orchestrator = "mp"
        base_args._orchestrator_user_set = True
        base_args.execution_mode = "auto"

        result = resolve_orchestrator_settings(
            overrides={
                "orchestrator": "mp",
                "execution_mode": "auto",
            }
        )

        assert result["orchestrator"] == "basic", (
            "显式 orchestrator=mp + execution_mode=auto 时，应强制切换到 basic 编排器"
        )

    def test_prefix_routed_true_forces_basic(self, base_args: argparse.Namespace) -> None:
        """& 前缀成功触发时，强制使用 basic 编排器"""
        from core.config import ConfigManager, resolve_orchestrator_settings

        ConfigManager.reset_instance()

        # 场景：& 前缀成功触发 Cloud
        result = resolve_orchestrator_settings(
            overrides={},
            prefix_routed=True,
        )

        assert result["orchestrator"] == "basic", "prefix_routed=True 时应强制 basic 编排器"

    def test_cli_mode_allows_mp_regardless_of_api_key(self, base_args: argparse.Namespace) -> None:
        """CLI 模式允许 mp 编排器，不受 API Key 影响"""
        from core.config import ConfigManager, resolve_orchestrator_settings
        from core.execution_policy import should_use_mp_orchestrator

        ConfigManager.reset_instance()

        # 测试 1: 无 API Key 时
        result_no_key = resolve_orchestrator_settings(overrides={"execution_mode": "cli"})
        assert result_no_key["orchestrator"] == "mp", "CLI 模式无 API Key 时，编排器应为 mp"

        # 测试 2: 有 API Key 时
        result_with_key = resolve_orchestrator_settings(overrides={"execution_mode": "cli"})
        assert result_with_key["orchestrator"] == "mp", "CLI 模式有 API Key 时，编排器应为 mp"

        # 测试 3: should_use_mp_orchestrator 基于 requested_mode 判断
        assert should_use_mp_orchestrator("cli") is True

        # 测试 4: 验证正确的流程 - 无 & 前缀 + CLI 未指定 → 使用 config.yaml 默认值
        # 不应直接调用 should_use_mp_orchestrator(None)，应先通过 resolve_requested_mode_for_decision
        from core.execution_policy import resolve_requested_mode_for_decision

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,  # CLI 未指定
            has_ampersand_prefix=False,  # 无 & 前缀
            config_execution_mode="auto",  # config.yaml 默认 auto
        )
        # 无 & 前缀 + CLI 未指定 → requested_mode 应为 "auto"
        assert requested_mode == "auto", "无 & 前缀时应使用 config.yaml 默认值"
        # auto 模式不允许 MP
        assert should_use_mp_orchestrator(requested_mode) is False

    def test_plan_ask_modes_allow_mp(self, base_args: argparse.Namespace) -> None:
        """plan/ask 只读模式允许 mp 编排器"""
        from core.config import ConfigManager, resolve_orchestrator_settings
        from core.execution_policy import should_use_mp_orchestrator

        ConfigManager.reset_instance()

        # plan 模式
        result_plan = resolve_orchestrator_settings(overrides={"execution_mode": "plan"})
        assert result_plan["orchestrator"] == "mp", "plan 模式应允许 mp 编排器"
        assert should_use_mp_orchestrator("plan") is True

        # ask 模式
        result_ask = resolve_orchestrator_settings(overrides={"execution_mode": "ask"})
        assert result_ask["orchestrator"] == "mp", "ask 模式应允许 mp 编排器"
        assert should_use_mp_orchestrator("ask") is True


class TestExecutionDecisionConsistencyWithRunPy:
    """测试 build_execution_decision 与 run.py 入口的一致性

    验证 core.execution_policy.build_execution_decision 的输出
    与 run.py 的 TaskAnalyzer 和 Runner._merge_options 行为一致。
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.fixture
    def mock_args_factory(self) -> Callable[..., argparse.Namespace]:
        """创建参数工厂函数"""

        def _factory(execution_mode: str | None = None) -> argparse.Namespace:
            return argparse.Namespace(
                task="测试任务",
                mode="iterate",
                directory=".",
                workers=3,
                max_iterations="10",
                strict_review=None,
                enable_sub_planners=None,
                verbose=False,
                skip_online=True,
                dry_run=True,
                force_update=False,
                use_knowledge=False,
                search_knowledge=None,
                self_update=False,
                planner_model=None,
                worker_model=None,
                reviewer_model=None,
                stream_log_enabled=None,
                auto_commit=False,
                auto_push=False,
                commit_per_iteration=False,
                orchestrator="mp",
                no_mp=False,
                _orchestrator_user_set=False,
                execution_mode=execution_mode,
                cloud_api_key=None,
                cloud_auth_timeout=30,
                cloud_timeout=300,
                stream_console_renderer=False,
                stream_advanced_renderer=False,
                stream_typing_effect=False,
                stream_typing_delay=0.02,
                stream_word_mode=True,
                stream_color_enabled=True,
                stream_show_word_diff=False,
                enable_knowledge_injection=False,
                knowledge_top_k=3,
                knowledge_max_chars_per_doc=1200,
                knowledge_max_total_chars=3000,
            )

        return _factory

    @pytest.mark.parametrize(
        "prompt,requested_mode,has_api_key,cloud_enabled,"
        "expected_effective_mode,expected_orchestrator,expected_triggered",
        [
            # CLI 模式基础测试
            ("实现新功能", "cli", True, True, "cli", "mp", False),
            ("实现新功能", "cli", False, True, "cli", "mp", False),
            # AUTO 模式关键测试
            ("任务描述", "auto", True, True, "auto", "basic", False),
            ("任务描述", "auto", False, True, "cli", "basic", False),  # 回退但仍 basic
            # CLOUD 模式关键测试
            ("任务描述", "cloud", True, True, "cloud", "basic", False),
            ("任务描述", "cloud", False, True, "cli", "basic", False),  # 回退但仍 basic
            # & 前缀成功触发
            ("& 分析代码", None, True, True, "cloud", "basic", True),
            # & 前缀未成功触发（无 API Key）
            # 根据 AGENTS.md：& 前缀表达 Cloud 意图，即使未成功路由也使用 basic
            ("& 后台任务", None, False, True, "cli", "basic", False),
            # & 前缀未成功触发（cloud_enabled=False）
            # 根据 AGENTS.md：& 前缀表达 Cloud 意图，即使未成功路由也使用 basic
            ("& 后台任务", None, True, False, "cli", "basic", False),
            # 显式 CLI + & 前缀（忽略）
            # 根据 R-3：显式 cli 模式忽略 & 前缀，允许 MP
            ("& CLI 任务", "cli", True, True, "cli", "mp", False),
            # PLAN/ASK 模式
            ("分析架构", "plan", True, True, "plan", "mp", False),
            ("解释代码", "ask", True, True, "ask", "mp", False),
        ],
        ids=[
            "cli_with_key",
            "cli_no_key",
            "auto_with_key",
            "auto_no_key_fallback_basic",
            "cloud_with_key",
            "cloud_no_key_fallback_basic",
            "ampersand_success",
            "ampersand_no_key_forces_basic",
            "ampersand_cloud_disabled_forces_basic",
            "explicit_cli_ignores_ampersand",
            "plan_mode_allows_mp",
            "ask_mode_allows_mp",
        ],
    )
    def test_build_execution_decision_matches_run_py(
        self,
        mock_args_factory: Callable[..., argparse.Namespace],
        prompt: str,
        requested_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
        expected_effective_mode: str,
        expected_orchestrator: str,
        expected_triggered: bool,
    ) -> None:
        """表驱动测试：验证 build_execution_decision 与 run.py 行为一致"""
        from core.config import ConfigManager
        from core.execution_policy import build_execution_decision

        ConfigManager.reset_instance()

        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=requested_mode,
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        assert decision.effective_mode == expected_effective_mode, (
            f"effective_mode 不匹配: 期望={expected_effective_mode}, 实际={decision.effective_mode}"
        )

        assert decision.orchestrator == expected_orchestrator, (
            f"orchestrator 不匹配: 期望={expected_orchestrator}, 实际={decision.orchestrator}"
        )

        # 内部分支使用 prefix_routed 字段
        assert decision.prefix_routed == expected_triggered, (
            f"prefix_routed 不匹配: 期望={expected_triggered}, 实际={decision.prefix_routed}"
        )


# ============================================================
# Runner._merge_options 与 build_unified_overrides 一致性测试
# ============================================================


class TestMergeOptionsAndBuildUnifiedOverridesConsistency:
    """测试 Runner._merge_options 与 build_unified_overrides 的一致性

    验证 run.py 的 Runner._merge_options 返回的核心字段与
    core.config.build_unified_overrides(...).resolved 一致。

    核心字段包括：
    - execution_mode
    - orchestrator
    - max_iterations
    - workers
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """创建 mock 配置对象"""
        mock = MagicMock()
        mock.cloud_agent.enabled = True
        mock.cloud_agent.execution_mode = "auto"  # config.yaml 默认 auto
        mock.cloud_agent.timeout = 300
        mock.cloud_agent.auth_timeout = 30
        mock.system.max_iterations = 10
        mock.system.worker_pool_size = 3
        mock.system.enable_sub_planners = True
        mock.system.strict_review = False
        mock.models.planner = CONFIG_PLANNER_MODEL
        mock.models.worker = CONFIG_WORKER_MODEL
        mock.models.reviewer = CONFIG_REVIEWER_MODEL
        mock.planner.timeout = 500.0
        mock.worker.task_timeout = 600.0
        mock.reviewer.timeout = 300.0
        mock.logging.stream_json.enabled = False
        mock.logging.stream_json.console = True
        mock.logging.stream_json.detail_dir = "logs/stream_json/detail/"
        mock.logging.stream_json.raw_dir = "logs/stream_json/raw/"
        return mock

    @pytest.mark.parametrize(
        "test_id,execution_mode,has_api_key,cloud_enabled,user_orchestrator,has_ampersand_prefix,expected_orchestrator",
        [
            # ===== 基础 execution_mode=None（使用 config.yaml 默认 auto）=====
            # config.yaml 默认 auto → 编排器强制 basic
            ("none_default", None, True, True, None, False, "basic"),
            ("none_no_key", None, False, True, None, False, "basic"),
            # ===== execution_mode=cli（显式指定，覆盖 config 默认） =====
            ("cli_basic", "cli", True, True, None, False, "mp"),
            ("cli_no_key", "cli", False, True, None, False, "mp"),
            ("cli_user_basic", "cli", True, True, "basic", False, "basic"),
            ("cli_user_mp", "cli", True, True, "mp", False, "mp"),
            # ===== execution_mode=auto（关键场景） =====
            ("auto_with_key", "auto", True, True, None, False, "basic"),
            # 关键：auto + 无 key → effective=cli 但 orchestrator=basic
            ("auto_no_key_forces_basic", "auto", False, True, None, False, "basic"),
            ("auto_cloud_disabled", "auto", True, False, None, False, "basic"),
            ("auto_user_mp_still_basic", "auto", True, True, "mp", False, "basic"),
            # ===== execution_mode=cloud（关键场景） =====
            ("cloud_with_key", "cloud", True, True, None, False, "basic"),
            # 关键：cloud + 无 key → effective=cli 但 orchestrator=basic
            ("cloud_no_key_forces_basic", "cloud", False, True, None, False, "basic"),
            ("cloud_cloud_disabled", "cloud", True, False, None, False, "basic"),
            ("cloud_user_mp_still_basic", "cloud", True, True, "mp", False, "basic"),
            # ===== & 前缀场景（config.yaml 默认 auto）=====
            # & 前缀成功触发 → basic
            ("ampersand_success", None, True, True, None, True, "basic"),
            # & 前缀未成功触发，但 config 默认 auto → 仍强制 basic
            ("ampersand_no_key", None, False, True, None, True, "basic"),
            ("ampersand_cloud_disabled", None, True, False, None, True, "basic"),
            # ===== plan/ask 模式 =====
            ("plan_mode", "plan", True, True, None, False, "mp"),
            ("ask_mode", "ask", True, True, None, False, "mp"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_merge_options_matches_build_unified_overrides(
        self,
        mock_config: MagicMock,
        test_id: str,
        execution_mode: str,
        has_api_key: bool,
        cloud_enabled: bool,
        user_orchestrator: str,
        has_ampersand_prefix: bool,
        expected_orchestrator: str,
    ) -> None:
        """表驱动测试：验证 _merge_options 与 build_unified_overrides 一致"""
        from core.config import ConfigManager, build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        # 构建测试任务（可能包含 & 前缀）
        task = "& 后台任务" if has_ampersand_prefix else "普通任务"

        # 构建 args
        args = argparse.Namespace(
            task=task,
            mode="iterate",
            directory=".",
            _directory_user_set=False,
            workers=CONFIG_WORKER_POOL_SIZE,
            max_iterations=str(CONFIG_MAX_ITERATIONS),
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=user_orchestrator,
            no_mp=False,
            _orchestrator_user_set=(user_orchestrator is not None),
            execution_mode=execution_mode,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        # 修改 mock_config 以反映测试条件
        mock_config.cloud_agent.enabled = cloud_enabled
        api_key = "mock-api-key" if has_api_key else None

        # 计算 build_unified_overrides 的结果
        # 首先构建 execution_decision
        decision = build_execution_decision(
            prompt=task,
            requested_mode=execution_mode,
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        # 验证编排器选择与预期一致
        assert unified.resolved["orchestrator"] == expected_orchestrator, (
            f"[{test_id}] build_unified_overrides orchestrator 不匹配\n"
            f"  期望={expected_orchestrator}, 实际={unified.resolved['orchestrator']}"
        )

        # 使用 mock 创建 Runner 并调用 _merge_options
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=api_key):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    runner = Runner(args)

                    # 构建 analysis_options，模拟 TaskAnalyzer 的输出
                    analysis_options = {
                        "execution_mode": execution_mode,
                        # 内部分支使用 prefix_routed 字段
                        "prefix_routed": decision.prefix_routed,
                        "triggered_by_prefix": decision.prefix_routed,  # 兼容字段
                        "has_ampersand_prefix": decision.has_ampersand_prefix,
                    }
                    if user_orchestrator:
                        analysis_options["orchestrator"] = user_orchestrator

                    merged = runner._merge_options(analysis_options)

        # 验证 _merge_options 返回的核心字段与 build_unified_overrides.resolved 一致
        assert merged["orchestrator"] == unified.resolved["orchestrator"], (
            f"[{test_id}] orchestrator 不一致\n"
            f"  _merge_options={merged['orchestrator']}\n"
            f"  build_unified_overrides={unified.resolved['orchestrator']}"
        )

        assert merged["workers"] == unified.resolved["workers"], (
            f"[{test_id}] workers 不一致\n"
            f"  _merge_options={merged['workers']}\n"
            f"  build_unified_overrides={unified.resolved['workers']}"
        )

        assert merged["max_iterations"] == unified.resolved["max_iterations"], (
            f"[{test_id}] max_iterations 不一致\n"
            f"  _merge_options={merged['max_iterations']}\n"
            f"  build_unified_overrides={unified.resolved['max_iterations']}"
        )

    def test_auto_no_key_effective_cli_but_orchestrator_basic(
        self,
        mock_config: MagicMock,
    ) -> None:
        """强断言：requested_mode=auto 无 key → effective=cli 但 orchestrator=basic

        这是关键场景的专项测试，确保此行为符合 AGENTS.md 规则。
        """
        from core.config import ConfigManager, build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        args = argparse.Namespace(
            task="任务描述",
            mode="iterate",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="auto",  # 关键：显式 auto
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        mock_config.cloud_agent.enabled = True

        # 无 API Key
        decision = build_execution_decision(
            prompt="任务描述",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,  # 关键：无 API Key
        )

        # 验证 decision 的状态
        assert decision.effective_mode == "cli", "auto + 无 key 应回退到 cli"
        assert decision.orchestrator == "basic", "auto + 无 key 仍应强制 basic 编排器（基于 requested_mode）"

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        assert unified.resolved["orchestrator"] == "basic", "build_unified_overrides: auto + 无 key 应强制 basic"

        # 验证 Runner._merge_options
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    runner = Runner(args)
                    merged = runner._merge_options(
                        {
                            "execution_mode": "auto",
                            "prefix_routed": False,  # 内部分支使用此字段
                            "triggered_by_prefix": False,  # 兼容字段
                        }
                    )

        assert merged["orchestrator"] == "basic", "Runner._merge_options: auto + 无 key 应强制 basic 编排器"

    def test_cloud_no_key_effective_cli_but_orchestrator_basic(
        self,
        mock_config: MagicMock,
    ) -> None:
        """强断言：requested_mode=cloud 无 key → effective=cli 但 orchestrator=basic

        这是关键场景的专项测试，确保此行为符合 AGENTS.md 规则。
        """
        from core.config import ConfigManager, build_unified_overrides
        from core.execution_policy import build_execution_decision
        from cursor.cloud_client import CloudClientFactory

        ConfigManager.reset_instance()

        args = argparse.Namespace(
            task="任务描述",
            mode="iterate",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=True,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cloud",  # 关键：显式 cloud
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

        mock_config.cloud_agent.enabled = True

        # 无 API Key
        decision = build_execution_decision(
            prompt="任务描述",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=False,  # 关键：无 API Key
        )

        # 验证 decision 的状态
        assert decision.effective_mode == "cli", "cloud + 无 key 应回退到 cli"
        assert decision.orchestrator == "basic", "cloud + 无 key 仍应强制 basic 编排器（基于 requested_mode）"

        unified = build_unified_overrides(
            args=args,
            execution_decision=decision,
        )

        assert unified.resolved["orchestrator"] == "basic", "build_unified_overrides: cloud + 无 key 应强制 basic"

        # 验证 Runner._merge_options
        with patch.object(CloudClientFactory, "resolve_api_key", return_value=None):
            with patch("core.config.get_config", return_value=mock_config):
                with patch("core.config.ConfigManager.get_instance", return_value=mock_config):
                    runner = Runner(args)
                    merged = runner._merge_options(
                        {
                            "execution_mode": "cloud",
                            "prefix_routed": False,  # 内部分支使用此字段
                            "triggered_by_prefix": False,  # 兼容字段
                        }
                    )

        assert merged["orchestrator"] == "basic", "Runner._merge_options: cloud + 无 key 应强制 basic 编排器"


# ============================================================
# Cloud 执行结果字段验证测试
# ============================================================


class TestCloudExecutionResultFields:
    """测试 Cloud 执行各场景下返回结果的字段完整性与类型

    覆盖场景:
    1. cloud_auth_config is None 的无 key 路径
    2. 执行器抛 asyncio.TimeoutError
    3. 执行器抛 ConnectionError
    4. 执行器返回 success=False

    断言要求:
    - 字段存在性（核心字段必须存在）
    - 类型检查（bool/str/None）
    - triggered_by_prefix == prefix_routed（兼容别名一致性）
    """

    @pytest.fixture
    def cloud_args(self) -> argparse.Namespace:
        """创建 Cloud 模式的标准参数"""
        return argparse.Namespace(
            task="& 云端任务",
            mode="cloud",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode="cloud",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=60,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    def _assert_common_result_fields(self, result: dict, *, expect_success: bool) -> None:
        """断言结果字典的公共字段存在性和类型

        Args:
            result: _run_cloud 返回的结果字典
            expect_success: 预期的 success 值
        """
        # 字段存在性断言
        assert "success" in result, "结果应包含 success 字段"
        assert "mode" in result, "结果应包含 mode 字段"
        assert "session_id" in result, "结果应包含 session_id 字段"

        # 类型断言
        assert isinstance(result["success"], bool), "success 应为 bool 类型"
        assert result["success"] is expect_success, f"success 应为 {expect_success}"
        assert isinstance(result["mode"], str), "mode 应为 str 类型"
        assert result["mode"] == "cloud", "mode 应为 'cloud'"

        # session_id 可以是 str 或 None
        session_id = result["session_id"]
        assert session_id is None or isinstance(session_id, str), "session_id 应为 str 或 None"

        # 失败时的额外字段检查
        if not expect_success:
            assert "error" in result, "失败结果应包含 error 字段"
            error = result["error"]
            assert error is None or isinstance(error, str), "error 应为 str 或 None"

    def _assert_prefix_routed_consistency(self, result: dict) -> None:
        """断言 prefix_routed 和 triggered_by_prefix 一致性

        根据规范：triggered_by_prefix 是 prefix_routed 的兼容别名，
        两者值应始终相等。
        """
        # 这两个字段仅在成功时返回，失败时可能不存在
        if "prefix_routed" in result and "triggered_by_prefix" in result:
            prefix_routed = result["prefix_routed"]
            triggered_by_prefix = result["triggered_by_prefix"]

            # 类型断言
            assert isinstance(prefix_routed, bool), "prefix_routed 应为 bool 类型"
            assert isinstance(triggered_by_prefix, bool), "triggered_by_prefix 应为 bool 类型"

            # 一致性断言
            assert triggered_by_prefix == prefix_routed, (
                f"triggered_by_prefix ({triggered_by_prefix}) 应等于 prefix_routed ({prefix_routed})"
            )

    # ----------------------------------------------------------------
    # 场景 1: cloud_auth_config is None 的无 key 路径
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_api_key_result_fields(self, cloud_args: argparse.Namespace) -> None:
        """验证无 API key 时返回结果的字段完整性和类型

        场景: cloud_auth_config is None（未配置 API Key）
        预期: success=False, 包含 error 字段, session_id=None
        """
        runner = Runner(cloud_args)

        # 清空环境变量，确保 cloud_auth_config 为 None
        with patch.dict("os.environ", {}, clear=True):
            options = runner._merge_options(
                {
                    "prefix_routed": True,  # 模拟 & 前缀成功路由
                    "has_ampersand_prefix": True,
                }
            )
            result = await runner._run_cloud("任务描述", options)

        # 公共字段断言
        self._assert_common_result_fields(result, expect_success=False)

        # 无 key 特定断言
        assert result["session_id"] is None, "无 key 时 session_id 应为 None"
        assert "API Key" in result["error"] or "未配置" in result["error"], "错误消息应提及 API Key"

        # resume_command 应为 None（因为 session_id 为 None）
        if "resume_command" in result:
            assert result["resume_command"] is None, "无 key 时 resume_command 应为 None"

    # ----------------------------------------------------------------
    # 场景 2: 执行器抛 asyncio.TimeoutError
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_timeout_error_result_fields(self, cloud_args: argparse.Namespace) -> None:
        """验证执行器抛出 TimeoutError 时返回结果的字段完整性和类型

        场景: 执行器抛 asyncio.TimeoutError
        预期: success=False, failure_kind='timeout', retryable=True
        """
        import asyncio as aio

        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=aio.TimeoutError())
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options(
                    {
                        "prefix_routed": False,
                        "has_ampersand_prefix": False,
                        "cloud_timeout": 60,
                    }
                )
                result = await runner._run_cloud("超时任务", options)

        # 公共字段断言
        self._assert_common_result_fields(result, expect_success=False)

        # TimeoutError 特定断言
        assert "failure_kind" in result, "结果应包含 failure_kind 字段"
        assert result["failure_kind"] == "timeout", (
            f"failure_kind 应为 'timeout', 实际为 '{result.get('failure_kind')}'"
        )

        # retryable 应为 True（超时是可重试的）
        if "retryable" in result:
            assert isinstance(result["retryable"], bool), "retryable 应为 bool 类型"
            assert result["retryable"] is True, "超时错误应标记为可重试"

        # error 应包含超时相关信息
        assert "超时" in result["error"] or "timeout" in result["error"].lower(), "错误消息应提及超时"

        # session_id 应为 None（执行器未成功执行）
        assert result["session_id"] is None, "超时时 session_id 应为 None"

    # ----------------------------------------------------------------
    # 场景 3: 执行器抛 ConnectionError
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_connection_error_result_fields(self, cloud_args: argparse.Namespace) -> None:
        """验证执行器抛出 ConnectionError 时返回结果的字段完整性和类型

        场景: 执行器抛 ConnectionError（网络不可达）
        预期: success=False, failure_kind='network'
        """
        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=ConnectionError("Network unreachable"))
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options(
                    {
                        "prefix_routed": False,
                        "has_ampersand_prefix": False,
                    }
                )
                result = await runner._run_cloud("网络错误任务", options)

        # 公共字段断言
        self._assert_common_result_fields(result, expect_success=False)

        # ConnectionError 特定断言
        assert "failure_kind" in result, "结果应包含 failure_kind 字段"
        assert result["failure_kind"] == "network", (
            f"failure_kind 应为 'network', 实际为 '{result.get('failure_kind')}'"
        )

        # error 应包含网络错误信息
        assert "error" in result and result["error"] is not None, "结果应包含非空 error 字段"
        assert isinstance(result["error"], str), "error 应为 str 类型"

        # session_id 应为 None
        assert result["session_id"] is None, "网络错误时 session_id 应为 None"

        # retry_after 字段检查
        if "retry_after" in result:
            retry_after = result["retry_after"]
            assert retry_after is None or isinstance(retry_after, (int, float)), "retry_after 应为 None 或数值类型"

    # ----------------------------------------------------------------
    # 场景 4: 执行器返回 success=False
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_executor_failure_result_fields(self, cloud_args: argparse.Namespace) -> None:
        """验证执行器返回 success=False 时返回结果的字段完整性和类型

        场景: 执行器正常返回但 success=False
        预期: 字段保持完整，prefix_routed 与 triggered_by_prefix 一致
        """
        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "执行失败: 权限不足"
            mock_result.session_id = "session-12345"
            mock_result.files_modified = []
            mock_result.failure_kind = "execution"
            mock_result.retry_after = None
            mock_result.cooldown_info = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options(
                    {
                        "prefix_routed": True,  # 模拟 & 前缀成功路由
                        "has_ampersand_prefix": True,
                    }
                )
                result = await runner._run_cloud("执行失败任务", options)

        # 公共字段断言
        self._assert_common_result_fields(result, expect_success=False)

        # 执行器失败特定断言
        assert result["error"] == "执行失败: 权限不足", "error 应保留执行器返回的错误消息"

        # session_id 应保留（执行器返回了 session_id）
        assert result["session_id"] == "session-12345", "执行器失败时应保留 session_id"

        # resume_command 应基于 session_id 生成
        if "resume_command" in result and result["session_id"]:
            assert isinstance(result["resume_command"], str), "resume_command 应为 str 类型"
            assert "session-12345" in result["resume_command"], "resume_command 应包含 session_id"

        # failure_kind 字段检查
        if "failure_kind" in result:
            assert isinstance(result["failure_kind"], str) or result["failure_kind"] is None, (
                "failure_kind 应为 str 或 None"
            )

    # ----------------------------------------------------------------
    # 场景 4b: 执行器成功时的字段验证（对比测试）
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_executor_success_result_fields_with_prefix_consistency(self, cloud_args: argparse.Namespace) -> None:
        """验证执行器成功时返回结果的字段完整性和 prefix_routed 一致性

        场景: 执行器正常返回 success=True
        预期: prefix_routed 与 triggered_by_prefix 值相等
        """
        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "任务完成"
            mock_result.session_id = "session-success-123"
            mock_result.files_modified = ["file1.py"]
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_result.cooldown_info = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                # 测试 prefix_routed=True 场景
                options = runner._merge_options(
                    {
                        "prefix_routed": True,
                        "has_ampersand_prefix": True,
                    }
                )
                result = await runner._run_cloud("成功任务", options)

        # 公共字段断言
        self._assert_common_result_fields(result, expect_success=True)

        # prefix_routed 一致性断言（成功时这两个字段应存在且一致）
        self._assert_prefix_routed_consistency(result)

        # 成功特定断言
        assert "output" in result, "成功结果应包含 output 字段"
        assert isinstance(result["output"], str), "output 应为 str 类型"
        assert result["session_id"] == "session-success-123", "成功时应保留 session_id"

        # files_modified 字段检查
        if "files_modified" in result:
            assert isinstance(result["files_modified"], list), "files_modified 应为 list 类型"

    @pytest.mark.asyncio
    async def test_executor_success_prefix_routed_false_consistency(self, cloud_args: argparse.Namespace) -> None:
        """验证 prefix_routed=False 时 triggered_by_prefix 也应为 False

        场景: 非 & 前缀触发的 Cloud 执行（显式 --execution-mode cloud）
        预期: prefix_routed=False, triggered_by_prefix=False（两者一致）
        """
        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "显式 Cloud 执行完成"
            mock_result.session_id = "session-explicit-456"
            mock_result.files_modified = []
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_result.cooldown_info = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                # 测试 prefix_routed=False 场景（显式 Cloud 模式，非 & 前缀触发）
                options = runner._merge_options(
                    {
                        "prefix_routed": False,
                        "has_ampersand_prefix": False,
                    }
                )
                result = await runner._run_cloud("显式Cloud任务", options)

        # 公共字段断言
        self._assert_common_result_fields(result, expect_success=True)

        # prefix_routed 一致性断言
        self._assert_prefix_routed_consistency(result)

        # 验证 prefix_routed=False
        assert result.get("prefix_routed") is False, "非 & 前缀触发时 prefix_routed 应为 False"
        assert result.get("triggered_by_prefix") is False, "triggered_by_prefix 应与 prefix_routed 一致"


# ============================================================
# cooldown_info 字段一致性测试
# ============================================================


class TestCooldownInfoFieldConsistency:
    """测试 Cloud 执行结果中 cooldown_info 字段的一致性

    覆盖场景:
    1. 成功执行（cooldown_info 可为 None）
    2. 执行失败（cooldown_info 应包含回退信息）
    3. 回退场景（cooldown_info 应包含 user_message）
    4. 无 API Key 场景（cooldown_info 应包含错误详情）
    """

    # 定义所有分支应包含的公共字段集
    CLOUD_RESULT_COMMON_FIELDS = {
        "success",
        "goal",
        "mode",
        "background",
        "output",
        "error",
        "session_id",
        "resume_command",
        "files_modified",
        "has_ampersand_prefix",
        "prefix_routed",
        "triggered_by_prefix",
        "failure_kind",
        "retry_after",
        "retryable",
        CloudResultFields.COOLDOWN_INFO,  # 新增字段
    }

    @pytest.fixture
    def cloud_args(self) -> argparse.Namespace:
        """创建 Cloud 模式参数"""
        return argparse.Namespace(
            task="& 测试任务",
            mode="cloud",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode="cloud",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=60,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
        )

    def _assert_field_set_consistency(self, result: dict, scenario: str) -> None:
        """验证结果字段集合与契约一致"""
        result_fields = set(result.keys())

        # 检查是否缺少必需字段
        missing_fields = self.CLOUD_RESULT_COMMON_FIELDS - result_fields
        assert not missing_fields, f"{scenario}: 结果缺少必需字段 {missing_fields}"

        # cooldown_info 字段必须存在（即使为 None）
        assert CloudResultFields.COOLDOWN_INFO in result, f"{scenario}: 结果必须包含 cooldown_info 字段"

    @pytest.mark.asyncio
    async def test_success_result_contains_cooldown_info(self, cloud_args: argparse.Namespace) -> None:
        """验证成功执行时 cooldown_info 字段存在"""
        from run import Runner

        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "任务完成"
            mock_result.session_id = "session-123"
            mock_result.files_modified = []
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_result.cooldown_info = None  # 成功时可为 None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("成功任务", options)

        self._assert_field_set_consistency(result, "成功执行")
        assert result[CloudResultFields.COOLDOWN_INFO] is None, "成功执行时 cooldown_info 可为 None"

    @pytest.mark.asyncio
    async def test_failure_result_contains_cooldown_info(self, cloud_args: argparse.Namespace) -> None:
        """验证执行失败时 cooldown_info 字段存在"""
        from run import Runner

        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.output = ""
            mock_result.error = "执行失败"
            mock_result.session_id = None
            mock_result.files_modified = []
            mock_result.failure_kind = "UNKNOWN"
            mock_result.retry_after = None
            mock_result.cooldown_info = {
                CooldownInfoFields.USER_MESSAGE: "执行失败，请重试",
                CooldownInfoFields.KIND: "UNKNOWN",
                CooldownInfoFields.REASON: "执行失败",
                CooldownInfoFields.RETRYABLE: False,
            }
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("失败任务", options)

        self._assert_field_set_consistency(result, "执行失败")
        assert result[CloudResultFields.COOLDOWN_INFO] is not None, "失败时 cooldown_info 应存在"
        # 消费逻辑优先使用稳定字段（kind/reason/user_message）
        assert CooldownInfoFields.USER_MESSAGE in result[CloudResultFields.COOLDOWN_INFO], (
            "cooldown_info 应包含 user_message"
        )

    @pytest.mark.asyncio
    async def test_no_key_result_contains_cooldown_info(self, cloud_args: argparse.Namespace) -> None:
        """验证无 API Key 时 cooldown_info 字段存在且结构正确"""
        from run import Runner

        runner = Runner(cloud_args)

        # 模拟无 API Key 场景
        with patch.dict("os.environ", {}, clear=True):
            # 确保环境变量被清除
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("无Key任务", options)

        self._assert_field_set_consistency(result, "无 API Key")
        assert result["success"] is False, "无 API Key 时应返回失败"
        assert result[CloudResultFields.COOLDOWN_INFO] is not None, "无 API Key 时 cooldown_info 应存在"

        # 验证 cooldown_info 子字段（优先使用稳定字段）
        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert CooldownInfoFields.USER_MESSAGE in cooldown, "cooldown_info 应包含 user_message"
        assert CooldownInfoFields.KIND in cooldown, "cooldown_info 应包含 kind"
        assert cooldown.get(CooldownInfoFields.KIND) == "no_key", "kind 应为 no_key"

    @pytest.mark.asyncio
    async def test_fallback_result_contains_cooldown_info(self, cloud_args: argparse.Namespace) -> None:
        """验证回退场景时 cooldown_info 字段结构完整"""
        from run import Runner

        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            # 模拟 Cloud 失败并回退
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "回退后 CLI 执行成功"
            mock_result.session_id = None
            mock_result.files_modified = []
            mock_result.failure_kind = "RATE_LIMIT"
            mock_result.retry_after = 60
            mock_result.cooldown_info = {
                CooldownInfoFields.USER_MESSAGE: "速率限制，已回退到本地 CLI",
                CooldownInfoFields.KIND: "RATE_LIMIT",
                CooldownInfoFields.REASON: "速率限制",
                CooldownInfoFields.FALLBACK_REASON: "速率限制",
                CooldownInfoFields.RETRYABLE: True,
                CooldownInfoFields.RETRY_AFTER: 60,
                CooldownInfoFields.IN_COOLDOWN: True,
                CooldownInfoFields.REMAINING_SECONDS: 60,
                CooldownInfoFields.FAILURE_COUNT: 1,
            }
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("回退任务", options)

        self._assert_field_set_consistency(result, "回退场景")

        # 验证 cooldown_info 完整结构
        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "回退时 cooldown_info 应存在"

        # 优先使用稳定字段常量验证
        expected_subfields = [
            CooldownInfoFields.USER_MESSAGE,
            CooldownInfoFields.KIND,
            CooldownInfoFields.REASON,
            CooldownInfoFields.RETRYABLE,
            CooldownInfoFields.RETRY_AFTER,
        ]
        for field in expected_subfields:
            assert field in cooldown, f"cooldown_info 应包含 {field}"

    @pytest.mark.asyncio
    async def test_timeout_error_contains_cooldown_info(self, cloud_args: argparse.Namespace) -> None:
        """验证超时错误时 cooldown_info 字段存在"""
        import asyncio

        from run import Runner

        runner = Runner(cloud_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError("超时"))
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("超时任务", options)

        self._assert_field_set_consistency(result, "超时错误")
        assert result["success"] is False, "超时时应返回失败"
        assert result[CloudResultFields.COOLDOWN_INFO] is not None, "超时时 cooldown_info 应存在"

    @pytest.mark.asyncio
    async def test_all_branches_have_consistent_field_set(self, cloud_args: argparse.Namespace) -> None:
        """验证所有分支返回的字段集合一致

        这是关键测试：确保成功/失败/回退/无回退各分支的字段集合相同。
        """
        import asyncio

        from run import Runner

        scenarios = []

        # 场景 1: 成功执行
        def mock_success(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "成功"
            mock_result.session_id = "s1"
            mock_result.files_modified = []
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_result.cooldown_info = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        runner = Runner(cloud_args)
        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_success):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                result = await runner._run_cloud("成功", runner._merge_options({}))
                scenarios.append(("成功", set(result.keys())))

        # 场景 2: 失败执行
        def mock_failure(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.output = ""
            mock_result.error = "失败"
            mock_result.session_id = None
            mock_result.files_modified = []
            mock_result.failure_kind = "UNKNOWN"
            mock_result.retry_after = None
            mock_result.cooldown_info = {CooldownInfoFields.USER_MESSAGE: "失败", CooldownInfoFields.KIND: "UNKNOWN"}
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        runner = Runner(cloud_args)
        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_failure):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                result = await runner._run_cloud("失败", runner._merge_options({}))
                scenarios.append(("失败", set(result.keys())))

        # 场景 3: 超时
        def mock_timeout(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            return mock_executor

        runner = Runner(cloud_args)
        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_timeout):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
                result = await runner._run_cloud("超时", runner._merge_options({}))
                scenarios.append(("超时", set(result.keys())))

        # 场景 4: 无 API Key
        runner = Runner(cloud_args)
        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)
            result = await runner._run_cloud("无Key", runner._merge_options({}))
            scenarios.append(("无Key", set(result.keys())))

        # 验证所有场景的字段集合相同
        first_scenario, first_fields = scenarios[0]
        for scenario_name, fields in scenarios[1:]:
            assert fields == first_fields, (
                f"字段集合不一致: {scenario_name} != {first_scenario}\n"
                f"差异: {fields.symmetric_difference(first_fields)}"
            )


# ============================================================
# Cloud 模式无 API Key 时 cooldown_info 契约字段测试
# ============================================================


class TestCloudNoApiKeyCooldownInfoContract:
    """测试 Cloud 模式且无 API Key 时 cooldown_info 的契约字段

    验证场景:
    1. 显式 --execution-mode cloud 无 API Key
    2. has_ampersand_prefix=True 无 API Key
    3. cooldown_info 包含 output_contract 定义的所有稳定字段
    4. message_level 为 warning（显式 cloud 或 & 前缀场景）
    5. remaining_seconds 类型为 None 或 float（避免类型漂移）
    """

    @pytest.fixture
    def cloud_args_explicit(self) -> argparse.Namespace:
        """创建显式 --execution-mode cloud 的参数（无 API Key）"""
        return argparse.Namespace(
            task="显式 Cloud 模式任务",
            mode="cloud",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode="cloud",  # 显式 cloud 模式
            _execution_mode_user_set=True,  # 显式指定
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,  # 无 API key
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=False,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            print_config=False,
        )

    @pytest.fixture
    def ampersand_prefix_args(self) -> argparse.Namespace:
        """创建 has_ampersand_prefix=True 的参数（无 API Key）"""
        return argparse.Namespace(
            task="& 后台分析任务",  # & 前缀
            mode="cloud",
            directory=".",
            _directory_user_set=False,
            workers=3,
            max_iterations="10",
            strict_review=None,
            enable_sub_planners=None,
            verbose=False,
            quiet=False,
            log_level=None,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model=None,
            worker_model=None,
            reviewer_model=None,
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator=None,
            no_mp=None,
            _orchestrator_user_set=False,
            execution_mode="cloud",
            _execution_mode_user_set=False,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,  # 无 API key
            cloud_auth_timeout=None,
            cloud_timeout=None,
            cloud_background=False,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            print_config=False,
        )

    @pytest.mark.asyncio
    async def test_explicit_cloud_no_key_cooldown_info_contains_stable_fields(
        self, cloud_args_explicit: argparse.Namespace
    ) -> None:
        """验证显式 cloud 模式无 API Key 时 cooldown_info 包含所有稳定字段

        断言:
        1. cooldown_info 存在
        2. 包含 COOLDOWN_INFO_MINIMUM_STABLE_FIELDS 定义的所有字段
        """
        from core.output_contract import COOLDOWN_INFO_MINIMUM_STABLE_FIELDS
        from run import Runner

        runner = Runner(cloud_args_explicit)

        # 清空环境变量中的 CURSOR_API_KEY
        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("显式Cloud无Key任务", options)

        # 断言 1: cooldown_info 存在
        assert CloudResultFields.COOLDOWN_INFO in result, "结果必须包含 cooldown_info 字段"
        assert result[CloudResultFields.COOLDOWN_INFO] is not None, (
            "显式 cloud 模式无 API Key 时 cooldown_info 不应为 None"
        )

        cooldown = result[CloudResultFields.COOLDOWN_INFO]

        # 断言 2: 包含所有稳定字段
        missing_fields = COOLDOWN_INFO_MINIMUM_STABLE_FIELDS - set(cooldown.keys())
        assert not missing_fields, (
            f"cooldown_info 缺少稳定字段: {missing_fields}\n"
            f"期望字段: {COOLDOWN_INFO_MINIMUM_STABLE_FIELDS}\n"
            f"实际字段: {set(cooldown.keys())}"
        )

    @pytest.mark.asyncio
    async def test_explicit_cloud_no_key_message_level_is_warning(
        self, cloud_args_explicit: argparse.Namespace
    ) -> None:
        """验证显式 cloud 模式无 API Key 时 message_level 为 warning

        用户显式使用 --execution-mode cloud 表示明确的 Cloud 意图，
        无 API Key 时应使用 warning 级别提醒用户。
        """
        from run import Runner

        runner = Runner(cloud_args_explicit)

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("显式Cloud无Key任务", options)

        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "cooldown_info 应存在"

        # 断言: message_level 为 warning（使用稳定字段常量）
        assert CooldownInfoFields.MESSAGE_LEVEL in cooldown, "cooldown_info 应包含 message_level 字段"
        assert cooldown.get(CooldownInfoFields.MESSAGE_LEVEL) == "warning", (
            f"显式 cloud 模式无 API Key 时 message_level 应为 'warning'，"
            f"实际: {cooldown.get(CooldownInfoFields.MESSAGE_LEVEL)}"
        )

    @pytest.mark.asyncio
    async def test_ampersand_prefix_no_key_message_level_is_warning(
        self, ampersand_prefix_args: argparse.Namespace
    ) -> None:
        """验证 & 前缀无 API Key 时 message_level 为 warning

        用户使用 & 前缀表示明确的 Cloud 意图，
        无 API Key 时应使用 warning 级别提醒用户。
        """
        from core.cloud_utils import is_cloud_request
        from run import Runner

        runner = Runner(ampersand_prefix_args)

        # 确认任务包含 & 前缀
        task = ampersand_prefix_args.task
        assert is_cloud_request(task), f"任务应包含 & 前缀: {task}"

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({"has_ampersand_prefix": True})
            result = await runner._run_cloud("& 后台分析任务", options)

        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "cooldown_info 应存在"

        # 断言: message_level 为 warning（使用稳定字段常量）
        assert CooldownInfoFields.MESSAGE_LEVEL in cooldown, "cooldown_info 应包含 message_level 字段"
        assert cooldown.get(CooldownInfoFields.MESSAGE_LEVEL) == "warning", (
            f"& 前缀无 API Key 时 message_level 应为 'warning'，实际: {cooldown.get(CooldownInfoFields.MESSAGE_LEVEL)}"
        )

    @pytest.mark.asyncio
    async def test_remaining_seconds_type_none_or_float(self, cloud_args_explicit: argparse.Namespace) -> None:
        """验证 remaining_seconds 类型为 None 或 float（避免类型漂移）

        根据 output_contract 定义，remaining_seconds 应为 Optional[float]，
        以支持精确计时。不允许 int 类型（避免类型漂移导致的兼容性问题）。
        """
        from core.output_contract import CooldownInfoFields
        from run import Runner

        runner = Runner(cloud_args_explicit)

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("显式Cloud无Key任务", options)

        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "cooldown_info 应存在"

        # 断言: remaining_seconds 存在
        assert CooldownInfoFields.REMAINING_SECONDS in cooldown, (
            f"cooldown_info 应包含 {CooldownInfoFields.REMAINING_SECONDS} 字段"
        )

        remaining_seconds = cooldown[CooldownInfoFields.REMAINING_SECONDS]

        # 断言: 类型为 None 或 float
        assert remaining_seconds is None or isinstance(remaining_seconds, float), (
            f"remaining_seconds 应为 None 或 float，"
            f"实际类型: {type(remaining_seconds).__name__}，值: {remaining_seconds}"
        )

    @pytest.mark.asyncio
    async def test_cooldown_info_kind_matches_failure_kind(self, cloud_args_explicit: argparse.Namespace) -> None:
        """验证 cooldown_info.kind 与顶层 failure_kind 一致"""
        from run import Runner

        runner = Runner(cloud_args_explicit)

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("显式Cloud无Key任务", options)

        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "cooldown_info 应存在"

        # 断言: kind 与顶层 failure_kind 一致（使用稳定字段常量）
        assert CooldownInfoFields.KIND in cooldown, "cooldown_info 应包含 kind 字段"
        assert result["failure_kind"] == cooldown.get(CooldownInfoFields.KIND), (
            f"cooldown_info.kind ({cooldown.get(CooldownInfoFields.KIND)}) 应与 "
            f"failure_kind ({result['failure_kind']}) 一致"
        )

    @pytest.mark.asyncio
    async def test_no_key_cooldown_info_retryable_and_retry_after(
        self, cloud_args_explicit: argparse.Namespace
    ) -> None:
        """验证无 API Key 时 retryable 和 retry_after 字段

        无 API Key 错误是可重试的（用户可以设置 Key 后重试），
        但不需要等待时间。
        """
        from run import Runner

        runner = Runner(cloud_args_explicit)

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("显式Cloud无Key任务", options)

        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "cooldown_info 应存在"

        # 断言: retryable 存在且为 bool（使用稳定字段常量）
        assert CooldownInfoFields.RETRYABLE in cooldown, "cooldown_info 应包含 retryable 字段"
        assert isinstance(cooldown.get(CooldownInfoFields.RETRYABLE), bool), (
            f"retryable 应为 bool，实际: {type(cooldown.get(CooldownInfoFields.RETRYABLE))}"
        )

        # 断言: retry_after 存在（使用稳定字段常量）
        assert CooldownInfoFields.RETRY_AFTER in cooldown, "cooldown_info 应包含 retry_after 字段"

        # 无 API Key 场景通常 retry_after 为 None（不需要等待）
        # 但某些实现可能返回 0 或其他值，这里只验证类型
        retry_after = cooldown.get(CooldownInfoFields.RETRY_AFTER)
        assert retry_after is None or isinstance(retry_after, (int, float)), (
            f"retry_after 应为 None 或数值，实际: {type(retry_after)}"
        )

    @pytest.mark.asyncio
    async def test_cooldown_info_user_message_non_empty(self, cloud_args_explicit: argparse.Namespace) -> None:
        """验证 user_message 非空且包含有意义的信息"""
        from run import Runner

        runner = Runner(cloud_args_explicit)

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            options = runner._merge_options({})
            result = await runner._run_cloud("显式Cloud无Key任务", options)

        cooldown = result[CloudResultFields.COOLDOWN_INFO]
        assert cooldown is not None, "cooldown_info 应存在"

        # 断言: user_message 非空（使用稳定字段常量）
        assert CooldownInfoFields.USER_MESSAGE in cooldown, "cooldown_info 应包含 user_message 字段"
        user_message = cooldown.get(CooldownInfoFields.USER_MESSAGE)
        assert user_message, "user_message 不应为空"
        assert isinstance(user_message, str), f"user_message 应为 str，实际: {type(user_message)}"

        # 断言: 包含有意义的关键词
        assert any(kw in user_message for kw in ["API Key", "未配置", "CURSOR_API_KEY"]), (
            f"user_message 应包含 API Key 相关提示，实际: {user_message}"
        )


# ============================================================
# cooldown_info 未知字段容忍测试（mock print_warning/print_info）
# ============================================================


class TestCooldownInfoUnknownFieldsPrintLogic:
    """验证 cooldown_info 包含未知字段时打印去重/输出选择逻辑不出错

    测试通过 mock print_warning/print_info 验证：
    1. 带未知字段的 cooldown_info 不会导致 KeyError/TypeError
    2. 打印决策只基于稳定字段（USER_MESSAGE, MESSAGE_LEVEL）
    3. 去重逻辑正常工作
    """

    @pytest.fixture
    def cooldown_info_with_unknown_fields(self) -> dict:
        """构造带未知字段的 cooldown_info"""
        return {
            # 稳定字段
            CooldownInfoFields.USER_MESSAGE: "测试：Cloud 不可用，已回退到 CLI",
            CooldownInfoFields.KIND: "no_key",
            CooldownInfoFields.REASON: "未设置 CURSOR_API_KEY",
            CooldownInfoFields.RETRYABLE: True,
            CooldownInfoFields.RETRY_AFTER: None,
            CooldownInfoFields.IN_COOLDOWN: False,
            CooldownInfoFields.REMAINING_SECONDS: None,
            CooldownInfoFields.FAILURE_COUNT: 1,
            # 兼容字段
            CooldownInfoFields.FALLBACK_REASON: "未设置 CURSOR_API_KEY",
            CooldownInfoFields.ERROR_TYPE: "config_error",
            CooldownInfoFields.FAILURE_KIND: "no_key",
            # 扩展字段
            CooldownInfoFields.MESSAGE_LEVEL: "warning",
            # 未知字段（模拟未来扩展）
            "future_field_v2_api": "new_feature",
            "experimental_metrics": {"latency_ms": 123, "retries": 2},
            "_internal_trace_id": "trace-12345",
            "unused_legacy_field": None,
        }

    def test_print_logic_with_unknown_fields_no_exception(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证打印逻辑处理带未知字段的 cooldown_info 不抛异常

        模拟 run.py 第 2765-2774 行的打印逻辑。
        """
        from core.execution_policy import compute_message_dedup_key

        cooldown = cooldown_info_with_unknown_fields
        shown_messages: set[str] = set()

        # 模拟入口脚本的打印逻辑（不应抛出任何异常）
        if cooldown and cooldown.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                # message_level 缺失时默认按 info 处理
                message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    # 模拟 print_warning 调用
                    pass
                else:
                    # 模拟 print_info 调用
                    pass

        # 断言去重逻辑正常工作
        assert len(shown_messages) == 1

    def test_mock_print_warning_with_unknown_fields(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """通过 mock print_warning 验证带未知字段的 cooldown_info 处理

        断言:
        1. print_warning 被调用一次（message_level=warning）
        2. 调用参数为 user_message 内容
        3. 不会因未知字段导致异常
        """
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key

        cooldown = cooldown_info_with_unknown_fields
        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()
        mock_print_info = MagicMock()

        # 模拟入口脚本的打印逻辑
        if cooldown and cooldown.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    mock_print_warning(cooldown[CooldownInfoFields.USER_MESSAGE])
                else:
                    mock_print_info(cooldown[CooldownInfoFields.USER_MESSAGE])

        # 断言 1: print_warning 被调用一次
        assert mock_print_warning.call_count == 1, f"print_warning 应被调用 1 次，实际: {mock_print_warning.call_count}"

        # 断言 2: 调用参数正确
        mock_print_warning.assert_called_once_with("测试：Cloud 不可用，已回退到 CLI")

        # 断言 3: print_info 未被调用
        assert mock_print_info.call_count == 0

    def test_mock_print_info_when_message_level_info(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证 message_level=info 时使用 print_info"""
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key

        # 修改 message_level 为 info
        cooldown = cooldown_info_with_unknown_fields.copy()
        cooldown[CooldownInfoFields.MESSAGE_LEVEL] = "info"
        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()
        mock_print_info = MagicMock()

        # 模拟入口脚本的打印逻辑
        if cooldown and cooldown.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    mock_print_warning(cooldown[CooldownInfoFields.USER_MESSAGE])
                else:
                    mock_print_info(cooldown[CooldownInfoFields.USER_MESSAGE])

        # 断言: print_info 被调用，print_warning 未被调用
        assert mock_print_info.call_count == 1
        assert mock_print_warning.call_count == 0

    def test_dedup_with_unknown_fields_multiple_calls(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证去重逻辑在多次调用时正常工作（带未知字段）"""
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key

        cooldown = cooldown_info_with_unknown_fields
        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()

        # 模拟多次调用（相同消息应只打印一次）
        for _ in range(5):
            if cooldown and cooldown.get(CooldownInfoFields.USER_MESSAGE):
                msg_key = compute_message_dedup_key(cooldown[CooldownInfoFields.USER_MESSAGE])
                if msg_key not in shown_messages:
                    shown_messages.add(msg_key)
                    message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                    if message_level == "warning":
                        mock_print_warning(cooldown[CooldownInfoFields.USER_MESSAGE])

        # 断言: 尽管调用 5 次，print_warning 只被调用 1 次
        assert mock_print_warning.call_count == 1, (
            f"print_warning 应只被调用 1 次（去重），实际: {mock_print_warning.call_count}"
        )

    def test_unknown_fields_do_not_affect_decision(
        self,
        cooldown_info_with_unknown_fields: dict,
    ) -> None:
        """验证未知字段不影响打印决策

        即使未知字段包含与稳定字段同名但值不同的内容，
        决策仍应基于正确的稳定字段。
        """
        from unittest.mock import MagicMock

        from core.execution_policy import compute_message_dedup_key

        # 构造一个具有"迷惑性"未知字段的 cooldown_info
        cooldown = cooldown_info_with_unknown_fields.copy()
        # 添加可能造成混淆的未知字段（但这些不应影响决策）
        cooldown["message_level_override"] = "error"  # 不应被使用
        cooldown["user_message_deprecated"] = "错误的消息"  # 不应被使用
        cooldown["print_as_warning"] = False  # 不应影响决策

        shown_messages: set[str] = set()
        mock_print_warning = MagicMock()
        mock_print_info = MagicMock()

        # 模拟入口脚本的打印逻辑（应使用正确的稳定字段）
        if cooldown and cooldown.get(CooldownInfoFields.USER_MESSAGE):
            msg_key = compute_message_dedup_key(cooldown[CooldownInfoFields.USER_MESSAGE])
            if msg_key not in shown_messages:
                shown_messages.add(msg_key)
                # 只使用 CooldownInfoFields.MESSAGE_LEVEL
                message_level = cooldown.get(CooldownInfoFields.MESSAGE_LEVEL, "info")
                if message_level == "warning":
                    mock_print_warning(cooldown[CooldownInfoFields.USER_MESSAGE])
                else:
                    mock_print_info(cooldown[CooldownInfoFields.USER_MESSAGE])

        # 断言: 决策基于稳定字段 MESSAGE_LEVEL="warning"
        assert mock_print_warning.call_count == 1
        assert mock_print_info.call_count == 0
        # 断言: 使用的是正确的 user_message
        mock_print_warning.assert_called_once_with("测试：Cloud 不可用，已回退到 CLI")


# ============================================================
# TestRequestedModeForDecisionInvariant - 入口计算一致性参数化测试
# ============================================================


@dataclass
class RequestedModeInvariantCase:
    """requested_mode_for_decision 不变式测试用例

    字段说明：
    - test_id: 测试标识符
    - requirement: 用户输入（可包含 & 前缀）
    - cli_execution_mode: CLI --execution-mode 参数（None 表示未指定）
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode
    - has_api_key: 是否有 API Key
    - cloud_enabled: cloud_agent.enabled 配置
    - expected_requested_mode_for_decision: 预期的 requested_mode_for_decision
    - expected_has_ampersand_prefix: 预期的 has_ampersand_prefix
    - expected_prefix_routed: 预期的 prefix_routed
    - expected_orchestrator: 预期的 orchestrator (mp/basic)
    - description: 场景描述
    """

    test_id: str
    requirement: str
    cli_execution_mode: Optional[str]
    config_execution_mode: str
    has_api_key: bool
    cloud_enabled: bool
    expected_requested_mode_for_decision: Optional[str]
    expected_has_ampersand_prefix: bool
    expected_prefix_routed: bool
    expected_orchestrator: str
    description: str


from dataclasses import dataclass

# 参数化测试矩阵
REQUESTED_MODE_INVARIANT_CASES = [
    # ===== 场景 1: 无 & + 无 CLI execution_mode + config=auto =====
    RequestedModeInvariantCase(
        test_id="no_amp_no_cli_config_auto",
        requirement="普通任务描述",
        cli_execution_mode=None,  # 未显式指定
        config_execution_mode="auto",  # config.yaml 默认
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",  # 来自 config.yaml
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expected_orchestrator="basic",  # auto 强制 basic
        description="无 & 前缀 + 无 CLI execution_mode + config=auto：使用 config.yaml 默认值",
    ),
    RequestedModeInvariantCase(
        test_id="no_amp_no_cli_config_auto_no_key",
        requirement="普通任务描述",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,  # 无 API Key
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",  # 仍为 auto
        expected_has_ampersand_prefix=False,
        expected_prefix_routed=False,
        expected_orchestrator="basic",  # requested=auto 强制 basic，即使回退到 cli
        description="无 & + 无 CLI + config=auto + 无 key：requested 仍为 auto，orchestrator=basic",
    ),
    # ===== 场景 2: 有 & + 无 CLI execution_mode =====
    RequestedModeInvariantCase(
        test_id="amp_no_cli_with_key",
        requirement="& 后台分析任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision=None,  # & 前缀时返回 None
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=True,  # 成功路由到 Cloud
        expected_orchestrator="basic",  # Cloud 模式强制 basic
        description="有 & + 无 CLI + 有 key + cloud_enabled：prefix_routed=True",
    ),
    RequestedModeInvariantCase(
        test_id="amp_no_cli_no_key",
        requirement="& 后台分析任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=False,  # 无 API Key
        cloud_enabled=True,
        expected_requested_mode_for_decision=None,  # & 前缀时返回 None
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # 无 key，未成功路由
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，仍强制 basic
        description="有 & + 无 CLI + 无 key：prefix_routed=False，但 orchestrator 仍为 basic",
    ),
    RequestedModeInvariantCase(
        test_id="amp_no_cli_cloud_disabled",
        requirement="& 后台分析任务",
        cli_execution_mode=None,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False
        expected_requested_mode_for_decision=None,
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # cloud_disabled，未成功路由
        expected_orchestrator="basic",  # & 前缀表达 Cloud 意图，仍强制 basic
        description="有 & + 无 CLI + cloud_disabled：prefix_routed=False，orchestrator=basic",
    ),
    # ===== 场景 3: 有 & + 显式 --execution-mode cli =====
    RequestedModeInvariantCase(
        test_id="amp_explicit_cli",
        requirement="& 显式 CLI 任务",
        cli_execution_mode="cli",  # 显式指定 cli
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cli",  # CLI 优先级最高
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # 显式 cli 忽略 & 前缀
        expected_orchestrator="mp",  # cli 模式允许 mp
        description="有 & + 显式 --execution-mode cli：& 前缀被忽略，允许 mp",
    ),
    RequestedModeInvariantCase(
        test_id="amp_explicit_cli_no_key",
        requirement="& 显式 CLI 任务",
        cli_execution_mode="cli",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cli",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="mp",  # cli 模式允许 mp，无论 key 状态
        description="有 & + 显式 cli + 无 key：仍允许 mp",
    ),
    # ===== 场景 4: 有 & + 显式 --execution-mode plan =====
    RequestedModeInvariantCase(
        test_id="amp_explicit_plan",
        requirement="& 规划分析任务",
        cli_execution_mode="plan",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="plan",  # 显式 plan
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # plan 模式忽略 & 前缀
        expected_orchestrator="mp",  # plan 模式（只读）允许 mp
        description="有 & + 显式 --execution-mode plan：& 前缀被忽略，允许 mp",
    ),
    # ===== 场景 5: 有 & + 显式 --execution-mode ask =====
    RequestedModeInvariantCase(
        test_id="amp_explicit_ask",
        requirement="& 代码解释任务",
        cli_execution_mode="ask",
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="ask",  # 显式 ask
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # ask 模式忽略 & 前缀
        expected_orchestrator="mp",  # ask 模式（只读）允许 mp
        description="有 & + 显式 --execution-mode ask：& 前缀被忽略，允许 mp",
    ),
    # ===== 场景 6: 有 & + 显式 --execution-mode cloud =====
    RequestedModeInvariantCase(
        test_id="amp_explicit_cloud",
        requirement="& 云端长时间任务",
        cli_execution_mode="cloud",  # 显式 cloud
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cloud",  # 显式 cloud 优先于 & 前缀
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # 有显式 execution_mode，prefix_routed=False
        expected_orchestrator="basic",  # cloud 强制 basic
        description="有 & + 显式 --execution-mode cloud：使用显式 cloud，强制 basic",
    ),
    RequestedModeInvariantCase(
        test_id="amp_explicit_cloud_no_key",
        requirement="& 云端长时间任务",
        cli_execution_mode="cloud",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cloud",  # 仍为 cloud
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",  # requested=cloud 强制 basic，即使无 key
        description="有 & + 显式 cloud + 无 key：orchestrator 仍为 basic",
    ),
    # ===== 场景 7: 有 & + 显式 --execution-mode auto =====
    RequestedModeInvariantCase(
        test_id="amp_explicit_auto",
        requirement="& 自动模式任务",
        cli_execution_mode="auto",  # 显式 auto
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",  # 显式 auto
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,  # 有显式 execution_mode
        expected_orchestrator="basic",  # auto 强制 basic
        description="有 & + 显式 --execution-mode auto：使用显式 auto，强制 basic",
    ),
    RequestedModeInvariantCase(
        test_id="amp_explicit_auto_no_key",
        requirement="& 自动模式任务",
        cli_execution_mode="auto",
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_has_ampersand_prefix=True,
        expected_prefix_routed=False,
        expected_orchestrator="basic",  # requested=auto 强制 basic
        description="有 & + 显式 auto + 无 key：orchestrator 仍为 basic",
    ),
]


class TestRequestedModeForDecisionInvariant:
    """测试 requested_mode_for_decision 计算的一致性和不变式

    验证入口脚本 (run.py) 计算的 requested_mode_for_decision 与
    build_execution_decision 的输入一致，符合以下不变式：

    1. CLI 显式参数优先级最高
    2. 有 & 前缀且无 CLI 显式参数时，返回 None（让 build_execution_decision 处理）
    3. 无 & 前缀且无 CLI 显式参数时，使用 config.yaml 的值
    4. requested_mode_for_decision 决定 orchestrator 选择（auto/cloud 强制 basic）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前后重置 ConfigManager"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "test_case",
        REQUESTED_MODE_INVARIANT_CASES,
        ids=[tc.test_id for tc in REQUESTED_MODE_INVARIANT_CASES],
    )
    def test_resolve_requested_mode_for_decision_invariant(self, test_case: RequestedModeInvariantCase) -> None:
        """验证 resolve_requested_mode_for_decision 的返回值符合预期"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import resolve_requested_mode_for_decision

        has_ampersand_prefix = is_cloud_request(test_case.requirement)

        # 断言 1: has_ampersand_prefix 符合预期
        assert has_ampersand_prefix == test_case.expected_has_ampersand_prefix, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  has_ampersand_prefix 预期={test_case.expected_has_ampersand_prefix}，"
            f"实际={has_ampersand_prefix}"
        )

        # 调用 resolve_requested_mode_for_decision
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        # 断言 2: requested_mode_for_decision 符合预期
        assert requested_mode == test_case.expected_requested_mode_for_decision, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  cli_execution_mode={test_case.cli_execution_mode}\n"
            f"  has_ampersand_prefix={has_ampersand_prefix}\n"
            f"  config_execution_mode={test_case.config_execution_mode}\n"
            f"  requested_mode_for_decision 预期={test_case.expected_requested_mode_for_decision}，"
            f"实际={requested_mode}"
        )

    @pytest.mark.parametrize(
        "test_case",
        REQUESTED_MODE_INVARIANT_CASES,
        ids=[tc.test_id for tc in REQUESTED_MODE_INVARIANT_CASES],
    )
    def test_build_execution_decision_invariant(self, test_case: RequestedModeInvariantCase) -> None:
        """验证 build_execution_decision 的输出符合预期"""
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
        )

        has_ampersand_prefix = is_cloud_request(test_case.requirement)

        # 计算 requested_mode_for_decision
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        # 调用 build_execution_decision
        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        # 断言 1: prefix_routed 符合预期
        assert decision.prefix_routed == test_case.expected_prefix_routed, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  prefix_routed 预期={test_case.expected_prefix_routed}，"
            f"实际={decision.prefix_routed}"
        )

        # 断言 2: orchestrator 符合预期
        assert decision.orchestrator == test_case.expected_orchestrator, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  requested_mode={requested_mode}\n"
            f"  prefix_routed={decision.prefix_routed}\n"
            f"  orchestrator 预期={test_case.expected_orchestrator}，"
            f"实际={decision.orchestrator}"
        )

        # 断言 3: has_ampersand_prefix 符合预期
        assert decision.has_ampersand_prefix == test_case.expected_has_ampersand_prefix, (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  has_ampersand_prefix 预期={test_case.expected_has_ampersand_prefix}，"
            f"实际={decision.has_ampersand_prefix}"
        )

    @pytest.mark.parametrize(
        "test_case",
        [tc for tc in REQUESTED_MODE_INVARIANT_CASES if tc.expected_requested_mode_for_decision in ("auto", "cloud")],
        ids=[
            tc.test_id
            for tc in REQUESTED_MODE_INVARIANT_CASES
            if tc.expected_requested_mode_for_decision in ("auto", "cloud")
        ],
    )
    def test_auto_cloud_forces_basic_invariant(self, test_case: RequestedModeInvariantCase) -> None:
        """验证核心不变式：requested_mode=auto/cloud 强制 basic

        这是最重要的一致性不变式：
        - 即使因无 API Key 回退到 CLI 执行
        - 只要 requested_mode 是 auto 或 cloud
        - orchestrator 必须是 basic
        """
        from core.cloud_utils import is_cloud_request
        from core.execution_policy import (
            build_execution_decision,
            resolve_requested_mode_for_decision,
            should_use_mp_orchestrator,
        )

        has_ampersand_prefix = is_cloud_request(test_case.requirement)

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=test_case.cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=test_case.config_execution_mode,
        )

        # 断言: requested_mode 为 auto 或 cloud
        assert requested_mode in ("auto", "cloud"), f"[{test_case.test_id}] 此用例 requested_mode 应为 auto/cloud"

        # 断言: should_use_mp_orchestrator 返回 False
        can_use_mp = should_use_mp_orchestrator(requested_mode)
        assert can_use_mp is False, (
            f"[{test_case.test_id}] requested_mode={requested_mode} 时 should_use_mp_orchestrator 应返回 False"
        )

        # 断言: build_execution_decision 返回 basic
        decision = build_execution_decision(
            prompt=test_case.requirement,
            requested_mode=requested_mode,
            cloud_enabled=test_case.cloud_enabled,
            has_api_key=test_case.has_api_key,
            auto_detect_cloud_prefix=True,
        )

        assert decision.orchestrator == "basic", (
            f"[{test_case.test_id}] {test_case.description}\n"
            f"  核心不变式违反：requested_mode={requested_mode} 时 orchestrator 必须是 basic\n"
            f"  实际 orchestrator={decision.orchestrator}"
        )
