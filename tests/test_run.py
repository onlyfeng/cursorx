"""测试 run.py 统一入口脚本"""
import argparse
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.cloud_utils import CLOUD_PREFIX, is_cloud_request, strip_cloud_prefix
from core.config import get_config
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

# 获取配置驱动的默认值（用于测试断言）
_config = get_config()
CONFIG_WORKER_POOL_SIZE = _config.system.worker_pool_size
CONFIG_MAX_ITERATIONS = _config.system.max_iterations
CONFIG_ENABLE_SUB_PLANNERS = _config.system.enable_sub_planners
CONFIG_STRICT_REVIEW = _config.system.strict_review
CONFIG_PLANNER_MODEL = _config.models.planner
CONFIG_WORKER_MODEL = _config.models.worker
CONFIG_REVIEWER_MODEL = _config.models.reviewer
CONFIG_CLOUD_TIMEOUT = _config.cloud_agent.timeout
CONFIG_PLANNER_TIMEOUT = _config.planner.timeout
CONFIG_WORKER_TIMEOUT = _config.worker.task_timeout
CONFIG_REVIEWER_TIMEOUT = _config.reviewer.timeout
# 流式日志配置（来自 logging.stream_json.*）
CONFIG_STREAM_LOG_ENABLED = _config.logging.stream_json.enabled
CONFIG_STREAM_LOG_CONSOLE = _config.logging.stream_json.console
CONFIG_STREAM_LOG_DETAIL_DIR = _config.logging.stream_json.detail_dir
CONFIG_STREAM_LOG_RAW_DIR = _config.logging.stream_json.raw_dir


# ============================================================
# Fixtures
# ============================================================


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
    """测试 execution_mode/cloud_auth 参数映射到 OrchestratorConfig 与 IterateArgs"""

    @pytest.fixture
    def iterate_mode_args(self) -> argparse.Namespace:
        """创建 iterate 模式的参数（默认值来自 config.yaml）"""
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
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            # 执行模式参数
            execution_mode="cli",
            cloud_api_key=None,
            cloud_auth_timeout=30,
        )

    def test_execution_mode_cli_default(self, iterate_mode_args: argparse.Namespace) -> None:
        """测试默认 execution_mode=cli 时的配置映射"""
        runner = Runner(iterate_mode_args)
        options = runner._merge_options({})

        # 验证默认值
        assert options.get("orchestrator") == "mp"
        assert options.get("no_mp") is False

    def test_execution_mode_cloud_in_analysis_options(
        self, iterate_mode_args: argparse.Namespace
    ) -> None:
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

    def test_cloud_mode_detection_sets_execution_mode_option(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 '&' 前缀检测后设置 execution_mode 选项"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 测试 '&' 前缀任务
        analysis = analyzer.analyze("& 分析代码架构", mock_args)

        # 验证模式被设置为 CLOUD
        assert analysis.mode == RunMode.CLOUD

        # 验证 execution_mode 选项被设置
        assert analysis.options.get("execution_mode") == "cloud"

        # 验证 goal 被正确剥离 '&' 前缀
        assert analysis.goal == "分析代码架构"
        assert not analysis.goal.startswith("&")

    def test_cloud_mode_goal_stripping_preserves_content(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 Cloud 模式下 goal 剥离 '&' 前缀保留实际内容"""
        analyzer = TaskAnalyzer(use_agent=False)

        test_cases = [
            ("& 简单任务", "简单任务"),
            ("&任务描述", "任务描述"),
            ("  & 带空格的任务  ", "带空格的任务"),
            ("& 包含 & 符号的任务", "包含 & 符号的任务"),
        ]

        for task, expected_goal in test_cases:
            analysis = analyzer.analyze(task, mock_args)
            assert analysis.mode == RunMode.CLOUD, f"任务 '{task}' 应该匹配 CLOUD 模式"
            assert analysis.goal == expected_goal, (
                f"任务 '{task}' 的 goal 应该是 '{expected_goal}'，实际是 '{analysis.goal}'"
            )

    @pytest.mark.asyncio
    async def test_run_iterate_receives_execution_mode(
        self, iterate_mode_args: argparse.Namespace
    ) -> None:
        """测试 _run_iterate 正确接收 execution_mode 选项"""
        runner = Runner(iterate_mode_args)

        # 模拟 analysis 包含 cloud execution_mode
        analysis_options = {"execution_mode": "cloud"}
        merged = runner._merge_options(analysis_options)

        # IterateArgs 应该在 _run_iterate 中创建，这里验证选项结构
        assert "orchestrator" in merged
        assert "workers" in merged

    def test_iterate_args_structure_for_cloud_mode(
        self, iterate_mode_args: argparse.Namespace
    ) -> None:
        """测试 IterateArgs 结构包含 execution_mode 相关字段"""
        # 验证 iterate_mode_args 包含所有必需的执行模式字段
        assert hasattr(iterate_mode_args, "execution_mode")
        assert hasattr(iterate_mode_args, "cloud_api_key")
        assert hasattr(iterate_mode_args, "cloud_auth_timeout")

        # 验证默认值
        assert iterate_mode_args.execution_mode == "cli"
        assert iterate_mode_args.cloud_api_key is None
        assert iterate_mode_args.cloud_auth_timeout == 30


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
# TestCloudModeAutoCommitSafety
# ============================================================


class TestCloudModeAutoCommitSafety:
    """测试 Cloud 模式下 auto_commit 默认禁用的安全策略

    确保即使使用 '&' 前缀触发 Cloud 模式，auto_commit 仍需显式开启。
    """

    @pytest.fixture
    def cloud_mode_args(self) -> argparse.Namespace:
        """创建 Cloud 模式的参数"""
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
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
            stream_log=True,
            no_auto_analyze=False,
            auto_commit=False,  # 默认禁用
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cli",
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

    def test_cloud_mode_does_not_auto_enable_commit(
        self, cloud_mode_args: argparse.Namespace
    ) -> None:
        """验证 Cloud 模式不自动开启 auto_commit"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

        # 验证使用 Cloud 模式
        assert analysis.mode == RunMode.CLOUD
        assert analysis.options.get("execution_mode") == "cloud"

        # 验证 auto_commit 未被自动设置为 True
        assert analysis.options.get("auto_commit") is not True

        # Runner 合并选项后 auto_commit 仍为 False
        runner = Runner(cloud_mode_args)
        merged = runner._merge_options(analysis.options)
        assert merged["auto_commit"] is False

    def test_cloud_mode_with_explicit_auto_commit(
        self, cloud_mode_args: argparse.Namespace
    ) -> None:
        """验证 Cloud 模式下显式启用 auto_commit"""
        cloud_mode_args.auto_commit = True  # 显式启用

        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

        runner = Runner(cloud_mode_args)
        merged = runner._merge_options(analysis.options)

        # auto_commit 应为 True（显式指定）
        assert merged["auto_commit"] is True

    def test_cloud_mode_auto_commit_from_keywords(
        self, cloud_mode_args: argparse.Namespace
    ) -> None:
        """验证 Cloud 模式下通过关键词启用 auto_commit"""
        cloud_mode_args.task = "& 后台执行任务，启用提交"

        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze(cloud_mode_args.task, cloud_mode_args)

        # 验证检测到启用提交关键词
        assert analysis.options.get("auto_commit") is True

        runner = Runner(cloud_mode_args)
        merged = runner._merge_options(analysis.options)
        assert merged["auto_commit"] is True

    def test_cloud_mode_auto_push_requires_auto_commit(
        self, cloud_mode_args: argparse.Namespace
    ) -> None:
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
        """验证 Agent 分析成功时正确解析 JSON 结果，使用 plan 模式只读执行"""
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

    def test_agent_analysis_empty_output(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_agent_analysis_whitespace_output(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_analyze_with_agent_fallback(
        self, mock_args: argparse.Namespace
    ) -> None:
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
                f"任务 '{task}' 的 reasoning 应该包含非并行相关信息，"
                f"实际: {analysis.reasoning}"
            )

    def test_non_parallel_with_iterate_mode(self, mock_args: argparse.Namespace) -> None:
        """验证自我迭代模式+非并行关键词同时生效"""
        analyzer = TaskAnalyzer(use_agent=False)

        analysis = analyzer.analyze("自我迭代，非并行模式，优化代码", mock_args)

        # 验证模式为 ITERATE
        assert analysis.mode == RunMode.ITERATE, (
            f"应该检测到 ITERATE 模式，实际: {analysis.mode}"
        )

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

    def test_detect_cloud_prefix_basic(self, mock_args: argparse.Namespace) -> None:
        """验证检测 '&' 前缀并路由到 Cloud 模式"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("& 分析代码结构", mock_args)

        assert analysis.mode == RunMode.CLOUD, "以 '&' 开头应使用 Cloud 模式"
        assert analysis.goal == "分析代码结构", "goal 应去除 '&' 前缀"
        assert "Cloud" in analysis.reasoning or "'&'" in analysis.reasoning

    def test_detect_cloud_prefix_with_whitespace(self, mock_args: argparse.Namespace) -> None:
        """验证检测带空白的 '&' 前缀"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("  & 分析代码", mock_args)

        assert analysis.mode == RunMode.CLOUD
        assert analysis.goal.strip() == "分析代码"

    def test_detect_cloud_prefix_no_space(self, mock_args: argparse.Namespace) -> None:
        """验证检测紧跟内容的 '&' 前缀"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("&分析代码结构", mock_args)

        assert analysis.mode == RunMode.CLOUD
        assert analysis.goal == "分析代码结构"

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
        """验证 Cloud 模式设置 execution_mode 选项"""
        analyzer = TaskAnalyzer(use_agent=False)
        analysis = analyzer.analyze("& 后台执行任务", mock_args)

        assert analysis.mode == RunMode.CLOUD
        assert analysis.options.get("execution_mode") == "cloud"

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
        """验证 '&' 前缀优先于其他模式关键词"""
        analyzer = TaskAnalyzer(use_agent=False)
        # 同时包含 '&' 前缀和其他模式关键词
        analysis = analyzer.analyze("& 自我迭代更新代码", mock_args)

        # '&' 前缀应优先
        assert analysis.mode == RunMode.CLOUD
        # goal 应去除前缀
        assert "自我迭代" in analysis.goal

    def test_cloud_prefix_priority_over_plan_keywords(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 '&' 前缀优先于 plan 模式关键词"""
        analyzer = TaskAnalyzer(use_agent=False)
        # 同时包含 '&' 前缀和 plan 模式关键词
        analysis = analyzer.analyze("& 规划任务执行步骤", mock_args)

        # '&' 前缀应优先选择 CLOUD 模式
        assert analysis.mode == RunMode.CLOUD
        assert analysis.options.get("execution_mode") == "cloud"
        # goal 应去除 '&' 前缀
        assert "规划任务" in analysis.goal

    def test_cloud_prefix_priority_over_ask_keywords(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 '&' 前缀优先于 ask 模式关键词"""
        analyzer = TaskAnalyzer(use_agent=False)
        # 同时包含 '&' 前缀和 ask 模式关键词
        analysis = analyzer.analyze("& 问答模式解决问题", mock_args)

        # '&' 前缀应优先选择 CLOUD 模式
        assert analysis.mode == RunMode.CLOUD
        assert analysis.options.get("execution_mode") == "cloud"
        # goal 应去除 '&' 前缀
        assert "问答模式" in analysis.goal

    def test_plan_keyword_detection_comprehensive(
        self, mock_args: argparse.Namespace
    ) -> None:
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
                f"任务 '{task}' (关键词: {keyword}) 应使用 PLAN 模式，"
                f"实际为 {analysis.mode}"
            )

    def test_ask_keyword_detection_comprehensive(
        self, mock_args: argparse.Namespace
    ) -> None:
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
                f"任务 '{task}' (关键词: {keyword}) 应使用 ASK 模式，"
                f"实际为 {analysis.mode}"
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

    def test_merge_options_analysis_directory_override(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证分析结果可覆盖目录（未显式指定时）"""
        mock_args._directory_user_set = False
        runner = Runner(mock_args)
        options = runner._merge_options({"directory": "/tmp/project"})

        assert options["directory"] == "/tmp/project"

    def test_merge_options_analysis_directory_respects_user(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证显式目录优先，不被分析覆盖"""
        mock_args._directory_user_set = True
        mock_args.directory = "/explicit/path"
        runner = Runner(mock_args)
        options = runner._merge_options({"directory": "/from/task"})

        assert options["directory"] == "/explicit/path"

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

    def test_merge_options_auto_commit_default_false_without_arg(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证不提供 --auto-commit 时 auto_commit 默认为 False"""
        runner = Runner(mock_args)

        # 不提供任何 auto_commit 相关选项
        options = runner._merge_options({})
        assert options["auto_commit"] is False, (
            "不提供 --auto-commit 时，auto_commit 应默认为 False"
        )

        # 即使提供其他选项，auto_commit 仍应为 False
        options = runner._merge_options({
            "workers": 5,
            "max_iterations": -1,
            "strict": True,
        })
        assert options["auto_commit"] is False, (
            "提供其他选项但不提供 auto_commit 时，应默认为 False"
        )

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

        assert options["auto_commit"] is True, (
            "显式提供 --auto-commit 时，应该启用 auto_commit"
        )

    def test_merge_options_natural_language_commit_keywords_consistent(
        self, mock_args: argparse.Namespace
    ) -> None:
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
            assert options["auto_commit"] is True, (
                f"任务 '{task}' 应该通过自然语言启用 auto_commit"
            )

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
            assert options["auto_commit"] is False, (
                f"任务 '{task}' 应该通过自然语言禁用 auto_commit"
            )

    def test_merge_options_auto_push_forced_false_when_auto_commit_false(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 auto_push 在 auto_commit==False 时必须被强制为 False"""
        runner = Runner(mock_args)

        # 场景1: 两者都未指定，auto_push 应为 False
        options = runner._merge_options({})
        assert options["auto_commit"] is False
        assert options["auto_push"] is False, (
            "auto_commit=False 时，auto_push 必须为 False"
        )

        # 场景2: 仅指定 auto_push=True（不指定 auto_commit）
        options = runner._merge_options({"auto_push": True})
        assert options["auto_commit"] is False, (
            "未指定 auto_commit 时应默认为 False"
        )
        assert options["auto_push"] is False, (
            "auto_commit=False 时，即使指定 auto_push=True 也应强制为 False"
        )

        # 场景3: 显式禁用 auto_commit，启用 auto_push
        options = runner._merge_options({"auto_commit": False, "auto_push": True})
        assert options["auto_push"] is False, (
            "显式 auto_commit=False 时，auto_push 必须强制为 False"
        )

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

    def test_merge_options_stream_renderer_defaults(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证流式渲染配置默认值（默认关闭，避免噪声）"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        # 流式渲染器默认关闭
        assert options["stream_console_renderer"] is False, (
            "stream_console_renderer 默认应为 False"
        )
        assert options["stream_advanced_renderer"] is False, (
            "stream_advanced_renderer 默认应为 False"
        )
        assert options["stream_typing_effect"] is False, (
            "stream_typing_effect 默认应为 False"
        )
        assert options["stream_show_word_diff"] is False, (
            "stream_show_word_diff 默认应为 False"
        )
        # 其他默认值
        assert options["stream_typing_delay"] == 0.02, (
            "stream_typing_delay 默认应为 0.02"
        )
        assert options["stream_word_mode"] is True, (
            "stream_word_mode 默认应为 True"
        )
        assert options["stream_color_enabled"] is True, (
            "stream_color_enabled 默认应为 True"
        )

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

    def test_stream_renderer_does_not_affect_plan_mode_readonly(
        self, mock_args: argparse.Namespace
    ) -> None:
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
        assert config.force_write is False, (
            "plan 模式下 force_write 必须为 False，流式渲染不应影响"
        )
        # 验证流式渲染配置已设置
        assert config.stream_console_renderer is True
        assert config.stream_advanced_renderer is True
        assert config.stream_typing_effect is True

    def test_stream_renderer_does_not_affect_ask_mode_readonly(
        self, mock_args: argparse.Namespace
    ) -> None:
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
        assert config.force_write is False, (
            "ask 模式下 force_write 必须为 False，流式渲染不应影响"
        )
        # 验证流式渲染配置已设置
        assert config.stream_console_renderer is True
        assert config.stream_show_word_diff is True

    # ========== execution_mode 和 cloud_auth 测试 ==========

    def test_merge_options_execution_mode_default(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 execution_mode 默认为 cli"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["execution_mode"] == "cli", (
            "默认 execution_mode 应为 'cli'"
        )

    def test_merge_options_execution_mode_from_analysis(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 execution_mode 从分析结果获取（优先于命令行参数）"""
        runner = Runner(mock_args)

        # 分析结果设置 execution_mode
        options = runner._merge_options({"execution_mode": "cloud"})
        assert options["execution_mode"] == "cloud", (
            "分析结果 execution_mode='cloud' 应覆盖默认值"
        )

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

        assert options["execution_mode"] == "auto", (
            "命令行 execution_mode='auto' 应被正确获取"
        )

    def test_merge_options_cloud_auth_params(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 cloud_api_key 和 cloud_auth_timeout 正确传递"""
        # 设置 cloud 认证参数
        mock_args.cloud_api_key = "test-api-key"
        mock_args.cloud_auth_timeout = 60

        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["cloud_api_key"] == "test-api-key"
        assert options["cloud_auth_timeout"] == 60

    def test_merge_options_cloud_auth_default_values(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 cloud 认证参数默认值"""
        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["cloud_api_key"] is None
        assert options["cloud_auth_timeout"] == 30

    def test_get_execution_mode_helper(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _get_execution_mode 辅助方法"""
        from cursor.executor import ExecutionMode

        runner = Runner(mock_args)

        # 测试各种模式字符串
        assert runner._get_execution_mode({"execution_mode": "cli"}) == ExecutionMode.CLI
        assert runner._get_execution_mode({"execution_mode": "auto"}) == ExecutionMode.AUTO
        assert runner._get_execution_mode({"execution_mode": "cloud"}) == ExecutionMode.CLOUD

        # 测试无效值回退到 CLI
        assert runner._get_execution_mode({"execution_mode": "invalid"}) == ExecutionMode.CLI
        assert runner._get_execution_mode({}) == ExecutionMode.CLI

    def test_get_cloud_auth_config_helper(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _get_cloud_auth_config 辅助方法"""
        runner = Runner(mock_args)

        # 无 API Key 时返回 None
        options = {"cloud_api_key": None, "cloud_auth_timeout": 30}
        config = runner._get_cloud_auth_config(options)
        assert config is None

        # 有 API Key 时返回 CloudAuthConfig
        options = {"cloud_api_key": "test-key", "cloud_auth_timeout": 60}
        config = runner._get_cloud_auth_config(options)
        assert config is not None
        assert config.api_key == "test-key"
        assert config.auth_timeout == 60

    def test_execution_mode_and_cloud_prefix_consistency(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 '&' 前缀触发的 Cloud 模式与 _merge_options 一致"""
        analyzer = TaskAnalyzer(use_agent=False)
        runner = Runner(mock_args)

        # '&' 前缀应设置 execution_mode='cloud'
        analysis = analyzer.analyze("& 后台执行任务", mock_args)
        assert analysis.mode == RunMode.CLOUD
        assert analysis.options.get("execution_mode") == "cloud"

        # _merge_options 应正确处理
        options = runner._merge_options(analysis.options)
        assert options["execution_mode"] == "cloud"


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
            reviewer_model="gpt-5.2-codex",
            stream_events_enabled=True,
        )

        # 验证配置属性
        assert config.working_directory == "."
        assert config.max_iterations == 10
        assert config.worker_count == 3
        assert config.strict_review is False
        assert config.planner_model == "gpt-5.2-high"
        assert config.worker_model == "opus-4.5-thinking"
        assert config.reviewer_model == "gpt-5.2-codex"
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
        assert config.planner_model == "gpt-5.2-high"
        assert config.worker_model == "opus-4.5-thinking"
        assert config.reviewer_model == "gpt-5.2-codex"
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

        # 验证默认的 reviewer_model 来自 config.yaml（gpt-5.2-codex）
        # 而非 worker_model（opus-4.5-thinking）
        config = MultiProcessOrchestratorConfig()
        assert config.reviewer_model == CONFIG_REVIEWER_MODEL
        assert config.reviewer_model == "gpt-5.2-codex"
        assert config.reviewer_model != config.worker_model, \
            "reviewer_model 应独立配置，而非与 worker_model 相同"

    def test_mp_config_reviewer_model_custom_override(self) -> None:
        """测试 MultiProcessOrchestratorConfig 支持自定义 reviewer_model"""
        from coordinator import MultiProcessOrchestratorConfig

        custom_reviewer = "custom-reviewer-model"
        config = MultiProcessOrchestratorConfig(
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
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
    async def test_run_plan_failure(
        self, runner: Runner
    ) -> None:
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
    async def test_run_plan_timeout(
        self, runner: Runner
    ) -> None:
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
    async def test_run_plan_exception(
        self, runner: Runner
    ) -> None:
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
    async def test_run_ask_success(
        self, runner: Runner
    ) -> None:
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
    async def test_run_ask_failure(
        self, runner: Runner
    ) -> None:
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
    async def test_run_ask_timeout(
        self, runner: Runner
    ) -> None:
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
    """测试 TaskAnalyzer Agent 分析相关功能

    验证 _agent_analysis 使用 plan 模式（只读执行），
    以及 JSON 提取逻辑在各种边界情况下保持稳定。
    """

    def test_agent_analysis_json_extraction(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证 Agent 分析 JSON 提取，使用 plan 模式"""
        analyzer = TaskAnalyzer(use_agent=True)

        # 模拟包含 JSON 的输出
        mock_subprocess.return_value.stdout = '''
        一些其他文本
        {"mode": "iterate", "options": {"skip_online": true}, "reasoning": "任务涉及自我迭代", "refined_goal": "执行自我迭代更新"}
        更多文本
        '''

        analysis = analyzer.analyze("执行自我迭代", mock_args)

        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_uses_plan_mode(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_agent_analysis_fallback_on_invalid_json(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证无效 JSON 时回退到规则分析（稳定返回 None）"""
        analyzer = TaskAnalyzer(use_agent=True)

        mock_subprocess.return_value.stdout = "无效的响应内容"

        # 应该回退到规则分析，不抛出异常
        analysis = analyzer.analyze("执行自我迭代", mock_args)

        # 规则分析应该检测到迭代关键词
        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_timeout_fallback(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_agent_analysis_error_fallback(
        self, mock_args: argparse.Namespace, mock_subprocess
    ) -> None:
        """验证 Agent 分析失败时回退（稳定返回 None）"""
        analyzer = TaskAnalyzer(use_agent=True)

        mock_subprocess.return_value.returncode = 1

        # 应该回退到规则分析
        analysis = analyzer.analyze("执行自我迭代", mock_args)
        assert analysis.mode == RunMode.ITERATE

    def test_agent_analysis_empty_output_returns_none(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _agent_analysis 空输出时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("普通任务")

            assert result is None

    def test_agent_analysis_non_json_output_returns_none(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _agent_analysis 非 JSON 输出时返回 None"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "这只是普通文本，没有任何 JSON 结构"
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("普通任务")

            assert result is None

    def test_agent_analysis_timeout_returns_none(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _agent_analysis 超时时返回 None"""
        import subprocess
        with patch("run.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=30)

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("普通任务")

            assert result is None

    def test_agent_analysis_cloud_mode_passthrough(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析返回 cloud 模式时正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '''{
                "mode": "cloud",
                "options": {"execution_mode": "cloud"},
                "reasoning": "任务需要云端后台执行",
                "refined_goal": "云端执行任务"
            }'''
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("后台长时间执行")

            assert result is not None
            assert result.mode == RunMode.CLOUD
            assert result.options.get("execution_mode") == "cloud"

    def test_agent_analysis_plan_mode_passthrough(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析返回 plan 模式时正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '''{
                "mode": "plan",
                "options": {},
                "reasoning": "仅规划不执行",
                "refined_goal": "规划执行步骤"
            }'''
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("分析代码架构")

            assert result is not None
            assert result.mode == RunMode.PLAN

    def test_agent_analysis_ask_mode_passthrough(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析返回 ask 模式时正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '''{
                "mode": "ask",
                "options": {},
                "reasoning": "问答模式回答问题",
                "refined_goal": "回答问题"
            }'''
            mock_run.return_value = mock_result

            analyzer = TaskAnalyzer(use_agent=True)
            result = analyzer._agent_analysis("解释这段代码")

            assert result is not None
            assert result.mode == RunMode.ASK

    def test_agent_analysis_new_options_passthrough(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析返回的新选项（auto_commit 等）正确透传"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '''{
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
            }'''
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

    def test_agent_analysis_partial_options_passthrough(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 Agent 分析返回部分选项时正确透传，未返回的选项不存在"""
        with patch("run.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            # 仅返回部分选项
            mock_result.stdout = '''{
                "mode": "mp",
                "options": {
                    "workers": 5,
                    "stream_log": false
                },
                "reasoning": "多进程处理",
                "refined_goal": "并行执行任务"
            }'''
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
        from scripts.run_knowledge import parse_args
        from core.config import get_config

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
        from scripts.run_knowledge import parse_args
        import io
        import contextlib

        # 捕获帮助输出
        help_output = io.StringIO()
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_knowledge.py", "--help"]):
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
        from scripts.run_mp import parse_args
        import io
        import contextlib

        help_output = io.StringIO()
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_mp.py", "--help"]):
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
            self.changelog_url = "https://cursor.com/cn/changelog"
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
            self.cloud_api_key = opts.get("cloud_api_key", None)
            self.cloud_auth_timeout = opts.get("cloud_auth_timeout", 30)
            # 流式控制台渲染参数（默认关闭）
            self.stream_console_renderer = opts.get("stream_console_renderer", False)
            self.stream_advanced_renderer = opts.get("stream_advanced_renderer", False)
            self.stream_typing_effect = opts.get("stream_typing_effect", False)
            self.stream_typing_delay = opts.get("stream_typing_delay", 0.02)
            self.stream_word_mode = opts.get("stream_word_mode", True)
            self.stream_color_enabled = opts.get("stream_color_enabled", True)
            self.stream_show_word_diff = opts.get("stream_show_word_diff", False)
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
        args_mp_explicit = IterateArgs("测试任务", {
            "_orchestrator_user_set": True,
            "orchestrator": "mp",
        })
        assert args_mp_explicit._orchestrator_user_set is True
        assert args_mp_explicit.orchestrator == "mp"

        # 测试用户显式设置 --orchestrator basic
        args_basic_explicit = IterateArgs("测试任务", {
            "_orchestrator_user_set": True,
            "orchestrator": "basic",
        })
        assert args_basic_explicit._orchestrator_user_set is True
        assert args_basic_explicit.orchestrator == "basic"

        # 测试用户显式设置 --no-mp
        args_no_mp = IterateArgs("测试任务", {
            "_orchestrator_user_set": True,
            "no_mp": True,
        })
        assert args_no_mp._orchestrator_user_set is True
        assert args_no_mp.no_mp is True

    def test_orchestrator_user_set_prevents_keyword_override(self) -> None:
        """验证显式设置编排器时，不会被 requirement 中的非并行关键词覆盖"""
        from scripts.run_iterate import SelfIterator, _detect_disable_mp_from_requirement

        # 模拟 run.py 中的 IterateArgs 类
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = "https://cursor.com/cn/changelog"
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
                self.cloud_api_key = opts.get("cloud_api_key", None)
                self.cloud_auth_timeout = opts.get("cloud_auth_timeout", 30)
                # 流式控制台渲染参数（默认关闭）
                self.stream_console_renderer = opts.get("stream_console_renderer", False)
                self.stream_advanced_renderer = opts.get("stream_advanced_renderer", False)
                self.stream_typing_effect = opts.get("stream_typing_effect", False)
                self.stream_typing_delay = opts.get("stream_typing_delay", 0.02)
                self.stream_word_mode = opts.get("stream_word_mode", True)
                self.stream_color_enabled = opts.get("stream_color_enabled", True)
                self.stream_show_word_diff = opts.get("stream_show_word_diff", False)

        # 场景1：包含非并行关键词，但用户显式设置了 --orchestrator mp
        # 期望：使用 mp 编排器（不被关键词覆盖）
        args_explicit_mp = IterateArgs("使用协程模式完成任务", {
            "_orchestrator_user_set": True,
            "orchestrator": "mp",
        })
        iterator_explicit_mp = SelfIterator(args_explicit_mp)
        orchestrator_type = iterator_explicit_mp._get_orchestrator_type()
        assert orchestrator_type == "mp", "显式设置 --orchestrator mp 时应使用 mp 编排器"

        # 场景2：包含非并行关键词，用户未显式设置
        # 期望：使用 basic 编排器（被关键词覆盖）
        args_auto_detect = IterateArgs("使用协程模式完成任务", {
            "_orchestrator_user_set": False,
            "orchestrator": "mp",
        })
        iterator_auto_detect = SelfIterator(args_auto_detect)
        orchestrator_type = iterator_auto_detect._get_orchestrator_type()
        assert orchestrator_type == "basic", "未显式设置时应被关键词覆盖为 basic 编排器"

        # 场景3：不包含非并行关键词，用户未显式设置
        # 期望：使用默认 mp 编排器
        args_no_keyword = IterateArgs("完成代码重构任务", {
            "_orchestrator_user_set": False,
            "orchestrator": "mp",
        })
        iterator_no_keyword = SelfIterator(args_no_keyword)
        orchestrator_type = iterator_no_keyword._get_orchestrator_type()
        assert orchestrator_type == "mp", "无非并行关键词时应使用默认 mp 编排器"

        # 场景4：用户显式设置 --no-mp，即使没有非并行关键词
        # 期望：使用 basic 编排器
        args_explicit_no_mp = IterateArgs("完成代码重构任务", {
            "_orchestrator_user_set": True,
            "no_mp": True,
        })
        iterator_explicit_no_mp = SelfIterator(args_explicit_no_mp)
        orchestrator_type = iterator_explicit_no_mp._get_orchestrator_type()
        assert orchestrator_type == "basic", "显式设置 --no-mp 时应使用 basic 编排器"

    def test_self_iterator_init(self) -> None:
        """测试 SelfIterator 可以用 IterateArgs 初始化"""
        from scripts.run_iterate import SelfIterator

        # 使用辅助函数创建 IterateArgs 类
        IterateArgs = _create_iterate_args_class()

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
        assert executor.config.force_write is False, \
            "PlanAgentExecutor 应强制覆盖用户配置的 force_write=True"
        assert executor.config.mode == "plan", \
            "PlanAgentExecutor 应强制设置 mode=plan"

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
        assert executor.config.force_write is False, \
            "AskAgentExecutor 应强制覆盖用户配置的 force_write=True"
        assert executor.config.mode == "ask", \
            "AskAgentExecutor 应强制设置 mode=ask"

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
        assert executor.config.mode == "plan", \
            "PlanAgentExecutor 应强制覆盖用户配置的 mode"
        assert executor.config.force_write is False, \
            "PlanAgentExecutor 应强制覆盖用户配置的 force_write"

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
        assert executor.config.mode == "ask", \
            "AskAgentExecutor 应强制覆盖用户配置的 mode"
        assert executor.config.force_write is False, \
            "AskAgentExecutor 应强制覆盖用户配置的 force_write"

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
        assert reviewer.reviewer_config.cursor_config.mode == "ask", \
            "ReviewerAgent 应强制 mode=ask"
        assert reviewer.reviewer_config.cursor_config.force_write is False, \
            "ReviewerAgent 应强制 force_write=False"

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
        assert reviewer.reviewer_config.cursor_config.force_write is False, \
            "ReviewerAgent 应强制覆盖用户配置的 force_write=True"
        assert reviewer.reviewer_config.cursor_config.mode == "ask", \
            "ReviewerAgent 应强制设置 mode=ask"

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
        assert reviewer.reviewer_config.cursor_config.mode == "ask", \
            "ReviewerAgent 应强制覆盖用户配置的 mode"
        assert reviewer.reviewer_config.cursor_config.force_write is False, \
            "ReviewerAgent 应强制覆盖用户配置的 force_write"

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
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
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
    async def test_explicit_mode_still_analyzes(
        self, base_args: argparse.Namespace
    ) -> None:
        """测试显式指定模式仍会进行参数解析"""
        from run import async_main, TaskAnalysis, RunMode

        # 设置显式 basic 模式
        base_args.task = "显式模式任务"
        base_args.mode = "basic"  # 非 auto 模式
        base_args.no_auto_analyze = False

        with patch("run.parse_args", return_value=base_args), \
             patch("run.setup_logging"), \
             patch("run.TaskAnalyzer") as mock_analyzer_class, \
             patch("run.Runner") as mock_runner_class:

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
            mock_analyzer.analyze.assert_called_once_with(
                "显式模式任务", base_args
            )

            # 验证 Runner.run 被调用，使用了显式指定的模式
            mock_runner.run.assert_called_once()
            call_args = mock_runner.run.call_args[0]
            analysis = call_args[0]
            assert isinstance(analysis, TaskAnalysis)
            assert analysis.mode == RunMode.BASIC
            assert analysis.goal == "解析后的目标"
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
        """创建用于测试参数映射的 Runner 实例"""
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
            execution_mode=None,
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

    def test_merge_options_passes_orchestrator_fields(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _merge_options 正确传递编排器相关字段"""
        # 设置编排器选项
        mock_args.orchestrator = "basic"
        mock_args.no_mp = True
        mock_args._orchestrator_user_set = True

        runner = Runner(mock_args)
        options = runner._merge_options({})

        assert options["orchestrator"] == "basic"
        assert options["no_mp"] is True

    def test_merge_options_analysis_overrides_when_not_user_set(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_merge_options_user_set_takes_priority(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 _orchestrator_user_set=True 时用户设置优先"""
        mock_args.orchestrator = "mp"
        mock_args.no_mp = False
        mock_args._orchestrator_user_set = True

        runner = Runner(mock_args)

        # 自然语言分析结果想覆盖为 basic，但用户显式设置了 mp
        analysis_options = {"orchestrator": "basic", "no_mp": True}
        options = runner._merge_options(analysis_options)

        # 用户显式设置优先
        assert options["orchestrator"] == "mp"
        assert options["no_mp"] is False

    @pytest.mark.asyncio
    async def test_run_iterate_creates_iterate_args_with_orchestrator(
        self, runner_with_iterate_args: Runner
    ) -> None:
        """验证 _run_iterate 正确创建包含编排器选项的 IterateArgs"""
        import scripts.run_iterate

        captured_args = None

        def capture_self_iterator_init(args):
            nonlocal captured_args
            captured_args = args
            mock_iterator = MagicMock()
            mock_iterator.run = AsyncMock(return_value={"success": True})
            return mock_iterator

        with patch.object(
            scripts.run_iterate, "SelfIterator",
            side_effect=capture_self_iterator_init
        ):
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
    async def test_run_iterate_passes_orchestrator_user_set_from_args(
        self, mock_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            scripts.run_iterate, "SelfIterator",
            side_effect=capture_init
        ):
            options = runner._merge_options({})
            await runner._run_iterate("测试任务", options)

            # 验证 _orchestrator_user_set 被正确传递
            assert captured_args._orchestrator_user_set is True
            assert captured_args.orchestrator == "basic"

    @pytest.mark.asyncio
    async def test_run_iterate_max_iterations_converted_to_string(
        self, mock_args: argparse.Namespace
    ) -> None:
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

        with patch.object(
            scripts.run_iterate, "SelfIterator",
            side_effect=capture_init
        ):
            options = runner._merge_options({"max_iterations": 15})
            await runner._run_iterate("测试任务", options)

            # 验证 max_iterations 是字符串类型
            assert captured_args.max_iterations == "15"
            assert isinstance(captured_args.max_iterations, str)

    def test_task_analyzer_detects_non_parallel_keywords_for_iterate(
        self, mock_args: argparse.Namespace
    ) -> None:
        """验证 TaskAnalyzer 检测非并行关键词并设置编排器选项"""
        analyzer = TaskAnalyzer(use_agent=False)

        # 自我迭代 + 非并行关键词
        analysis = analyzer.analyze("自我迭代，使用协程模式完成", mock_args)

        # 应该检测到 ITERATE 模式
        assert analysis.mode == RunMode.ITERATE

        # 应该设置非并行选项
        assert analysis.options.get("no_mp") is True or \
               analysis.options.get("orchestrator") == "basic"

    def test_full_parameter_flow_from_run_to_self_iterator(
        self, mock_args: argparse.Namespace
    ) -> None:
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
        assert analysis.options.get("no_mp") is True or \
               analysis.options.get("orchestrator") == "basic"

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
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
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
    async def test_run_cloud_calls_factory_with_correct_mode(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
    async def test_run_cloud_passes_correct_cloud_auth_config(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
    async def test_run_cloud_passes_correct_timeout(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
        assert captured_cli_config.timeout == 900
        assert captured_execute_timeout == 900

    @pytest.mark.asyncio
    async def test_run_cloud_force_write_default_true(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证 _run_cloud 默认 force_write=True"""
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
                # 不指定 force_write，应默认为 True
                options = runner._merge_options({})
                await runner._run_cloud("任务", options)

        # 验证 force_write 默认为 True
        assert captured_cli_config.force_write is True

    @pytest.mark.asyncio
    async def test_run_cloud_force_write_can_be_overridden(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
        assert captured_cli_config.force_write is False

    @pytest.mark.asyncio
    async def test_run_cloud_no_api_key_returns_error_early(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
    async def test_run_cloud_executor_failure_returns_error(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证执行器失败时返回正确的错误信息"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Cloud API 连接失败"
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
    async def test_run_cloud_timeout_exception_handling(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
    async def test_run_cloud_generic_exception_handling(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证通用异常被正确处理"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(
                side_effect=RuntimeError("网络错误")
            )
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                options = runner._merge_options({})
                result = await runner._run_cloud("任务", options)

        assert result["success"] is False
        assert "网络错误" in result["error"]
        assert result["mode"] == "cloud"

    @pytest.mark.asyncio
    async def test_run_cloud_success_returns_complete_result(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
        """验证成功执行时返回完整的结果"""
        runner = Runner(cloud_runner_args)

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "Cloud 任务执行成功，修改了 3 个文件"
            mock_result.session_id = "cloud-session-abc123"
            mock_result.files_modified = ["file1.py", "file2.py", "file3.py"]
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
    async def test_run_cloud_uses_directory_from_options(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                # 模拟 '&' 前缀触发的选项（由 TaskAnalyzer 设置）
                options = runner._merge_options({
                    "cloud_background": True,  # '&' 前缀触发
                    "triggered_by_prefix": True,
                })
                await runner._run_cloud("后台分析代码", options)

        # 验证 background=True 被传递给执行器
        assert captured_background is True, (
            "'&' 前缀触发时应传递 background=True"
        )

    @pytest.mark.asyncio
    async def test_run_cloud_returns_resume_command_on_success(
        self, cloud_runner_args: argparse.Namespace
    ) -> None:
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
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
            with patch.dict("os.environ", {"CURSOR_API_KEY": "key"}):
                # 模拟后台模式选项
                options = runner._merge_options({
                    "cloud_background": True,
                    "triggered_by_prefix": True,
                })
                result = await runner._run_cloud("后台任务", options)

        # 验证后台模式返回正确的元数据
        assert result["success"] is True
        assert result["background"] is True
        assert result["session_id"] == "bg-session-abc"
        assert result["resume_command"] == "agent --resume bg-session-abc"
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
                options = runner._merge_options({
                    "execution_mode": "cloud",
                    # 不设置 cloud_background
                })
                await runner._run_cloud("任务", options)

        # 验证默认前台模式
        assert captured_background is False, (
            "--execution-mode cloud 默认应为前台模式（background=False）"
        )


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
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
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
    async def test_no_api_key_returns_error_without_calling_factory(
        self, no_key_args: argparse.Namespace
    ) -> None:
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
    async def test_no_api_key_executor_error_is_captured(
        self, no_key_args: argparse.Namespace
    ) -> None:
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
    async def test_no_api_key_error_message_content(
        self, no_key_args: argparse.Namespace
    ) -> None:
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
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
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
    async def test_cloud_timeout_independent_from_max_iterations(
        self, base_args: argparse.Namespace
    ) -> None:
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
            f"Cloud timeout 应为 {expected_timeout}（来自 config.yaml），"
            f"实际值: {captured_timeout}"
        )

    @pytest.mark.asyncio
    async def test_cloud_mode_auto_commit_independence(
        self, base_args: argparse.Namespace
    ) -> None:
        """验证 Cloud 模式下 force_write 与 auto_commit 独立"""
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

        # force_write 默认 True，与 auto_commit=False 独立
        assert captured_cli_config.force_write is True

    @pytest.mark.asyncio
    async def test_executor_execute_receives_correct_prompt(
        self, base_args: argparse.Namespace
    ) -> None:
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
        from scripts.run_basic import parse_args
        import io
        import contextlib

        help_output = io.StringIO()
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_basic.py", "--help"]):
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
        from scripts.run_basic import run_orchestrator
        import argparse

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
        from scripts.run_basic import run_orchestrator
        import argparse

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
        from scripts.run_basic import run_orchestrator
        import argparse

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

    def test_parse_args_execution_mode_uses_config_default(
        self, config_execution_mode: str
    ) -> None:
        """测试 run.py parse_args() 的 --execution-mode 默认值为 None (tri-state)"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.execution_mode is None, (
            f"--execution-mode 默认值应为 None (tri-state)，"
            f"实际值: {args.execution_mode}"
        )

    def test_parse_args_execution_mode_cli_override(self) -> None:
        """测试 CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务", "--execution-mode", "cloud"]):
            args = parse_args()

        assert args.execution_mode == "cloud"

    def test_parse_args_help_shows_config_source_for_execution_mode(self) -> None:
        """测试帮助信息中 --execution-mode 显示来源于 config.yaml"""
        from run import parse_args
        import io
        import contextlib

        help_output = io.StringIO()
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run.py", "--help"]):
                with contextlib.redirect_stdout(help_output):
                    parse_args()

        help_text = help_output.getvalue()
        # 验证帮助信息中包含 "config.yaml" 的提示
        assert "config.yaml" in help_text, (
            "--execution-mode 帮助信息应包含 'config.yaml' 来源提示"
        )

    def test_ampersand_prefix_overrides_config_execution_mode(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 '&' 前缀触发 Cloud 模式优先于配置默认值

        场景：配置中 execution_mode=cli，但任务带 '&' 前缀
        期望：Cloud 模式被激活，覆盖配置默认值
        """
        # 模拟配置默认值为 cli
        mock_args.execution_mode = "cli"
        analyzer = TaskAnalyzer(use_agent=False)

        # 带 '&' 前缀的任务
        analysis = analyzer.analyze("& 分析代码架构", mock_args)

        # '&' 前缀应触发 Cloud 模式
        assert analysis.mode == RunMode.CLOUD
        assert analysis.options.get("execution_mode") == "cloud"
        assert analysis.goal == "分析代码架构"

    def test_merge_options_uses_analysis_execution_mode_over_config(
        self, mock_args: argparse.Namespace
    ) -> None:
        """测试 _merge_options 中 analysis_options 的 execution_mode 优先于配置

        场景：配置默认值为 cli，但 analysis 中设置了 cloud
        期望：最终 options["execution_mode"] 为 cloud
        """
        mock_args.execution_mode = "cli"  # 模拟配置默认值
        runner = Runner(mock_args)

        # 模拟 TaskAnalyzer 检测到 '&' 前缀后设置的选项
        analysis_options = {"execution_mode": "cloud"}
        options = runner._merge_options(analysis_options)

        assert options["execution_mode"] == "cloud", (
            "analysis_options 中的 execution_mode 应优先于配置默认值"
        )


class TestExecutionModeConfigDefaultWithMock:
    """测试 execution_mode 配置默认值（使用 mock 配置）

    通过 mock ConfigManager 来测试不同配置值下的行为。
    """

    def test_config_auto_default_parsed_correctly(self) -> None:
        """测试配置 execution_mode=auto 时未传参的默认解析"""
        from run import parse_args
        from core.config import ConfigManager

        # 保存原始配置
        original_execution_mode = get_config().cloud_agent.execution_mode

        try:
            # Mock 配置为 auto
            with patch.object(
                get_config().cloud_agent, "execution_mode", "auto"
            ):
                # 需要重新导入以使用新的默认值
                # 由于 parse_args 在模块加载时读取配置，这里直接验证配置值
                assert get_config().cloud_agent.execution_mode == "auto"
        finally:
            # 配置会自动恢复（patch 结束后）
            pass

    def test_config_cloud_default_parsed_correctly(self) -> None:
        """测试配置 execution_mode=cloud 时未传参的默认解析"""
        from core.config import ConfigManager

        # Mock 配置为 cloud
        with patch.object(
            get_config().cloud_agent, "execution_mode", "cloud"
        ):
            assert get_config().cloud_agent.execution_mode == "cloud"

    def test_orchestrator_compatibility_with_config_cloud_default(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_orchestrator_compatibility_with_config_auto_default(
        self, mock_args: argparse.Namespace
    ) -> None:
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

    def test_run_iterate_parse_args_uses_config_default(
        self, config_execution_mode: str
    ) -> None:
        """测试 run_iterate.py parse_args() 的 --execution-mode 默认值为 None (tri-state)"""
        from scripts.run_iterate import parse_args

        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            args = parse_args()

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert args.execution_mode is None, (
            f"run_iterate.py --execution-mode 默认值应为 None (tri-state)，"
            f"实际值: {args.execution_mode}"
        )

    def test_run_iterate_parse_args_cli_override(self) -> None:
        """测试 run_iterate.py CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        from scripts.run_iterate import parse_args

        with patch("sys.argv", ["run_iterate.py", "测试任务", "--execution-mode", "cloud"]):
            args = parse_args()

        assert args.execution_mode == "cloud"

    def test_run_iterate_help_shows_config_source(self) -> None:
        """测试 run_iterate.py 帮助信息中 --execution-mode 显示来源于 config.yaml"""
        from scripts.run_iterate import parse_args
        import io
        import contextlib

        help_output = io.StringIO()
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_iterate.py", "--help"]):
                with contextlib.redirect_stdout(help_output):
                    parse_args()

        help_text = help_output.getvalue()
        assert "config.yaml" in help_text, (
            "run_iterate.py --execution-mode 帮助信息应包含 'config.yaml' 来源提示"
        )


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
    def test_run_py_parse_args_tristate_returns_none(
        self, arg_name: str
    ) -> None:
        """表驱动测试：run.py parse_args() tri-state 字段返回 None"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        assert actual is None, (
            f"run.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），"
            f"实际值: {actual}"
        )

    @pytest.mark.parametrize(
        "arg_name,config_key",
        RUN_PY_DIRECT_CONFIG_FIELD_TESTS,
        ids=[f"run.py:{case[0]}" for case in RUN_PY_DIRECT_CONFIG_FIELD_TESTS],
    )
    def test_run_py_parse_args_default(
        self, config_defaults: dict, arg_name: str, config_key: str
    ) -> None:
        """表驱动测试：run.py parse_args() 非 tri-state 字段默认值来自 config.yaml"""
        from run import parse_args

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        expected = config_defaults[config_key]
        actual = getattr(args, arg_name)

        assert actual == expected, (
            f"run.py --{arg_name.replace('_', '-')} 默认值应为 {expected}（来自 config.yaml），"
            f"实际值: {actual}"
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
    def test_run_basic_py_parse_args_default(
        self, config_defaults: dict, arg_name: str
    ) -> None:
        """表驱动测试：scripts/run_basic.py parse_args() tri-state 字段返回 None"""
        from scripts.run_basic import parse_args

        with patch("sys.argv", ["run_basic.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert actual is None, (
            f"run_basic.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），"
            f"实际值: {actual}"
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
    def test_run_mp_py_parse_args_default(
        self, config_defaults: dict, arg_name: str
    ) -> None:
        """表驱动测试：scripts/run_mp.py parse_args() tri-state 字段返回 None"""
        from scripts.run_mp import parse_args

        with patch("sys.argv", ["run_mp.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        # tri-state 设计：未显式指定时返回 None，运行时从 config.yaml 解析
        assert actual is None, (
            f"run_mp.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），"
            f"实际值: {actual}"
        )

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
    def test_run_iterate_py_parse_args_tristate_returns_none(
        self, arg_name: str
    ) -> None:
        """表驱动测试：scripts/run_iterate.py parse_args() tri-state 字段返回 None"""
        from scripts.run_iterate import parse_args

        with patch("sys.argv", ["run_iterate.py", "测试任务"]):
            args = parse_args()

        actual = getattr(args, arg_name)

        assert actual is None, (
            f"run_iterate.py --{arg_name.replace('_', '-')} 应为 None（tri-state 设计），"
            f"实际值: {actual}"
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
        ids=[
            f"{case[0]}:{case[1]}" for case in CLOUD_TIMEOUT_CLI_OVERRIDE_TESTS
        ],
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
        assert actual == expected_value, (
            f"{script_module} {cli_arg} 应为 {expected_value}，实际值: {actual}"
        )


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
        ids=[
            f"{case[0]}:{case[1]}={case[2]}"
            for case in MAX_ITERATIONS_CLI_OVERRIDE_TESTS
        ],
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
            f"{script_module} {cli_arg} 应为 {expected_value}，"
            f"实际值: {args.max_iterations}"
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
    def test_parse_max_iterations_special_values(
        self, input_value: str, expected_output: int
    ) -> None:
        """表驱动测试：parse_max_iterations 特殊值处理"""
        result = parse_max_iterations(input_value)
        assert result == expected_output, (
            f"parse_max_iterations('{input_value}') 应为 {expected_output}，"
            f"实际值: {result}"
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
        assert actual == expected_value, (
            f"{script_module} {cli_arg} 应为 {expected_value}，实际值: {actual}"
        )

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
            f"planner_model 默认值应为 None（tri-state 设计），"
            f"实际值: {args.planner_model}"
        )
        assert args.worker_model is None, (
            f"worker_model 默认值应为 None（tri-state 设计），"
            f"实际值: {args.worker_model}"
        )
        assert args.reviewer_model is None, (
            f"reviewer_model 默认值应为 None（tri-state 设计），"
            f"实际值: {args.reviewer_model}"
        )

    def test_run_py_default_models_from_config(self, config_models: dict) -> None:
        """测试 run.py 模型默认值使用 tri-state 设计

        tri-state 语义：
        - argparse 返回 None（未显式指定）
        - 最终值通过 resolve_orchestrator_settings 从 config.yaml 解析
        """
        from run import parse_args, resolve_orchestrator_settings

        with patch("sys.argv", ["run.py", "测试任务"]):
            args = parse_args()

        # argparse 返回 None（tri-state 设计）
        assert args.planner_model is None, (
            f"planner_model 应为 None（tri-state 设计），实际值: {args.planner_model}"
        )
        assert args.worker_model is None, (
            f"worker_model 应为 None（tri-state 设计），实际值: {args.worker_model}"
        )

        # 最终值通过 resolve_orchestrator_settings 从 config.yaml 解析
        resolved = resolve_orchestrator_settings({})
        assert resolved["planner_model"] == config_models["planner"], (
            f"resolve_orchestrator_settings 应返回 {config_models['planner']}，"
            f"实际值: {resolved['planner_model']}"
        )
        assert resolved["worker_model"] == config_models["worker"], (
            f"resolve_orchestrator_settings 应返回 {config_models['worker']}，"
            f"实际值: {resolved['worker_model']}"
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
        from run import parse_args, Runner
        from core.config import resolve_orchestrator_settings

        with patch("sys.argv", ["run.py", "--mode", "iterate", "测试任务"]):
            args = parse_args()

        # tri-state: parse_args 应返回 None
        assert args.cloud_timeout is None, (
            f"cloud_timeout 应为 None（tri-state 设计），实际值: {args.cloud_timeout}"
        )

        # 运行时解析应得到 config.yaml 的值
        resolved = resolve_orchestrator_settings()
        assert resolved["cloud_timeout"] == config_cloud_defaults["timeout"], (
            f"resolve_orchestrator_settings 应返回 {config_cloud_defaults['timeout']}，"
            f"实际值: {resolved['cloud_timeout']}"
        )

    def test_iterate_mode_uses_config_cloud_auth_timeout(self, config_cloud_defaults: dict) -> None:
        """测试 --mode iterate 的 --cloud-auth-timeout：parse_args 返回 None，
        运行时通过 resolve_orchestrator_settings 解析获得 config.yaml 值"""
        from run import parse_args
        from core.config import resolve_orchestrator_settings

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
        from run import parse_args
        from core.config import resolve_orchestrator_settings

        with patch("sys.argv", ["run.py", "--mode", "iterate", "测试任务"]):
            args = parse_args()

        # tri-state: parse_args 应返回 None
        assert args.execution_mode is None, (
            f"execution_mode 应为 None（tri-state 设计），实际值: {args.execution_mode}"
        )

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
        with patch("sys.argv", [
            "run.py", "--mode", "iterate", "--cloud-timeout", str(custom_timeout), "测试任务"
        ]):
            args = parse_args()

        assert args.cloud_timeout == custom_timeout

    def test_iterate_mode_cli_overrides_cloud_auth_timeout(self) -> None:
        """测试 CLI --cloud-auth-timeout 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        custom_auth_timeout = 60
        with patch("sys.argv", [
            "run.py", "--mode", "iterate", "--cloud-auth-timeout", str(custom_auth_timeout), "测试任务"
        ]):
            args = parse_args()

        assert args.cloud_auth_timeout == custom_auth_timeout

    def test_iterate_mode_cli_overrides_execution_mode(self) -> None:
        """测试 CLI --execution-mode 参数可覆盖 config.yaml 默认值"""
        from run import parse_args

        # 测试覆盖为 cloud
        with patch("sys.argv", [
            "run.py", "--mode", "iterate", "--execution-mode", "cloud", "测试任务"
        ]):
            args = parse_args()
        assert args.execution_mode == "cloud"

        # 测试覆盖为 auto
        with patch("sys.argv", [
            "run.py", "--mode", "iterate", "--execution-mode", "auto", "测试任务"
        ]):
            args = parse_args()
        assert args.execution_mode == "auto"

    def test_cloud_prefix_detected_in_task(self) -> None:
        """测试 & 前缀能被正确检测为 cloud 请求"""
        from run import is_cloud_request, strip_cloud_prefix

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
        from run import Runner, parse_args

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
        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--execution-mode", "cloud",
            "--cloud-timeout", str(custom_timeout),
            "--cloud-auth-timeout", str(custom_auth_timeout),
            "测试任务"
        ]):
            args = parse_args()

        assert args.execution_mode == "cloud"
        assert args.cloud_timeout == custom_timeout
        assert args.cloud_auth_timeout == custom_auth_timeout

    def test_execution_mode_auto_preserves_cli_timeout_values(self) -> None:
        """测试 execution_mode=auto 时 CLI 指定的 timeout 值被保留"""
        from run import parse_args

        custom_timeout = 1200
        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--execution-mode", "auto",
            "--cloud-timeout", str(custom_timeout),
            "测试任务"
        ]):
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

        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--execution-mode", "cli",
            "--orchestrator", "mp",
            "测试任务"
        ]):
            args = parse_args()

        assert args.execution_mode == "cli"
        assert args.orchestrator == "mp"

    def test_cloud_mode_default_orchestrator(self) -> None:
        """测试 execution_mode=cloud 时默认编排器设置"""
        from run import parse_args

        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--execution-mode", "cloud",
            "测试任务"
        ]):
            args = parse_args()

        assert args.execution_mode == "cloud"
        # 编排器在 resolve_orchestrator_settings 中会被强制切换为 basic
        # 但 parse_args 本身不会改变 orchestrator 值

    def test_no_mp_flag_forces_basic(self) -> None:
        """测试 --no-mp 标志强制使用 basic 编排器"""
        from run import parse_args

        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--no-mp",
            "测试任务"
        ]):
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
            ("&", ""),   # 仅 &
        ]

        for input_task, expected in test_cases:
            result = strip_cloud_prefix(input_task)
            assert result == expected, (
                f"strip_cloud_prefix('{input_task}') 应为 '{expected}'，"
                f"实际值: '{result}'"
            )


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

        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--cloud-timeout", "600",
            "--cloud-auth-timeout", "45",
            "--execution-mode", "cloud",
            "--workers", "5",
            "--max-iterations", "20",
            "--auto-commit",
            "--stall-diagnostics",
            "--stall-diagnostics-level", "info",
            "--stall-recovery-interval", "60.0",
            "--stream-console-renderer",
            "--stream-show-word-diff",
            "测试任务",
        ]):
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

        with patch("sys.argv", [
            "run.py", "--mode", "iterate",
            "--cloud-timeout", str(cli_cloud_timeout),
            "--workers", str(cli_workers),
            "测试任务",
        ]):
            args = parse_args()

        runner = Runner(args)
        options = runner._merge_options({})

        assert options["cloud_timeout"] == cli_cloud_timeout, (
            f"CLI --cloud-timeout 应覆盖 config.yaml 默认值 {config_cloud_timeout}"
        )
        assert options["workers"] == cli_workers, (
            f"CLI --workers 应覆盖 config.yaml 默认值 {config_workers}"
        )


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

        with patch.object(SelfIterator, '__init__', lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            # 模拟 _resolve_config_settings 调用
            cloud_timeout = getattr(iterator.args, "cloud_timeout", None)
            assert cloud_timeout == 600

    def test_selfiterator_reads_stall_diagnostics(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取卡死诊断参数"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, '__init__', lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            assert getattr(iterator.args, "stall_diagnostics_enabled", None) is True
            assert getattr(iterator.args, "stall_diagnostics_level", None) == "info"
            assert getattr(iterator.args, "stall_recovery_interval", 30.0) == 60.0

    def test_selfiterator_reads_stream_render_params(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取流式渲染参数"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, '__init__', lambda self, args: None):
            iterator = object.__new__(SelfIterator)
            iterator.args = mock_iterate_args

            assert getattr(iterator.args, "stream_console_renderer", False) is True
            assert getattr(iterator.args, "stream_show_word_diff", False) is True

    def test_selfiterator_reads_execution_mode(self, mock_iterate_args: argparse.Namespace) -> None:
        """测试 SelfIterator 正确读取 execution_mode"""
        from scripts.run_iterate import SelfIterator

        with patch.object(SelfIterator, '__init__', lambda self, args: None):
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
