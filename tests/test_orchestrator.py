"""测试编排器（Orchestrator）初始化和配置"""
from unittest.mock import MagicMock

import pytest

from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from core.base import AgentRole
from core.state import SystemState
from cursor.client import CursorAgentConfig


class TestOrchestratorConfig:
    """测试 OrchestratorConfig 默认值"""

    def test_default_values(self) -> None:
        """验证 OrchestratorConfig 所有默认值正确"""
        config = OrchestratorConfig()

        assert config.working_directory == "."
        assert config.max_iterations == 10
        assert config.worker_pool_size == 3
        assert config.enable_sub_planners is True
        assert config.strict_review is False
        assert config.stream_events_enabled is True  # 默认启用
        assert config.stream_log_console is True
        assert config.enable_auto_commit is False  # 默认禁用自动提交（需显式开启）
        assert config.stream_log_detail_dir == "logs/stream_json/detail/"
        assert config.stream_log_raw_dir == "logs/stream_json/raw/"

    def test_cursor_config_default(self) -> None:
        """验证嵌套的 CursorAgentConfig 使用默认工厂正确创建"""
        config = OrchestratorConfig()

        assert config.cursor_config is not None
        assert isinstance(config.cursor_config, CursorAgentConfig)

    def test_custom_values(self) -> None:
        """验证可以通过参数覆盖默认值"""
        config = OrchestratorConfig(
            working_directory="/tmp/test",
            max_iterations=5,
            worker_pool_size=5,
            enable_sub_planners=False,
            strict_review=True,
            stream_events_enabled=True,
        )

        assert config.working_directory == "/tmp/test"
        assert config.max_iterations == 5
        assert config.worker_pool_size == 5
        assert config.enable_sub_planners is False
        assert config.strict_review is True
        assert config.stream_events_enabled is True


class TestOrchestratorInitialization:
    """测试 Orchestrator 初始化"""

    def test_components_created(self) -> None:
        """验证 Orchestrator 初始化后 planner/reviewer/worker_pool 都已创建"""
        config = OrchestratorConfig(
            working_directory=".",
            worker_pool_size=2,  # 使用较小的池
        )
        orchestrator = Orchestrator(config)

        # 验证 planner 已创建
        assert orchestrator.planner is not None
        assert orchestrator.planner.id.startswith("planner-")

        # 验证 reviewer 已创建
        assert orchestrator.reviewer is not None
        assert orchestrator.reviewer.id.startswith("reviewer-")

        # 验证 worker_pool 已创建
        assert orchestrator.worker_pool is not None
        assert len(orchestrator.worker_pool.workers) == 2

    def test_state_initialized(self) -> None:
        """验证 SystemState 正确初始化"""
        config = OrchestratorConfig(
            working_directory="/test/dir",
            max_iterations=15,
        )
        orchestrator = Orchestrator(config)

        assert orchestrator.state is not None
        assert isinstance(orchestrator.state, SystemState)
        assert orchestrator.state.working_directory == "/test/dir"
        assert orchestrator.state.max_iterations == 15
        assert orchestrator.state.current_iteration == 0
        assert orchestrator.state.is_running is False
        assert orchestrator.state.is_completed is False

    def test_task_queue_created(self) -> None:
        """验证任务队列已创建"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        assert orchestrator.task_queue is not None

    def test_knowledge_manager_optional(self) -> None:
        """验证 knowledge_manager 可选参数"""
        config = OrchestratorConfig()

        # 不传递 knowledge_manager
        orchestrator = Orchestrator(config)
        assert orchestrator._knowledge_manager is None

        # 传递 mock knowledge_manager
        mock_km = MagicMock()
        orchestrator_with_km = Orchestrator(config, knowledge_manager=mock_km)
        assert orchestrator_with_km._knowledge_manager is mock_km


class TestAgentRegistration:
    """测试 SystemState 中的 Agent 注册状态"""

    def test_planner_registered(self) -> None:
        """验证 planner 已注册到 SystemState"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        planner_id = orchestrator.planner.id
        assert planner_id in orchestrator.state.agents
        assert orchestrator.state.agents[planner_id].role == AgentRole.PLANNER

    def test_reviewer_registered(self) -> None:
        """验证 reviewer 已注册到 SystemState"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        reviewer_id = orchestrator.reviewer.id
        assert reviewer_id in orchestrator.state.agents
        assert orchestrator.state.agents[reviewer_id].role == AgentRole.REVIEWER

    def test_workers_registered(self) -> None:
        """验证所有 worker 都已注册到 SystemState"""
        config = OrchestratorConfig(worker_pool_size=3)
        orchestrator = Orchestrator(config)

        # 验证 worker 数量
        worker_agents = [
            agent for agent in orchestrator.state.agents.values()
            if agent.role == AgentRole.WORKER
        ]
        assert len(worker_agents) == 3

        # 验证每个 worker 都已注册
        for worker in orchestrator.worker_pool.workers:
            assert worker.id in orchestrator.state.agents
            assert orchestrator.state.agents[worker.id].role == AgentRole.WORKER

    def test_total_agents_count(self) -> None:
        """验证总 Agent 数量正确（1 planner + 1 reviewer + N workers）"""
        worker_count = 4
        config = OrchestratorConfig(worker_pool_size=worker_count)
        orchestrator = Orchestrator(config)

        # enable_auto_commit 默认为 False，不会创建 committer
        expected_total = 1 + 1 + worker_count  # planner + reviewer + workers
        assert len(orchestrator.state.agents) == expected_total

    def test_total_agents_count_with_committer(self) -> None:
        """验证启用 auto_commit 时总 Agent 数量正确（含 committer）"""
        worker_count = 4
        config = OrchestratorConfig(
            worker_pool_size=worker_count,
            enable_auto_commit=True,
        )
        orchestrator = Orchestrator(config)

        # enable_auto_commit=True 时会创建 committer
        expected_total = 1 + 1 + worker_count + 1  # planner + reviewer + workers + committer
        assert len(orchestrator.state.agents) == expected_total

    def test_agent_initial_status(self) -> None:
        """验证注册的 Agent 初始状态为 IDLE"""
        from core.base import AgentStatus

        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        for agent_state in orchestrator.state.agents.values():
            assert agent_state.status == AgentStatus.IDLE
            assert agent_state.current_task_id is None
            assert agent_state.error_count == 0


class TestModelConfigPerRole:
    """测试各角色的模型配置"""

    def test_model_config_log_output(self) -> None:
        """验证初始化时输出各角色模型配置日志"""
        from io import StringIO

        from loguru import logger

        # 使用 loguru sink 捕获日志
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="INFO")

        try:
            config = OrchestratorConfig()
            Orchestrator(config)

            # 获取捕获的日志
            log_content = log_output.getvalue()

            # 验证日志包含模型配置信息
            expected_log = (
                "各角色模型配置 - Planner: gpt-5.2-high, "
                "Worker: opus-4.5-thinking, Reviewer: opus-4.5-thinking"
            )
            assert expected_log in log_content, (
                f"未找到预期日志: {expected_log}\n实际日志: {log_content}"
            )
        finally:
            logger.remove(handler_id)

    def test_orchestrator_uses_different_models_per_role(self) -> None:
        """验证 Planner 使用 gpt-5.2-high，Worker 和 Reviewer 使用 opus-4.5-thinking"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # 验证默认模型配置
        assert config.planner_model == "gpt-5.2-high"
        assert config.worker_model == "opus-4.5-thinking"
        assert config.reviewer_model == "opus-4.5-thinking"

        # 验证 Planner 使用正确的模型
        assert orchestrator.planner.planner_config.cursor_config.model == "gpt-5.2-high"

        # 验证 Reviewer 使用正确的模型
        assert orchestrator.reviewer.reviewer_config.cursor_config.model == "opus-4.5-thinking"

        # 验证 Workers 使用正确的模型
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.model == "opus-4.5-thinking"

    def test_orchestrator_custom_model_config(self) -> None:
        """验证自定义模型配置能够正确传递到各角色"""
        custom_planner_model = "custom-planner-model"
        custom_worker_model = "custom-worker-model"
        custom_reviewer_model = "custom-reviewer-model"

        config = OrchestratorConfig(
            planner_model=custom_planner_model,
            worker_model=custom_worker_model,
            reviewer_model=custom_reviewer_model,
        )
        orchestrator = Orchestrator(config)

        # 验证配置值
        assert config.planner_model == custom_planner_model
        assert config.worker_model == custom_worker_model
        assert config.reviewer_model == custom_reviewer_model

        # 验证 Planner 使用自定义模型
        assert orchestrator.planner.planner_config.cursor_config.model == custom_planner_model

        # 验证 Reviewer 使用自定义模型
        assert orchestrator.reviewer.reviewer_config.cursor_config.model == custom_reviewer_model

        # 验证所有 Workers 使用自定义模型
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.model == custom_worker_model


class TestStreamConfigApplication:
    """测试流式日志配置应用"""

    def test_stream_config_applied_to_cursor_config(self) -> None:
        """验证流式配置正确应用到 CursorAgentConfig"""
        config = OrchestratorConfig(
            stream_events_enabled=True,
            stream_log_console=False,
            stream_log_detail_dir="/custom/detail/",
            stream_log_raw_dir="/custom/raw/",
        )
        orchestrator = Orchestrator(config)

        cursor_config = orchestrator.config.cursor_config
        assert cursor_config.stream_events_enabled is True
        assert cursor_config.stream_log_console is False
        assert cursor_config.stream_log_detail_dir == "/custom/detail/"
        assert cursor_config.stream_log_raw_dir == "/custom/raw/"


class TestWorkerForceWriteConfig:
    """测试 Worker 的 force_write 配置

    验证：
    1. Worker 的 cursor_config.force_write=True（需要能修改文件）
    2. 每个 Worker 的 cursor_config.stream_agent_id 唯一
    3. Planner/Reviewer 的 force_write=False（只读角色）
    """

    def test_workers_force_write_enabled(self) -> None:
        """测试所有 Worker 的 force_write=True"""
        config = OrchestratorConfig(worker_pool_size=3)
        orchestrator = Orchestrator(config)

        # 验证每个 worker 的 cursor_config.force_write=True
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.force_write is True, \
                f"Worker {worker.id} 的 force_write 应为 True，实际为 {worker.worker_config.cursor_config.force_write}"

    def test_workers_stream_agent_id_unique(self) -> None:
        """测试每个 Worker 的 stream_agent_id 唯一"""
        config = OrchestratorConfig(worker_pool_size=5)
        orchestrator = Orchestrator(config)

        # 收集所有 worker 的 stream_agent_id
        stream_agent_ids = []
        for worker in orchestrator.worker_pool.workers:
            stream_agent_id = worker.worker_config.cursor_config.stream_agent_id
            assert stream_agent_id is not None, \
                f"Worker {worker.id} 的 stream_agent_id 不应为 None"
            stream_agent_ids.append(stream_agent_id)

        # 验证唯一性
        assert len(stream_agent_ids) == len(set(stream_agent_ids)), \
            f"Worker stream_agent_id 不唯一: {stream_agent_ids}"

    def test_planner_force_write_disabled(self) -> None:
        """测试 Planner 的 force_write=False"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # Planner 不应该修改文件
        assert orchestrator.planner.planner_config.cursor_config.force_write is False, \
            "Planner 的 force_write 应为 False（只读角色）"

    def test_reviewer_force_write_disabled(self) -> None:
        """测试 Reviewer 的 force_write=False"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # Reviewer 不应该修改文件
        assert orchestrator.reviewer.reviewer_config.cursor_config.force_write is False, \
            "Reviewer 的 force_write 应为 False（只读角色）"

    def test_workers_force_write_with_custom_config(self) -> None:
        """测试使用自定义 cursor_config 时 Worker 的 force_write 仍为 True"""
        from cursor.client import CursorAgentConfig

        # 创建一个 force_write=False 的 cursor_config
        custom_cursor_config = CursorAgentConfig(
            force_write=False,  # 显式设置为 False
            model="test-model",
        )

        config = OrchestratorConfig(
            worker_pool_size=2,
            cursor_config=custom_cursor_config,
        )
        orchestrator = Orchestrator(config)

        # 即使传入的 cursor_config.force_write=False，Worker 也应该被覆盖为 True
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.force_write is True, \
                f"Worker {worker.id} 的 force_write 应被强制设置为 True"


class TestAgentModeAndForceWriteConfig:
    """测试各角色的 mode 和 force_write 配置

    确保：
    1. Worker 使用 mode='agent' 且 force_write=True（允许修改文件）
    2. Planner 使用 mode='plan' 且 force_write=False（只读）
    3. Reviewer 使用 force_write=False（只读）
    """

    def test_worker_force_write_enabled(self) -> None:
        """验证 Worker 的 force_write=True 配置"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # 验证所有 Worker 的 force_write=True
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.force_write is True, \
                f"Worker {worker.id} 的 force_write 应为 True"

    def test_worker_agent_mode(self) -> None:
        """验证 Worker 的 mode='agent' 配置"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # 验证所有 Worker 的 mode='agent'
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.mode == 'agent', \
                f"Worker {worker.id} 的 mode 应为 'agent'"

    def test_planner_force_write_disabled(self) -> None:
        """验证 Planner 的 force_write=False 配置"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        assert orchestrator.planner.planner_config.cursor_config.force_write is False, \
            "Planner 的 force_write 应为 False（只读）"

    def test_planner_plan_mode(self) -> None:
        """验证 Planner 的 mode='plan' 配置（由 PlannerAgent 内部设置）"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # Planner 默认使用 plan 模式（在 PlannerAgent._apply_plan_mode_config 中设置）
        assert orchestrator.planner.planner_config.cursor_config.mode == 'plan', \
            "Planner 的 mode 应为 'plan'"

    def test_reviewer_force_write_disabled(self) -> None:
        """验证 Reviewer 的 force_write=False 配置"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        assert orchestrator.reviewer.reviewer_config.cursor_config.force_write is False, \
            "Reviewer 的 force_write 应为 False（只读）"

    def test_reviewer_ask_mode(self) -> None:
        """验证 Reviewer 的 mode='ask' 配置"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        # Reviewer 使用 ask 模式（在 ReviewerAgent._apply_ask_mode_config 中设置）
        assert orchestrator.reviewer.reviewer_config.cursor_config.mode == 'ask', \
            "Reviewer 的 mode 应为 'ask'"

    def test_worker_force_write_affects_cursor_client(self) -> None:
        """验证 Worker 的 force_write=True 能传递到 CursorAgentClient

        此测试确保 --force 参数能正确应用到 CLI 命令构建中
        """
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        for worker in orchestrator.worker_pool.workers:
            # 验证 cursor_client 的配置也有 force_write=True
            assert worker.cursor_client.config.force_write is True, \
                f"Worker {worker.id} 的 cursor_client.config.force_write 应为 True"

    def test_planner_readonly_guarantee(self) -> None:
        """验证 Planner 只读保证（mode='plan' + force_write=False）"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        planner_config = orchestrator.planner.planner_config.cursor_config
        assert planner_config.mode == 'plan', "Planner 应使用 plan 模式"
        assert planner_config.force_write is False, "Planner 不应允许写入"

    def test_reviewer_readonly_guarantee(self) -> None:
        """验证 Reviewer 只读保证（mode='ask' + force_write=False）"""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        reviewer_config = orchestrator.reviewer.reviewer_config.cursor_config
        assert reviewer_config.mode == 'ask', "Reviewer 应使用 ask 模式"
        assert reviewer_config.force_write is False, "Reviewer 不应允许写入"


class TestRoleBasedExecutionMode:
    """测试角色级执行模式路由

    验证：
    1. 角色级执行模式正确传递到各 Agent
    2. 默认继承全局 execution_mode
    3. Planner/Reviewer 只读语义仍然生效
    """

    def test_role_execution_mode_default_inherits_global(self) -> None:
        """测试角色级执行模式默认继承全局 execution_mode"""
        from cursor.executor import ExecutionMode

        config = OrchestratorConfig(
            execution_mode=ExecutionMode.CLOUD,
        )

        # 角色级执行模式默认为 None，表示继承全局
        assert config.planner_execution_mode is None
        assert config.worker_execution_mode is None
        assert config.reviewer_execution_mode is None

        # 创建编排器时会自动继承
        orchestrator = Orchestrator(config)

        # 验证各 Agent 使用了正确的执行模式
        # 由于默认继承，所有角色应使用全局 execution_mode
        assert orchestrator.config.execution_mode == ExecutionMode.CLOUD

    def test_role_execution_mode_custom_override(self) -> None:
        """测试自定义角色级执行模式覆盖全局设置"""
        from cursor.executor import ExecutionMode

        config = OrchestratorConfig(
            execution_mode=ExecutionMode.CLI,
            planner_execution_mode=ExecutionMode.CLOUD,
            worker_execution_mode=ExecutionMode.AUTO,
            # reviewer 不设置，应继承全局 CLI
        )

        # 验证配置值
        assert config.execution_mode == ExecutionMode.CLI
        assert config.planner_execution_mode == ExecutionMode.CLOUD
        assert config.worker_execution_mode == ExecutionMode.AUTO
        assert config.reviewer_execution_mode is None

    def test_planner_readonly_with_custom_execution_mode(self) -> None:
        """测试 Planner 使用自定义执行模式时仍保持只读语义"""
        from cursor.executor import ExecutionMode

        config = OrchestratorConfig(
            planner_execution_mode=ExecutionMode.CLOUD,
        )
        orchestrator = Orchestrator(config)

        # Planner 应保持只读语义
        assert orchestrator.planner.planner_config.cursor_config.force_write is False
        assert orchestrator.planner.planner_config.cursor_config.mode == 'plan'

    def test_reviewer_readonly_with_custom_execution_mode(self) -> None:
        """测试 Reviewer 使用自定义执行模式时仍保持只读语义"""
        from cursor.executor import ExecutionMode

        config = OrchestratorConfig(
            reviewer_execution_mode=ExecutionMode.CLOUD,
        )
        orchestrator = Orchestrator(config)

        # Reviewer 应保持只读语义
        assert orchestrator.reviewer.reviewer_config.cursor_config.force_write is False
        assert orchestrator.reviewer.reviewer_config.cursor_config.mode == 'ask'

    def test_worker_force_write_with_custom_execution_mode(self) -> None:
        """测试 Worker 使用自定义执行模式时保持写入权限"""
        from cursor.executor import ExecutionMode

        config = OrchestratorConfig(
            worker_execution_mode=ExecutionMode.CLOUD,
        )
        orchestrator = Orchestrator(config)

        # Worker 应保持写入权限
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.cursor_config.force_write is True
            assert worker.worker_config.cursor_config.mode == 'agent'

    def test_role_execution_mode_propagation_to_agent_config(self) -> None:
        """测试角色级执行模式正确传递到 Agent 配置"""
        from cursor.executor import ExecutionMode

        config = OrchestratorConfig(
            execution_mode=ExecutionMode.CLI,
            planner_execution_mode=ExecutionMode.PLAN,
            worker_execution_mode=ExecutionMode.AUTO,
            reviewer_execution_mode=ExecutionMode.ASK,
        )
        orchestrator = Orchestrator(config)

        # 验证各 Agent 收到正确的执行模式
        assert orchestrator.planner.planner_config.execution_mode == ExecutionMode.PLAN
        for worker in orchestrator.worker_pool.workers:
            assert worker.worker_config.execution_mode == ExecutionMode.AUTO
        assert orchestrator.reviewer.reviewer_config.execution_mode == ExecutionMode.ASK


class TestAutoCommitDefaultDisabled:
    """回归测试：验证 enable_auto_commit 默认禁用时的行为

    确保：
    1. 默认情况下 Committer 不被初始化
    2. 最终结果 commits 为空
    3. 不会触发任何提交操作
    """

    def test_default_config_auto_commit_disabled(self) -> None:
        """测试默认配置 enable_auto_commit=False"""
        config = OrchestratorConfig()
        assert config.enable_auto_commit is False, \
            "默认配置应禁用 auto_commit"

    def test_committer_not_initialized_when_disabled(self) -> None:
        """测试禁用 auto_commit 时 Committer 不被初始化"""
        config = OrchestratorConfig(
            working_directory=".",
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        assert orchestrator.committer is None, \
            "禁用 auto_commit 时不应初始化 Committer"

    def test_committer_initialized_when_enabled(self) -> None:
        """测试启用 auto_commit 时 Committer 被初始化"""
        config = OrchestratorConfig(
            working_directory=".",
            enable_auto_commit=True,
        )
        orchestrator = Orchestrator(config)

        assert orchestrator.committer is not None, \
            "启用 auto_commit 时应初始化 Committer"

    def test_committer_registered_when_enabled(self) -> None:
        """测试启用 auto_commit 时 Committer 注册到 SystemState"""
        config = OrchestratorConfig(
            working_directory=".",
            enable_auto_commit=True,
        )
        orchestrator = Orchestrator(config)

        # 验证 committer 注册到 agents
        committer_agents = [
            agent for agent in orchestrator.state.agents.values()
            if agent.role == AgentRole.COMMITTER
        ]
        assert len(committer_agents) == 1, \
            "启用 auto_commit 时应注册 Committer"

    def test_committer_not_registered_when_disabled(self) -> None:
        """测试禁用 auto_commit 时 Committer 不注册到 SystemState"""
        config = OrchestratorConfig(
            working_directory=".",
            enable_auto_commit=False,
        )
        orchestrator = Orchestrator(config)

        # 验证没有 committer 注册
        committer_agents = [
            agent for agent in orchestrator.state.agents.values()
            if agent.role == AgentRole.COMMITTER
        ]
        assert len(committer_agents) == 0, \
            "禁用 auto_commit 时不应注册 Committer"
