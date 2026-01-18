"""测试编排器（Orchestrator）初始化和配置"""
from unittest.mock import MagicMock

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
        assert config.enable_auto_commit is True  # 默认启用自动提交
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
        """验证总 Agent 数量正确（1 planner + 1 reviewer + N workers + 1 committer）"""
        worker_count = 4
        config = OrchestratorConfig(worker_pool_size=worker_count)
        orchestrator = Orchestrator(config)

        # enable_auto_commit 默认为 True，所以会创建 committer
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
