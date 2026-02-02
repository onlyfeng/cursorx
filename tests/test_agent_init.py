"""测试 Agent 类初始化和配置验证

验证各 Agent 类能正确初始化，检查配置类的默认值和必填项
"""

import pytest

from agents.planner import PlannerAgent, PlannerConfig
from agents.reviewer import ReviewDecision, ReviewerAgent, ReviewerConfig
from agents.worker import WorkerAgent, WorkerConfig
from core.base import AgentRole, AgentStatus
from cursor.client import CursorAgentConfig


class TestPlannerConfig:
    """测试 PlannerConfig 配置类"""

    def test_default_values(self):
        """测试默认值是否正确"""
        config = PlannerConfig()

        # 验证默认值
        assert config.name == "planner"
        assert config.working_directory == "."
        assert config.max_sub_planners == 3
        assert config.max_tasks_per_plan == 10
        assert config.exploration_depth == 3
        assert config.enable_semantic_search is False
        assert config.semantic_search_top_k == 10
        assert config.semantic_search_min_score == 0.3
        # cursor_config 应该是 CursorAgentConfig 实例
        assert isinstance(config.cursor_config, CursorAgentConfig)

    def test_custom_values(self):
        """测试自定义值"""
        config = PlannerConfig(
            name="custom-planner",
            working_directory="/tmp/test",
            max_sub_planners=5,
            max_tasks_per_plan=20,
            exploration_depth=5,
            enable_semantic_search=True,
            semantic_search_top_k=15,
            semantic_search_min_score=0.5,
        )

        assert config.name == "custom-planner"
        assert config.working_directory == "/tmp/test"
        assert config.max_sub_planners == 5
        assert config.max_tasks_per_plan == 20
        assert config.exploration_depth == 5
        assert config.enable_semantic_search is True
        assert config.semantic_search_top_k == 15
        assert config.semantic_search_min_score == 0.5

    def test_no_required_fields(self):
        """测试无必填项 - 空配置应该能正常创建"""
        config = PlannerConfig()
        assert config is not None


class TestWorkerConfig:
    """测试 WorkerConfig 配置类"""

    def test_default_values(self):
        """测试默认值是否正确"""
        config = WorkerConfig()

        # 验证默认值
        assert config.name == "worker"
        assert config.working_directory == "."
        assert config.max_concurrent_tasks == 1
        assert config.task_timeout == 500
        assert config.enable_context_search is False
        assert config.context_search_top_k == 5
        assert config.context_search_min_score == 0.4
        assert config.enable_knowledge_search is True
        assert config.knowledge_search_top_k == 5
        # cursor_config 应该是 CursorAgentConfig 实例
        assert isinstance(config.cursor_config, CursorAgentConfig)

    def test_custom_values(self):
        """测试自定义值"""
        config = WorkerConfig(
            name="custom-worker",
            working_directory="/tmp/work",
            max_concurrent_tasks=3,
            task_timeout=600,
            enable_context_search=True,
            context_search_top_k=10,
            context_search_min_score=0.6,
            enable_knowledge_search=False,
            knowledge_search_top_k=10,
        )

        assert config.name == "custom-worker"
        assert config.working_directory == "/tmp/work"
        assert config.max_concurrent_tasks == 3
        assert config.task_timeout == 600
        assert config.enable_context_search is True
        assert config.context_search_top_k == 10
        assert config.context_search_min_score == 0.6
        assert config.enable_knowledge_search is False
        assert config.knowledge_search_top_k == 10

    def test_no_required_fields(self):
        """测试无必填项 - 空配置应该能正常创建"""
        config = WorkerConfig()
        assert config is not None


class TestReviewerConfig:
    """测试 ReviewerConfig 配置类"""

    def test_default_values(self):
        """测试默认值是否正确"""
        config = ReviewerConfig()

        # 验证默认值
        assert config.name == "reviewer"
        assert config.working_directory == "."
        assert config.strict_mode is False
        # cursor_config 应该是 CursorAgentConfig 实例
        assert isinstance(config.cursor_config, CursorAgentConfig)

    def test_custom_values(self):
        """测试自定义值"""
        config = ReviewerConfig(
            name="strict-reviewer",
            working_directory="/tmp/review",
            strict_mode=True,
        )

        assert config.name == "strict-reviewer"
        assert config.working_directory == "/tmp/review"
        assert config.strict_mode is True

    def test_no_required_fields(self):
        """测试无必填项 - 空配置应该能正常创建"""
        config = ReviewerConfig()
        assert config is not None


class TestPlannerAgentInit:
    """测试 PlannerAgent 初始化"""

    def test_default_init(self):
        """测试使用默认配置初始化"""
        config = PlannerConfig()
        agent = PlannerAgent(config)

        # 验证 Agent 基本属性
        assert agent.role == AgentRole.PLANNER
        assert agent.name == "planner"
        assert agent.status == AgentStatus.IDLE
        assert agent.planner_config == config
        assert agent.sub_planners == []
        # 语义搜索默认禁用
        assert agent._search_enabled is False
        assert agent._semantic_search is None

    def test_custom_name_init(self):
        """测试使用自定义名称初始化"""
        config = PlannerConfig(name="my-planner")
        agent = PlannerAgent(config)

        assert agent.name == "my-planner"
        assert "planner" in agent.id  # ID 包含角色标识

    def test_with_semantic_search_enabled(self):
        """测试启用语义搜索时（无搜索实例）"""
        config = PlannerConfig(enable_semantic_search=True)
        agent = PlannerAgent(config)

        # 配置启用但没有搜索实例，实际搜索应该禁用
        assert config.enable_semantic_search is True
        assert agent._search_enabled is False  # 因为没有传入 semantic_search 实例

    def test_id_format(self):
        """测试 Agent ID 格式"""
        config = PlannerConfig()
        agent = PlannerAgent(config)

        # ID 格式：{role}-{random_hex}
        assert agent.id.startswith("planner-")
        assert len(agent.id) > len("planner-")


class TestWorkerAgentInit:
    """测试 WorkerAgent 初始化"""

    def test_default_init(self):
        """测试使用默认配置初始化"""
        config = WorkerConfig()
        agent = WorkerAgent(config)

        # 验证 Agent 基本属性
        assert agent.role == AgentRole.WORKER
        assert agent.name == "worker"
        assert agent.status == AgentStatus.IDLE
        assert agent.worker_config == config
        assert agent.current_task is None
        assert agent.completed_tasks == []
        # 搜索默认禁用
        assert agent._search_enabled is False
        assert agent._knowledge_search_enabled is False

    def test_custom_name_init(self):
        """测试使用自定义名称初始化"""
        config = WorkerConfig(name="my-worker")
        agent = WorkerAgent(config)

        assert agent.name == "my-worker"
        assert "worker" in agent.id

    def test_with_context_search_enabled(self):
        """测试启用上下文搜索时（无搜索实例）"""
        config = WorkerConfig(enable_context_search=True)
        agent = WorkerAgent(config)

        # 配置启用但没有搜索实例，实际搜索应该禁用
        assert config.enable_context_search is True
        assert agent._search_enabled is False

    def test_with_knowledge_search_enabled(self):
        """测试启用知识库搜索时（无管理器实例）"""
        config = WorkerConfig(enable_knowledge_search=True)
        agent = WorkerAgent(config)

        # 配置启用但没有管理器实例，实际搜索应该禁用
        assert config.enable_knowledge_search is True
        assert agent._knowledge_search_enabled is False

    def test_get_statistics(self):
        """测试统计信息获取"""
        config = WorkerConfig()
        agent = WorkerAgent(config)

        stats = agent.get_statistics()
        assert "worker_id" in stats
        assert stats["status"] == "idle"
        assert stats["completed_tasks_count"] == 0
        assert stats["current_task"] is None
        assert stats["context_search_enabled"] is False
        assert stats["knowledge_search_enabled"] is False


class TestReviewerAgentInit:
    """测试 ReviewerAgent 初始化"""

    def test_default_init(self):
        """测试使用默认配置初始化"""
        config = ReviewerConfig()
        agent = ReviewerAgent(config)

        # 验证 Agent 基本属性
        assert agent.role == AgentRole.REVIEWER
        assert agent.name == "reviewer"
        assert agent.status == AgentStatus.IDLE
        assert agent.reviewer_config == config
        assert agent.review_history == []

    def test_custom_name_init(self):
        """测试使用自定义名称初始化"""
        config = ReviewerConfig(name="my-reviewer")
        agent = ReviewerAgent(config)

        assert agent.name == "my-reviewer"
        assert "reviewer" in agent.id

    def test_strict_mode(self):
        """测试严格模式配置"""
        config = ReviewerConfig(strict_mode=True)
        agent = ReviewerAgent(config)

        assert agent.reviewer_config.strict_mode is True

    def test_get_review_summary_empty(self):
        """测试无评审历史时的摘要"""
        config = ReviewerConfig()
        agent = ReviewerAgent(config)

        summary = agent.get_review_summary()
        assert summary["total_reviews"] == 0

    def test_review_decision_enum(self):
        """测试 ReviewDecision 枚举值"""
        assert ReviewDecision.CONTINUE == "continue"
        assert ReviewDecision.COMPLETE == "complete"
        assert ReviewDecision.ADJUST == "adjust"
        assert ReviewDecision.ABORT == "abort"


class TestCursorAgentConfigDefaults:
    """测试 CursorAgentConfig 默认值（被各 Agent 配置引用）"""

    def test_default_values(self):
        """测试 CursorAgentConfig 默认值"""
        from core.config import DEFAULT_WORKER_MODEL

        config = CursorAgentConfig()

        assert config.agent_path == "agent"
        assert config.api_key is None
        assert config.working_directory == "."
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.model == DEFAULT_WORKER_MODEL
        assert config.output_format == "text"
        assert config.stream_partial_output is False
        assert config.non_interactive is True
        assert config.force_write is False
        assert config.resume_thread_id is None
        assert config.background is False
        assert config.fullscreen is False
        # 流式日志配置（默认关闭，与 DEFAULT_STREAM_EVENTS_ENABLED 同步）
        assert config.stream_events_enabled is False
        assert config.stream_log_console is True


class TestAgentReset:
    """测试 Agent 重置功能"""

    @pytest.mark.asyncio
    async def test_planner_reset(self):
        """测试 PlannerAgent 重置"""
        config = PlannerConfig()
        agent = PlannerAgent(config)

        # 设置一些上下文
        agent.set_context("test_key", "test_value")
        agent.update_status(AgentStatus.RUNNING)

        # 重置
        await agent.reset()

        assert agent.status == AgentStatus.IDLE
        assert agent.get_context("test_key") is None

    @pytest.mark.asyncio
    async def test_worker_reset(self):
        """测试 WorkerAgent 重置"""
        config = WorkerConfig()
        agent = WorkerAgent(config)

        # 设置一些上下文
        agent.set_context("test_key", "test_value")
        agent.update_status(AgentStatus.RUNNING)

        # 重置
        await agent.reset()

        assert agent.status == AgentStatus.IDLE
        assert agent.current_task is None
        assert agent.get_context("test_key") is None

    @pytest.mark.asyncio
    async def test_reviewer_reset(self):
        """测试 ReviewerAgent 重置"""
        config = ReviewerConfig()
        agent = ReviewerAgent(config)

        # 设置一些上下文
        agent.set_context("test_key", "test_value")
        agent.update_status(AgentStatus.RUNNING)

        # 重置
        await agent.reset()

        assert agent.status == AgentStatus.IDLE
        assert agent.get_context("test_key") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
