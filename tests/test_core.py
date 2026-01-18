"""core 模块单元测试

测试覆盖:
- core/base.py: AgentRole, AgentStatus, AgentConfig, BaseAgent
- core/message.py: MessageType, Message
- core/state.py: IterationStatus, AgentState, IterationState, SystemState
"""

import uuid
from datetime import datetime
from typing import Any, Optional

import pytest

from core.base import AgentConfig, AgentRole, AgentStatus, BaseAgent
from core.message import Message, MessageType
from core.state import AgentState, IterationState, IterationStatus, SystemState


# ============================================================================
# BaseAgent 的具体实现（用于测试）
# ============================================================================


class ConcreteAgent(BaseAgent):
    """用于测试的具体 Agent 实现"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.execute_called = False
        self.reset_called = False

    async def execute(
        self, instruction: str, context: Optional[dict] = None
    ) -> dict[str, Any]:
        self.execute_called = True
        self._context["last_instruction"] = instruction
        return {"status": "success", "instruction": instruction}

    async def reset(self) -> None:
        self.reset_called = True
        self.clear_context()
        self.status = AgentStatus.IDLE


# ============================================================================
# AgentRole 测试
# ============================================================================


class TestAgentRole:
    """AgentRole 枚举测试"""

    def test_role_values(self):
        """测试角色枚举值"""
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.SUB_PLANNER.value == "sub_planner"
        assert AgentRole.WORKER.value == "worker"
        assert AgentRole.REVIEWER.value == "reviewer"
        assert AgentRole.COMMITTER.value == "committer"

    def test_role_is_string_enum(self):
        """测试角色是字符串枚举"""
        assert isinstance(AgentRole.PLANNER, str)
        assert AgentRole.PLANNER == "planner"

    def test_all_roles_defined(self):
        """测试所有角色已定义"""
        expected_roles = {"planner", "sub_planner", "worker", "reviewer", "committer"}
        actual_roles = {role.value for role in AgentRole}
        assert actual_roles == expected_roles


# ============================================================================
# AgentStatus 测试
# ============================================================================


class TestAgentStatus:
    """AgentStatus 枚举测试"""

    def test_status_values(self):
        """测试状态枚举值"""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"

    def test_status_is_string_enum(self):
        """测试状态是字符串枚举"""
        assert isinstance(AgentStatus.IDLE, str)
        assert AgentStatus.IDLE == "idle"

    def test_all_statuses_defined(self):
        """测试所有状态已定义"""
        expected_statuses = {"idle", "running", "waiting", "completed", "failed"}
        actual_statuses = {status.value for status in AgentStatus}
        assert actual_statuses == expected_statuses


# ============================================================================
# AgentConfig 测试
# ============================================================================


class TestAgentConfig:
    """AgentConfig 配置模型测试"""

    def test_minimal_config(self):
        """测试最小配置"""
        config = AgentConfig(role=AgentRole.WORKER)
        assert config.role == AgentRole.WORKER
        assert config.name.startswith("agent-")
        assert config.max_retries == 3
        assert config.timeout == 300
        assert config.working_directory == "."
        assert config.cursor_agent_mode is True

    def test_full_config(self):
        """测试完整配置"""
        config = AgentConfig(
            role=AgentRole.PLANNER,
            name="my-planner",
            max_retries=5,
            timeout=600,
            working_directory="/tmp",
            cursor_agent_mode=False,
        )
        assert config.role == AgentRole.PLANNER
        assert config.name == "my-planner"
        assert config.max_retries == 5
        assert config.timeout == 600
        assert config.working_directory == "/tmp"
        assert config.cursor_agent_mode is False

    def test_auto_generated_name_is_unique(self):
        """测试自动生成的名称唯一"""
        config1 = AgentConfig(role=AgentRole.WORKER)
        config2 = AgentConfig(role=AgentRole.WORKER)
        assert config1.name != config2.name

    def test_config_with_all_roles(self):
        """测试所有角色的配置"""
        for role in AgentRole:
            config = AgentConfig(role=role)
            assert config.role == role


# ============================================================================
# BaseAgent 测试
# ============================================================================


class TestBaseAgent:
    """BaseAgent 基类测试"""

    @pytest.fixture
    def agent(self) -> ConcreteAgent:
        """创建测试 Agent"""
        config = AgentConfig(role=AgentRole.WORKER, name="test-worker")
        return ConcreteAgent(config)

    def test_agent_initialization(self, agent: ConcreteAgent):
        """测试 Agent 初始化"""
        assert agent.role == AgentRole.WORKER
        assert agent.name == "test-worker"
        assert agent.status == AgentStatus.IDLE
        assert agent.id.startswith("worker-")

    def test_agent_id_format(self, agent: ConcreteAgent):
        """测试 Agent ID 格式"""
        # ID 格式: {role}-{uuid_hex[:8]}
        parts = agent.id.split("-")
        assert parts[0] == "worker"
        assert len(parts[1]) == 8

    def test_update_status(self, agent: ConcreteAgent):
        """测试状态更新"""
        agent.update_status(AgentStatus.RUNNING)
        assert agent.status == AgentStatus.RUNNING

        agent.update_status(AgentStatus.COMPLETED)
        assert agent.status == AgentStatus.COMPLETED

    def test_update_status_same_value(self, agent: ConcreteAgent):
        """测试更新相同状态（边界条件）"""
        agent.update_status(AgentStatus.IDLE)
        assert agent.status == AgentStatus.IDLE

    def test_context_operations(self, agent: ConcreteAgent):
        """测试上下文操作"""
        # 设置上下文
        agent.set_context("key1", "value1")
        agent.set_context("key2", {"nested": "data"})

        # 获取上下文
        assert agent.get_context("key1") == "value1"
        assert agent.get_context("key2") == {"nested": "data"}

        # 获取不存在的键
        assert agent.get_context("nonexistent") is None
        assert agent.get_context("nonexistent", "default") == "default"

        # 清除上下文
        agent.clear_context()
        assert agent.get_context("key1") is None

    def test_context_overwrite(self, agent: ConcreteAgent):
        """测试上下文覆盖"""
        agent.set_context("key", "old_value")
        agent.set_context("key", "new_value")
        assert agent.get_context("key") == "new_value"

    def test_repr(self, agent: ConcreteAgent):
        """测试字符串表示"""
        repr_str = repr(agent)
        assert "ConcreteAgent" in repr_str
        assert agent.id in repr_str
        assert "idle" in repr_str

    @pytest.mark.asyncio
    async def test_execute(self, agent: ConcreteAgent):
        """测试执行方法"""
        result = await agent.execute("test instruction")
        assert agent.execute_called is True
        assert result["status"] == "success"
        assert result["instruction"] == "test instruction"

    @pytest.mark.asyncio
    async def test_reset(self, agent: ConcreteAgent):
        """测试重置方法"""
        agent.set_context("key", "value")
        agent.update_status(AgentStatus.RUNNING)

        await agent.reset()

        assert agent.reset_called is True
        assert agent.status == AgentStatus.IDLE
        assert agent.get_context("key") is None


# ============================================================================
# MessageType 测试
# ============================================================================


class TestMessageType:
    """MessageType 枚举测试"""

    def test_task_message_types(self):
        """测试任务相关消息类型"""
        assert MessageType.TASK_CREATE.value == "task_create"
        assert MessageType.TASK_ASSIGN.value == "task_assign"
        assert MessageType.TASK_COMPLETE.value == "task_complete"
        assert MessageType.TASK_FAILED.value == "task_failed"

    def test_plan_message_types(self):
        """测试规划相关消息类型"""
        assert MessageType.PLAN_REQUEST.value == "plan_request"
        assert MessageType.PLAN_RESULT.value == "plan_result"
        assert MessageType.SPAWN_PLANNER.value == "spawn_planner"

    def test_execute_message_types(self):
        """测试执行相关消息类型"""
        assert MessageType.EXECUTE_REQUEST.value == "execute_request"
        assert MessageType.EXECUTE_RESULT.value == "execute_result"

    def test_review_message_types(self):
        """测试评审相关消息类型"""
        assert MessageType.REVIEW_REQUEST.value == "review_request"
        assert MessageType.REVIEW_RESULT.value == "review_result"

    def test_system_message_types(self):
        """测试系统相关消息类型"""
        assert MessageType.SYSTEM_RESET.value == "system_reset"
        assert MessageType.ITERATION_START.value == "iteration_start"
        assert MessageType.ITERATION_END.value == "iteration_end"


# ============================================================================
# Message 测试
# ============================================================================


class TestMessage:
    """Message 消息模型测试"""

    def test_message_creation_minimal(self):
        """测试最小消息创建"""
        msg = Message(
            type=MessageType.TASK_CREATE,
            sender="planner-123",
        )
        assert msg.type == MessageType.TASK_CREATE
        assert msg.sender == "planner-123"
        assert msg.receiver is None
        assert msg.payload == {}
        assert msg.correlation_id is None
        assert isinstance(msg.id, str)
        assert isinstance(msg.timestamp, datetime)

    def test_message_creation_full(self):
        """测试完整消息创建"""
        payload = {"task_id": "task-001", "description": "Test task"}
        msg = Message(
            type=MessageType.TASK_ASSIGN,
            sender="planner-123",
            receiver="worker-456",
            payload=payload,
            correlation_id="corr-789",
        )
        assert msg.type == MessageType.TASK_ASSIGN
        assert msg.sender == "planner-123"
        assert msg.receiver == "worker-456"
        assert msg.payload == payload
        assert msg.correlation_id == "corr-789"

    def test_message_id_auto_generated(self):
        """测试消息 ID 自动生成"""
        msg1 = Message(type=MessageType.TASK_CREATE, sender="sender")
        msg2 = Message(type=MessageType.TASK_CREATE, sender="sender")
        assert msg1.id != msg2.id

    def test_message_timestamp_auto_generated(self):
        """测试时间戳自动生成"""
        before = datetime.now()
        msg = Message(type=MessageType.TASK_CREATE, sender="sender")
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_create_reply(self):
        """测试创建回复消息"""
        original = Message(
            type=MessageType.TASK_ASSIGN,
            sender="planner-123",
            receiver="worker-456",
            payload={"task_id": "task-001"},
        )

        reply = original.create_reply(
            MessageType.TASK_COMPLETE,
            {"result": "success"},
        )

        assert reply.type == MessageType.TASK_COMPLETE
        assert reply.sender == "worker-456"  # 原消息的 receiver
        assert reply.receiver == "planner-123"  # 原消息的 sender
        assert reply.payload == {"result": "success"}
        assert reply.correlation_id == original.id

    def test_create_reply_broadcast_message(self):
        """测试广播消息的回复（边界条件）"""
        original = Message(
            type=MessageType.SYSTEM_RESET,
            sender="system",
            receiver=None,  # 广播消息
        )

        reply = original.create_reply(
            MessageType.TASK_COMPLETE,
            {"status": "acknowledged"},
        )

        assert reply.sender == ""  # receiver 为 None 时返回空字符串
        assert reply.receiver == "system"

    def test_timestamp_serialization(self):
        """测试时间戳序列化"""
        msg = Message(type=MessageType.TASK_CREATE, sender="sender")
        data = msg.model_dump()
        # timestamp 应该被序列化为 ISO 格式字符串
        assert isinstance(data["timestamp"], str)
        # 验证可以解析回 datetime
        datetime.fromisoformat(data["timestamp"])


# ============================================================================
# IterationStatus 测试
# ============================================================================


class TestIterationStatus:
    """IterationStatus 枚举测试"""

    def test_iteration_status_values(self):
        """测试迭代状态枚举值"""
        assert IterationStatus.PLANNING.value == "planning"
        assert IterationStatus.EXECUTING.value == "executing"
        assert IterationStatus.REVIEWING.value == "reviewing"
        assert IterationStatus.COMMITTING.value == "committing"
        assert IterationStatus.COMPLETED.value == "completed"
        assert IterationStatus.FAILED.value == "failed"


# ============================================================================
# AgentState 测试
# ============================================================================


class TestAgentState:
    """AgentState 状态模型测试"""

    def test_agent_state_creation_minimal(self):
        """测试最小状态创建"""
        state = AgentState(
            agent_id="worker-123",
            role=AgentRole.WORKER,
        )
        assert state.agent_id == "worker-123"
        assert state.role == AgentRole.WORKER
        assert state.status == AgentStatus.IDLE
        assert state.current_task_id is None
        assert state.completed_tasks == []
        assert state.error_count == 0
        assert isinstance(state.last_activity, datetime)

    def test_agent_state_creation_full(self):
        """测试完整状态创建"""
        state = AgentState(
            agent_id="planner-456",
            role=AgentRole.PLANNER,
            status=AgentStatus.RUNNING,
            current_task_id="task-001",
            completed_tasks=["task-000"],
            error_count=2,
        )
        assert state.agent_id == "planner-456"
        assert state.role == AgentRole.PLANNER
        assert state.status == AgentStatus.RUNNING
        assert state.current_task_id == "task-001"
        assert state.completed_tasks == ["task-000"]
        assert state.error_count == 2

    def test_agent_state_to_dict(self):
        """测试状态转换为字典"""
        state = AgentState(
            agent_id="worker-123",
            role=AgentRole.WORKER,
        )
        data = state.to_dict()
        assert data["agent_id"] == "worker-123"
        assert data["role"] == AgentRole.WORKER
        assert data["status"] == AgentStatus.IDLE


# ============================================================================
# IterationState 测试
# ============================================================================


class TestIterationState:
    """IterationState 状态模型测试"""

    def test_iteration_state_creation_minimal(self):
        """测试最小迭代状态创建"""
        state = IterationState(iteration_id=1)
        assert state.iteration_id == 1
        assert state.status == IterationStatus.PLANNING
        assert isinstance(state.started_at, datetime)
        assert state.completed_at is None
        assert state.tasks_created == 0
        assert state.tasks_completed == 0
        assert state.tasks_failed == 0
        assert state.review_passed is False
        assert state.review_feedback is None
        assert state.commit_hash is None
        assert state.commit_message is None
        assert state.pushed is False
        assert state.commit_files == []

    def test_iteration_state_creation_full(self):
        """测试完整迭代状态创建"""
        now = datetime.now()
        state = IterationState(
            iteration_id=5,
            status=IterationStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            tasks_created=10,
            tasks_completed=9,
            tasks_failed=1,
            review_passed=True,
            review_feedback="Good work",
            commit_hash="abc123",
            commit_message="feat: new feature",
            pushed=True,
            commit_files=["file1.py", "file2.py"],
        )
        assert state.iteration_id == 5
        assert state.status == IterationStatus.COMPLETED
        assert state.tasks_created == 10
        assert state.tasks_completed == 9
        assert state.tasks_failed == 1
        assert state.review_passed is True
        assert state.review_feedback == "Good work"
        assert state.commit_hash == "abc123"
        assert state.pushed is True
        assert len(state.commit_files) == 2


# ============================================================================
# SystemState 测试
# ============================================================================


class TestSystemState:
    """SystemState 系统状态模型测试"""

    @pytest.fixture
    def system_state(self) -> SystemState:
        """创建测试系统状态"""
        return SystemState(
            goal="Implement new feature",
            working_directory="/project",
            max_iterations=5,
        )

    def test_system_state_creation_default(self):
        """测试默认系统状态创建"""
        state = SystemState()
        assert state.goal == ""
        assert state.working_directory == "."
        assert state.current_iteration == 0
        assert state.max_iterations == 10
        assert state.iterations == []
        assert state.agents == {}
        assert state.total_tasks_created == 0
        assert state.total_tasks_completed == 0
        assert state.total_tasks_failed == 0
        assert state.is_running is False
        assert state.is_completed is False
        assert state.final_result is None

    def test_system_state_creation_custom(self, system_state: SystemState):
        """测试自定义系统状态创建"""
        assert system_state.goal == "Implement new feature"
        assert system_state.working_directory == "/project"
        assert system_state.max_iterations == 5

    def test_start_new_iteration(self, system_state: SystemState):
        """测试开始新迭代"""
        assert system_state.current_iteration == 0
        assert len(system_state.iterations) == 0

        iteration = system_state.start_new_iteration()

        assert system_state.current_iteration == 1
        assert len(system_state.iterations) == 1
        assert iteration.iteration_id == 1
        assert iteration.status == IterationStatus.PLANNING

    def test_start_multiple_iterations(self, system_state: SystemState):
        """测试开始多次迭代"""
        iter1 = system_state.start_new_iteration()
        iter2 = system_state.start_new_iteration()
        iter3 = system_state.start_new_iteration()

        assert system_state.current_iteration == 3
        assert len(system_state.iterations) == 3
        assert iter1.iteration_id == 1
        assert iter2.iteration_id == 2
        assert iter3.iteration_id == 3

    def test_get_current_iteration(self, system_state: SystemState):
        """测试获取当前迭代"""
        # 无迭代时返回 None
        assert system_state.get_current_iteration() is None

        # 有迭代时返回最新的
        system_state.start_new_iteration()
        system_state.start_new_iteration()

        current = system_state.get_current_iteration()
        assert current is not None
        assert current.iteration_id == 2

    def test_register_agent(self, system_state: SystemState):
        """测试注册 Agent"""
        agent_state = system_state.register_agent("worker-123", AgentRole.WORKER)

        assert agent_state.agent_id == "worker-123"
        assert agent_state.role == AgentRole.WORKER
        assert agent_state.status == AgentStatus.IDLE
        assert "worker-123" in system_state.agents

    def test_register_multiple_agents(self, system_state: SystemState):
        """测试注册多个 Agent"""
        system_state.register_agent("planner-001", AgentRole.PLANNER)
        system_state.register_agent("worker-001", AgentRole.WORKER)
        system_state.register_agent("reviewer-001", AgentRole.REVIEWER)

        assert len(system_state.agents) == 3
        assert "planner-001" in system_state.agents
        assert "worker-001" in system_state.agents
        assert "reviewer-001" in system_state.agents

    def test_update_agent_status(self, system_state: SystemState):
        """测试更新 Agent 状态"""
        system_state.register_agent("worker-123", AgentRole.WORKER)

        system_state.update_agent_status("worker-123", AgentStatus.RUNNING)

        assert system_state.agents["worker-123"].status == AgentStatus.RUNNING

    def test_update_agent_status_updates_last_activity(self, system_state: SystemState):
        """测试更新状态时更新最后活动时间"""
        system_state.register_agent("worker-123", AgentRole.WORKER)
        original_time = system_state.agents["worker-123"].last_activity

        # 等待一小段时间确保时间不同
        import time

        time.sleep(0.01)

        system_state.update_agent_status("worker-123", AgentStatus.RUNNING)

        assert system_state.agents["worker-123"].last_activity >= original_time

    def test_update_agent_status_nonexistent(self, system_state: SystemState):
        """测试更新不存在的 Agent 状态（边界条件）"""
        # 不应该抛出异常
        system_state.update_agent_status("nonexistent-agent", AgentStatus.RUNNING)
        assert "nonexistent-agent" not in system_state.agents

    def test_reset_for_new_iteration(self, system_state: SystemState):
        """测试为新迭代重置状态"""
        # 注册 Agent 并设置状态
        system_state.register_agent("worker-123", AgentRole.WORKER)
        system_state.agents["worker-123"].status = AgentStatus.RUNNING
        system_state.agents["worker-123"].current_task_id = "task-001"
        system_state.agents["worker-123"].error_count = 3

        # 重置
        system_state.reset_for_new_iteration()

        # 验证状态被重置
        agent = system_state.agents["worker-123"]
        assert agent.status == AgentStatus.IDLE
        assert agent.current_task_id is None
        assert agent.error_count == 0

    def test_reset_for_new_iteration_preserves_history(self, system_state: SystemState):
        """测试重置保留历史记录"""
        system_state.register_agent("worker-123", AgentRole.WORKER)
        system_state.agents["worker-123"].completed_tasks = ["task-001", "task-002"]

        system_state.reset_for_new_iteration()

        # 历史任务记录应该保留
        assert system_state.agents["worker-123"].completed_tasks == [
            "task-001",
            "task-002",
        ]


# ============================================================================
# 集成测试
# ============================================================================


class TestCoreIntegration:
    """核心模块集成测试"""

    def test_agent_state_tracking_in_system(self):
        """测试在系统中追踪 Agent 状态"""
        system = SystemState(goal="Test integration")

        # 创建 Agent 配置和实例
        config = AgentConfig(role=AgentRole.WORKER, name="test-worker")
        agent = ConcreteAgent(config)

        # 在系统中注册
        system.register_agent(agent.id, agent.role)

        # 同步状态
        agent.update_status(AgentStatus.RUNNING)
        system.update_agent_status(agent.id, agent.status)

        assert system.agents[agent.id].status == AgentStatus.RUNNING

    def test_message_flow_simulation(self):
        """测试消息流模拟"""
        # 模拟 Planner -> Worker -> Planner 的消息流
        task_assign = Message(
            type=MessageType.TASK_ASSIGN,
            sender="planner-001",
            receiver="worker-001",
            payload={"task_id": "task-001", "instruction": "Implement feature"},
        )

        # Worker 完成任务后创建回复
        task_complete = task_assign.create_reply(
            MessageType.TASK_COMPLETE,
            {"task_id": "task-001", "result": "Feature implemented"},
        )

        assert task_complete.sender == "worker-001"
        assert task_complete.receiver == "planner-001"
        assert task_complete.correlation_id == task_assign.id

    def test_iteration_lifecycle(self):
        """测试迭代生命周期"""
        system = SystemState(goal="Complete feature", max_iterations=3)

        # 开始迭代
        iteration = system.start_new_iteration()
        assert iteration.status == IterationStatus.PLANNING

        # 执行阶段
        iteration.status = IterationStatus.EXECUTING
        iteration.tasks_created = 5

        # 评审阶段
        iteration.status = IterationStatus.REVIEWING
        iteration.tasks_completed = 4
        iteration.tasks_failed = 1

        # 完成
        iteration.status = IterationStatus.COMPLETED
        iteration.review_passed = True
        iteration.completed_at = datetime.now()

        assert system.current_iteration == 1
        current = system.get_current_iteration()
        assert current is not None
        assert current.review_passed is True
        assert current.completed_at is not None
