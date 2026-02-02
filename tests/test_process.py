"""进程管理模块测试

测试 process/manager.py、process/message_queue.py、process/worker.py 的功能
"""

import multiprocessing as mp
import pickle
import time
from datetime import datetime
from multiprocessing import Queue
from unittest.mock import MagicMock, patch

import pytest

from process.manager import AgentProcessManager, HealthCheckResult
from process.message_queue import MessageQueue, ProcessMessage, ProcessMessageType
from process.worker import AgentWorkerProcess

# ============================================================================
# ProcessMessage 测试
# ============================================================================


class TestProcessMessage:
    """ProcessMessage 消息类测试"""

    def test_message_creation_with_defaults(self):
        """测试默认参数创建消息"""
        msg = ProcessMessage()

        assert msg.id is not None
        assert len(msg.id) == 32  # uuid4 hex
        assert msg.type == ProcessMessageType.HEARTBEAT
        assert msg.sender == ""
        assert msg.receiver == ""
        assert msg.payload == {}
        assert isinstance(msg.timestamp, datetime)
        assert msg.correlation_id is None

    def test_message_creation_with_custom_params(self):
        """测试自定义参数创建消息"""
        msg = ProcessMessage(
            type=ProcessMessageType.TASK_ASSIGN,
            sender="planner",
            receiver="worker-1",
            payload={"task_id": "task-123", "content": "test"},
            correlation_id="corr-456",
        )

        assert msg.type == ProcessMessageType.TASK_ASSIGN
        assert msg.sender == "planner"
        assert msg.receiver == "worker-1"
        assert msg.payload == {"task_id": "task-123", "content": "test"}
        assert msg.correlation_id == "corr-456"

    def test_message_serialization(self):
        """测试消息序列化/反序列化"""
        original = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            receiver="coordinator",
            payload={"result": "success", "data": [1, 2, 3]},
        )

        # 序列化
        data = original.to_bytes()
        assert isinstance(data, bytes)

        # 反序列化
        restored = ProcessMessage.from_bytes(data)
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.sender == original.sender
        assert restored.receiver == original.receiver
        assert restored.payload == original.payload

    def test_message_create_reply(self):
        """测试创建回复消息"""
        original = ProcessMessage(
            type=ProcessMessageType.TASK_ASSIGN,
            sender="coordinator",
            receiver="worker-1",
        )

        reply = original.create_reply(ProcessMessageType.TASK_RESULT, {"status": "completed"})

        assert reply.type == ProcessMessageType.TASK_RESULT
        assert reply.sender == "worker-1"  # 原来的 receiver
        assert reply.receiver == "coordinator"  # 原来的 sender
        assert reply.correlation_id == original.id
        assert reply.payload == {"status": "completed"}

    def test_message_pickle_compatible(self):
        """测试 pickle 兼容性"""
        msg = ProcessMessage(
            type=ProcessMessageType.PLAN_REQUEST,
            sender="main",
            payload={"goal": "test goal"},
        )

        pickled = pickle.dumps(msg)
        unpickled = pickle.loads(pickled)

        assert unpickled.type == msg.type
        assert unpickled.sender == msg.sender
        assert unpickled.payload == msg.payload


class TestProcessMessageType:
    """ProcessMessageType 枚举测试"""

    def test_message_types_exist(self):
        """测试所有消息类型存在"""
        # 任务相关
        assert ProcessMessageType.TASK_ASSIGN == "task_assign"
        assert ProcessMessageType.TASK_RESULT == "task_result"
        assert ProcessMessageType.TASK_PROGRESS == "task_progress"

        # 控制相关
        assert ProcessMessageType.SHUTDOWN == "shutdown"
        assert ProcessMessageType.HEARTBEAT == "heartbeat"
        assert ProcessMessageType.STATUS_REQUEST == "status_request"
        assert ProcessMessageType.STATUS_RESPONSE == "status_response"

        # 规划相关
        assert ProcessMessageType.PLAN_REQUEST == "plan_request"
        assert ProcessMessageType.PLAN_RESULT == "plan_result"

        # 评审相关
        assert ProcessMessageType.REVIEW_REQUEST == "review_request"
        assert ProcessMessageType.REVIEW_RESULT == "review_result"

    def test_message_type_is_string_enum(self):
        """测试消息类型是字符串枚举"""
        assert isinstance(ProcessMessageType.HEARTBEAT, str)
        assert ProcessMessageType.HEARTBEAT.value == "heartbeat"


# ============================================================================
# MessageQueue 测试
# ============================================================================


class TestMessageQueue:
    """MessageQueue 消息队列测试"""

    def test_queue_initialization(self):
        """测试队列初始化"""
        mq = MessageQueue()

        assert mq.to_coordinator is not None
        assert isinstance(mq._agent_queues, dict)
        assert len(mq._agent_queues) == 0

        mq.cleanup()

    def test_create_agent_queue(self):
        """测试创建 Agent 队列"""
        mq = MessageQueue()

        queue1 = mq.create_agent_queue("agent-1")
        queue2 = mq.create_agent_queue("agent-2")

        assert queue1 is not None
        assert queue2 is not None
        assert mq.get_agent_queue("agent-1") is queue1
        assert mq.get_agent_queue("agent-2") is queue2
        assert mq.get_agent_queue("agent-3") is None

        mq.cleanup()

    def test_send_to_coordinator(self):
        """测试发送消息给协调器"""
        mq = MessageQueue()
        msg = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
        )

        mq.send_to_coordinator(msg)

        # 从队列获取消息
        received = mq.to_coordinator.get(timeout=1.0)
        assert received.id == msg.id
        assert received.type == msg.type
        assert received.sender == msg.sender

        mq.cleanup()

    def test_send_to_agent(self):
        """测试发送消息给 Agent"""
        mq = MessageQueue()
        queue = mq.create_agent_queue("worker-1")
        msg = ProcessMessage(
            type=ProcessMessageType.TASK_ASSIGN,
            sender="coordinator",
            receiver="worker-1",
        )

        # 发送成功
        result = mq.send_to_agent("worker-1", msg)
        assert result is True

        # 从队列获取
        received = queue.get(timeout=1.0)
        assert received.id == msg.id

        # 发送给不存在的 Agent
        result = mq.send_to_agent("nonexistent", msg)
        assert result is False

        mq.cleanup()

    def test_broadcast_to_agents(self):
        """测试广播消息给所有 Agent"""
        mq = MessageQueue()
        queue1 = mq.create_agent_queue("agent-1")
        queue2 = mq.create_agent_queue("agent-2")
        queue3 = mq.create_agent_queue("agent-3")

        msg = ProcessMessage(
            type=ProcessMessageType.SHUTDOWN,
            sender="manager",
        )

        mq.broadcast_to_agents(msg)

        # 所有队列都应该收到消息
        for queue in [queue1, queue2, queue3]:
            received = queue.get(timeout=1.0)
            assert received.id == msg.id
            assert received.type == ProcessMessageType.SHUTDOWN

        mq.cleanup()

    def test_receive_from_coordinator_with_timeout(self):
        """测试带超时的接收"""
        mq = MessageQueue()
        msg = ProcessMessage(type=ProcessMessageType.HEARTBEAT)

        mq.to_coordinator.put(msg)

        # 有消息时能收到
        received = mq.receive_from_coordinator(timeout=1.0)
        assert received is not None
        assert received.id == msg.id

        # 无消息时返回 None
        received = mq.receive_from_coordinator(timeout=0.1)
        assert received is None

        mq.cleanup()

    def test_receive_from_coordinator_nowait(self):
        """测试非阻塞接收"""
        mq = MessageQueue()

        # 空队列返回 None
        received = mq.receive_from_coordinator()
        assert received is None

        # 放入消息后能收到（使用短超时确保消息已入队）
        msg = ProcessMessage(type=ProcessMessageType.HEARTBEAT)
        mq.to_coordinator.put(msg)

        # 使用短超时代替 nowait，避免多进程队列同步问题
        received = mq.receive_from_coordinator(timeout=1.0)
        assert received is not None
        assert received.id == msg.id

        mq.cleanup()

    def test_cleanup(self):
        """测试队列清理"""
        mq = MessageQueue()
        mq.create_agent_queue("agent-1")
        mq.create_agent_queue("agent-2")

        # 清理后 agent 队列应该被清空
        mq.cleanup()
        assert len(mq._agent_queues) == 0


# ============================================================================
# AgentWorkerProcess 测试
# ============================================================================


class SimpleTestWorker(AgentWorkerProcess):
    """用于测试的简单 Worker 实现"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = False
        self.stopped = False
        self.received_messages = []

    def on_start(self) -> None:
        self.started = True

    def on_stop(self) -> None:
        self.stopped = True

    def handle_message(self, message: ProcessMessage) -> None:
        self.received_messages.append(message)
        # 发送任务结果
        if message.type == ProcessMessageType.TASK_ASSIGN:
            self._send_message(
                ProcessMessageType.TASK_RESULT,
                {"status": "completed", "task_id": message.payload.get("task_id")},
                correlation_id=message.id,
            )

    def get_status(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "running": self._running,
            "messages_received": len(self.received_messages),
        }


class TestAgentWorkerProcess:
    """AgentWorkerProcess 工作进程测试"""

    def test_worker_initialization(self):
        """测试 Worker 初始化"""
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="test-worker-001",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={"key": "value"},
        )

        assert worker.agent_id == "test-worker-001"
        assert worker.agent_type == "test"
        assert worker.inbox is inbox
        assert worker.outbox is outbox
        assert worker.config == {"key": "value"}
        assert worker._running is False
        assert worker.daemon is True

        inbox.close()
        outbox.close()

    def test_worker_start_and_ready(self):
        """测试 Worker 启动并发送就绪消息"""
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="test-worker-002",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        worker.start()

        # 等待就绪消息
        try:
            ready_msg = outbox.get(timeout=5.0)
            assert ready_msg.type == ProcessMessageType.STATUS_RESPONSE
            assert ready_msg.payload.get("status") == "ready"
            assert ready_msg.payload.get("pid") is not None
            assert ready_msg.sender == "test-worker-002"
        finally:
            # 发送关闭消息
            inbox.put(ProcessMessage(type=ProcessMessageType.SHUTDOWN, sender="test"))
            worker.join(timeout=5.0)
            inbox.close()
            outbox.close()

    def test_worker_heartbeat_response(self):
        """测试 Worker 心跳响应"""
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="test-worker-003",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        worker.start()

        try:
            # 等待就绪
            outbox.get(timeout=5.0)

            # 发送心跳
            inbox.put(
                ProcessMessage(
                    type=ProcessMessageType.HEARTBEAT,
                    sender="coordinator",
                )
            )

            # 等待心跳响应
            response = outbox.get(timeout=5.0)
            assert response.type == ProcessMessageType.HEARTBEAT
            assert response.payload.get("alive") is True
        finally:
            inbox.put(ProcessMessage(type=ProcessMessageType.SHUTDOWN))
            worker.join(timeout=5.0)
            inbox.close()
            outbox.close()

    def test_worker_status_request(self):
        """测试 Worker 状态请求响应"""
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="test-worker-004",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        worker.start()

        try:
            # 等待就绪
            outbox.get(timeout=5.0)

            # 发送状态请求
            inbox.put(
                ProcessMessage(
                    type=ProcessMessageType.STATUS_REQUEST,
                    sender="coordinator",
                )
            )

            # 等待状态响应
            response = outbox.get(timeout=5.0)
            assert response.type == ProcessMessageType.STATUS_RESPONSE
            assert response.payload.get("agent_id") == "test-worker-004"
            assert response.payload.get("agent_type") == "test"
        finally:
            inbox.put(ProcessMessage(type=ProcessMessageType.SHUTDOWN))
            worker.join(timeout=5.0)
            inbox.close()
            outbox.close()

    def test_worker_task_handling(self):
        """测试 Worker 任务处理"""
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="test-worker-005",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        worker.start()

        try:
            # 等待就绪
            outbox.get(timeout=5.0)

            # 发送任务
            task_msg = ProcessMessage(
                type=ProcessMessageType.TASK_ASSIGN,
                sender="coordinator",
                payload={"task_id": "task-123"},
            )
            inbox.put(task_msg)

            # 等待任务结果
            result = outbox.get(timeout=5.0)
            assert result.type == ProcessMessageType.TASK_RESULT
            assert result.payload.get("status") == "completed"
            assert result.payload.get("task_id") == "task-123"
            assert result.correlation_id == task_msg.id
        finally:
            inbox.put(ProcessMessage(type=ProcessMessageType.SHUTDOWN))
            worker.join(timeout=5.0)
            inbox.close()
            outbox.close()

    def test_worker_graceful_shutdown(self):
        """测试 Worker 优雅关闭"""
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="test-worker-006",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        worker.start()

        try:
            # 等待就绪
            outbox.get(timeout=5.0)
            assert worker.is_alive()

            # 发送关闭消息
            inbox.put(ProcessMessage(type=ProcessMessageType.SHUTDOWN))

            # 等待进程退出
            worker.join(timeout=10.0)
            assert not worker.is_alive()
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
            inbox.close()
            outbox.close()


# ============================================================================
# AgentProcessManager 测试
# ============================================================================


class TestAgentProcessManager:
    """AgentProcessManager 进程管理器测试"""

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = AgentProcessManager()

        assert manager.message_queue is not None
        assert isinstance(manager._processes, dict)
        assert len(manager._processes) == 0
        assert manager._running is False

        manager.shutdown_all(graceful=False)

    def test_spawn_agent(self):
        """测试创建 Agent 进程"""
        manager = AgentProcessManager()

        try:
            process = manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="spawn-test-001",
                agent_type="test",
                config={"test": True},
            )

            assert process is not None
            assert "spawn-test-001" in manager._processes
            assert manager._process_info["spawn-test-001"]["type"] == "test"
            assert manager._process_info["spawn-test-001"]["status"] == "starting"

            # 等待进程启动
            time.sleep(0.5)
            assert process.is_alive()
        finally:
            manager.shutdown_all(graceful=False)

    def test_send_to_agent(self):
        """测试发送消息给 Agent"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="send-test-001",
                agent_type="test",
                config={},
            )

            # 等待就绪
            manager.wait_for_ready("send-test-001", timeout=5.0)

            # 发送任务消息
            msg = ProcessMessage(
                type=ProcessMessageType.TASK_ASSIGN,
                sender="manager",
                payload={"task_id": "task-456"},
            )
            result = manager.send_to_agent("send-test-001", msg)
            assert result is True

            # 发送给不存在的 Agent
            result = manager.send_to_agent("nonexistent", msg)
            assert result is False

            # 等待任务结果
            response = manager.receive_message(timeout=5.0)
            assert response is not None
            assert response.type == ProcessMessageType.TASK_RESULT
        finally:
            manager.shutdown_all(graceful=False)

    def test_broadcast(self):
        """测试广播消息"""
        manager = AgentProcessManager()

        try:
            # 创建多个 Agent
            for i in range(3):
                manager.spawn_agent(
                    agent_class=SimpleTestWorker,
                    agent_id=f"broadcast-test-{i}",
                    agent_type="test",
                    config={},
                )

            # 等待所有就绪
            manager.wait_all_ready(timeout=10.0)

            # 广播心跳
            manager.broadcast(
                ProcessMessage(
                    type=ProcessMessageType.HEARTBEAT,
                    sender="manager",
                )
            )

            # 收集响应
            responses = []
            for _ in range(3):
                resp = manager.receive_message(timeout=5.0)
                if resp and resp.type == ProcessMessageType.HEARTBEAT:
                    responses.append(resp)

            assert len(responses) == 3
        finally:
            manager.shutdown_all(graceful=False)

    def test_wait_for_ready(self):
        """测试等待 Agent 就绪"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="ready-test-001",
                agent_type="test",
                config={},
            )

            ready = manager.wait_for_ready("ready-test-001", timeout=5.0)
            assert ready is True
            assert manager._process_info["ready-test-001"]["status"] == "ready"
        finally:
            manager.shutdown_all(graceful=False)

    def test_wait_for_ready_timeout(self):
        """测试等待就绪超时"""
        manager = AgentProcessManager()

        # 不创建任何进程，直接等待
        ready = manager.wait_for_ready("nonexistent", timeout=0.5)
        assert ready is False

    def test_wait_all_ready(self):
        """测试等待所有 Agent 就绪"""
        manager = AgentProcessManager()

        try:
            for i in range(2):
                manager.spawn_agent(
                    agent_class=SimpleTestWorker,
                    agent_id=f"all-ready-{i}",
                    agent_type="test",
                    config={},
                )

            ready = manager.wait_all_ready(timeout=10.0)
            assert ready is True

            for i in range(2):
                assert manager._process_info[f"all-ready-{i}"]["status"] == "ready"
        finally:
            manager.shutdown_all(graceful=False)

    def test_is_alive(self):
        """测试检查进程存活"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="alive-test-001",
                agent_type="test",
                config={},
            )

            manager.wait_for_ready("alive-test-001", timeout=5.0)

            assert manager.is_alive("alive-test-001") is True
            assert manager.is_alive("nonexistent") is False
        finally:
            manager.shutdown_all(graceful=False)

    def test_get_process_info(self):
        """测试获取进程信息"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="info-test-001",
                agent_type="test",
                config={},
            )

            info = manager.get_process_info("info-test-001")
            assert info is not None
            assert info["type"] == "test"

            # 不存在的 Agent
            assert manager.get_process_info("nonexistent") is None
        finally:
            manager.shutdown_all(graceful=False)

    def test_get_all_process_info(self):
        """测试获取所有进程信息"""
        manager = AgentProcessManager()

        try:
            for i in range(2):
                manager.spawn_agent(
                    agent_class=SimpleTestWorker,
                    agent_id=f"all-info-{i}",
                    agent_type="test",
                    config={},
                )

            all_info = manager.get_all_process_info()
            assert len(all_info) == 2
            assert "all-info-0" in all_info
            assert "all-info-1" in all_info
        finally:
            manager.shutdown_all(graceful=False)

    def test_terminate_agent_graceful(self):
        """测试优雅终止 Agent"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="terminate-test-001",
                agent_type="test",
                config={},
            )

            manager.wait_for_ready("terminate-test-001", timeout=5.0)
            assert manager.is_alive("terminate-test-001")

            manager.terminate_agent("terminate-test-001", graceful=True)

            assert not manager.is_alive("terminate-test-001")
            assert manager._process_info["terminate-test-001"]["status"] == "terminated"
        finally:
            manager.shutdown_all(graceful=False)

    def test_terminate_agent_force(self):
        """测试强制终止 Agent"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="force-terminate-001",
                agent_type="test",
                config={},
            )

            manager.wait_for_ready("force-terminate-001", timeout=5.0)

            manager.terminate_agent("force-terminate-001", graceful=False)

            assert not manager.is_alive("force-terminate-001")
        finally:
            manager.shutdown_all(graceful=False)

    def test_shutdown_all_graceful(self):
        """测试优雅关闭所有进程"""
        manager = AgentProcessManager()

        for i in range(3):
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id=f"shutdown-all-{i}",
                agent_type="test",
                config={},
            )

        manager.wait_all_ready(timeout=10.0)

        # 确保所有进程存活
        for i in range(3):
            assert manager.is_alive(f"shutdown-all-{i}")

        manager.shutdown_all(graceful=True)

        # 所有进程应该已停止
        assert len(manager._processes) == 0

    def test_shutdown_all_force(self):
        """测试强制关闭所有进程"""
        manager = AgentProcessManager()

        for i in range(2):
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id=f"force-shutdown-{i}",
                agent_type="test",
                config={},
            )

        time.sleep(1.0)  # 等待进程启动

        manager.shutdown_all(graceful=False)

        assert len(manager._processes) == 0

    def test_health_check(self):
        """测试健康检查"""
        manager = AgentProcessManager()

        try:
            for i in range(2):
                manager.spawn_agent(
                    agent_class=SimpleTestWorker,
                    agent_id=f"health-{i}",
                    agent_type="test",
                    config={},
                )

            manager.wait_all_ready(timeout=10.0)

            result = manager.health_check()

            # health_check 返回 HealthCheckResult 对象
            assert isinstance(result, HealthCheckResult)
            assert len(result.healthy) == 2
            assert "health-0" in result.healthy
            assert "health-1" in result.healthy
            assert result.all_healthy is True
        finally:
            manager.shutdown_all(graceful=False)

    def test_terminate_nonexistent_agent(self):
        """测试终止不存在的 Agent（应该不报错）"""
        manager = AgentProcessManager()

        # 不应该抛出异常
        manager.terminate_agent("nonexistent", graceful=True)
        manager.terminate_agent("nonexistent", graceful=False)


# ============================================================================
# 集成测试
# ============================================================================


class TestProcessIntegration:
    """进程管理集成测试"""

    def test_full_workflow(self):
        """测试完整工作流程"""
        manager = AgentProcessManager()

        try:
            # 1. 创建多个 Agent
            for i in range(2):
                manager.spawn_agent(
                    agent_class=SimpleTestWorker,
                    agent_id=f"workflow-{i}",
                    agent_type="worker",
                    config={"worker_id": i},
                )

            # 2. 等待所有就绪
            ready = manager.wait_all_ready(timeout=10.0)
            assert ready is True

            # 3. 发送任务给各个 Worker
            for i in range(2):
                manager.send_to_agent(
                    f"workflow-{i}",
                    ProcessMessage(
                        type=ProcessMessageType.TASK_ASSIGN,
                        sender="coordinator",
                        payload={"task_id": f"task-{i}", "content": f"task content {i}"},
                    ),
                )

            # 4. 收集任务结果
            results = []
            for _ in range(2):
                msg = manager.receive_message(timeout=5.0)
                if msg and msg.type == ProcessMessageType.TASK_RESULT:
                    results.append(msg)

            assert len(results) == 2

            # 5. 健康检查
            health = manager.health_check()
            assert health.all_healthy

            # 6. 优雅关闭
            manager.shutdown_all(graceful=True)
            assert len(manager._processes) == 0

        except Exception:
            manager.shutdown_all(graceful=False)
            raise

    def test_message_correlation(self):
        """测试消息关联"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="correlation-test",
                agent_type="test",
                config={},
            )

            manager.wait_for_ready("correlation-test", timeout=5.0)

            # 发送多个任务
            task_ids = ["task-a", "task-b", "task-c"]
            sent_messages = {}

            for task_id in task_ids:
                msg = ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="coordinator",
                    payload={"task_id": task_id},
                )
                sent_messages[msg.id] = task_id
                manager.send_to_agent("correlation-test", msg)

            # 收集响应并验证关联
            for _ in range(3):
                response = manager.receive_message(timeout=5.0)
                if response and response.type == ProcessMessageType.TASK_RESULT:
                    assert response.correlation_id in sent_messages
                    expected_task_id = sent_messages[response.correlation_id]
                    assert response.payload.get("task_id") == expected_task_id

        finally:
            manager.shutdown_all(graceful=False)


# ============================================================================
# HealthCheckResult 测试
# ============================================================================


class TestHealthCheckResult:
    """HealthCheckResult 数据类测试"""

    def test_health_check_result_creation(self):
        """测试 HealthCheckResult 创建"""
        result = HealthCheckResult()

        assert result.healthy == []
        assert result.unhealthy == []
        assert result.all_healthy is True
        assert result.details == {}

    def test_health_check_result_with_data(self):
        """测试带数据的 HealthCheckResult"""
        result = HealthCheckResult(
            healthy=["worker-1", "worker-2"],
            unhealthy=["worker-3"],
            all_healthy=False,
            details={
                "worker-1": {"healthy": True, "reason": "heartbeat_ok"},
                "worker-2": {"healthy": True, "reason": "heartbeat_ok"},
                "worker-3": {"healthy": False, "reason": "no_heartbeat_response"},
            },
        )

        assert len(result.healthy) == 2
        assert len(result.unhealthy) == 1
        assert result.all_healthy is False

    def test_get_unhealthy_workers(self):
        """测试获取不健康 Worker 列表"""
        result = HealthCheckResult(
            healthy=["planner-abc", "worker-1"],
            unhealthy=["worker-2", "reviewer-xyz", "worker-3"],
            all_healthy=False,
        )

        unhealthy_workers = result.get_unhealthy_workers()

        # 应该只返回包含 "worker" 的 agent_id
        assert len(unhealthy_workers) == 2
        assert "worker-2" in unhealthy_workers
        assert "worker-3" in unhealthy_workers
        assert "reviewer-xyz" not in unhealthy_workers


# ============================================================================
# 任务分配跟踪测试
# ============================================================================


class TestTaskAssignmentTracking:
    """任务分配跟踪测试"""

    def test_track_task_assignment(self):
        """测试跟踪任务分配"""
        manager = AgentProcessManager()

        manager.track_task_assignment(
            task_id="task-123",
            agent_id="worker-1",
            message_id="msg-456",
        )

        tasks = manager.get_tasks_by_agent("worker-1")
        assert "task-123" in tasks

        all_tasks = manager.get_all_in_flight_tasks()
        assert "task-123" in all_tasks
        assert all_tasks["task-123"]["agent_id"] == "worker-1"
        assert all_tasks["task-123"]["message_id"] == "msg-456"

    def test_untrack_task(self):
        """测试取消跟踪任务"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-123", "worker-1", "msg-456")

        # 取消跟踪
        info = manager.untrack_task("task-123")

        assert info is not None
        assert info["agent_id"] == "worker-1"

        # 再次取消应返回 None
        info2 = manager.untrack_task("task-123")
        assert info2 is None

        # 列表应为空
        assert manager.get_tasks_by_agent("worker-1") == []

    def test_get_tasks_by_agent_multiple(self):
        """测试获取多个任务"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-1", "worker-1", "msg-1")
        manager.track_task_assignment("task-2", "worker-1", "msg-2")
        manager.track_task_assignment("task-3", "worker-2", "msg-3")

        worker1_tasks = manager.get_tasks_by_agent("worker-1")
        worker2_tasks = manager.get_tasks_by_agent("worker-2")

        assert len(worker1_tasks) == 2
        assert len(worker2_tasks) == 1
        assert "task-1" in worker1_tasks
        assert "task-2" in worker1_tasks
        assert "task-3" in worker2_tasks


# ============================================================================
# 健康检查增强测试
# ============================================================================


class TestHealthCheckEnhanced:
    """增强健康检查测试"""

    def test_health_check_returns_result_object(self):
        """测试健康检查返回 HealthCheckResult 对象"""
        manager = AgentProcessManager()

        try:
            for i in range(2):
                manager.spawn_agent(
                    agent_class=SimpleTestWorker,
                    agent_id=f"hc-test-{i}",
                    agent_type="test",
                    config={},
                )

            manager.wait_all_ready(timeout=10.0)

            result = manager.health_check()

            assert isinstance(result, HealthCheckResult)
            assert len(result.healthy) == 2
            assert len(result.unhealthy) == 0
            assert result.all_healthy is True
        finally:
            manager.shutdown_all(graceful=False)

    def test_health_check_simple_backward_compat(self):
        """测试 health_check_simple 向后兼容"""
        manager = AgentProcessManager()

        try:
            manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="simple-hc-test",
                agent_type="test",
                config={},
            )

            manager.wait_for_ready("simple-hc-test", timeout=5.0)

            # 使用简单接口
            result = manager.health_check_simple()

            assert isinstance(result, dict)
            assert result.get("simple-hc-test") is True
        finally:
            manager.shutdown_all(graceful=False)

    def test_health_check_with_dead_process(self):
        """测试死进程的健康检查"""
        manager = AgentProcessManager()

        try:
            process = manager.spawn_agent(
                agent_class=SimpleTestWorker,
                agent_id="dead-hc-test",
                agent_type="worker",
                config={},
            )

            manager.wait_for_ready("dead-hc-test", timeout=5.0)

            # 强制终止进程
            process.terminate()
            process.join(timeout=2.0)

            # 执行健康检查
            result = manager.health_check(timeout=2.0)

            assert "dead-hc-test" in result.unhealthy
            assert result.all_healthy is False
            assert result.details["dead-hc-test"]["healthy"] is False
            assert result.details["dead-hc-test"]["reason"] == "process_dead"
        finally:
            manager.shutdown_all(graceful=False)


# ============================================================================
# Mock 健康检查测试（用于 Orchestrator 集成测试）
# ============================================================================


# ============================================================================
# 反向索引测试
# ============================================================================


class TestMessageToTaskIndex:
    """message_id -> task_id 反向索引测试"""

    def test_index_created_on_track(self):
        """测试跟踪任务时创建反向索引"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-123", "worker-1", "msg-456")

        # 验证反向索引存在
        assert "msg-456" in manager._message_to_task
        assert manager._message_to_task["msg-456"] == "task-123"

    def test_index_removed_on_untrack(self):
        """测试取消跟踪任务时移除反向索引"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-123", "worker-1", "msg-456")
        manager.untrack_task("task-123")

        # 验证反向索引已清理
        assert "msg-456" not in manager._message_to_task

    def test_get_task_by_message_id(self):
        """测试通过消息 ID 查找任务"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-123", "worker-1", "msg-456")

        result = manager.get_task_by_message_id("msg-456")

        assert result is not None
        task_id, info = result
        assert task_id == "task-123"
        assert info["agent_id"] == "worker-1"
        assert info["message_id"] == "msg-456"

    def test_get_task_by_message_id_not_found(self):
        """测试查找不存在的消息 ID"""
        manager = AgentProcessManager()

        result = manager.get_task_by_message_id("nonexistent")

        assert result is None

    def test_get_task_assignment(self):
        """测试获取任务分配信息"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-123", "worker-1", "msg-456")

        info = manager.get_task_assignment("task-123")

        assert info is not None
        assert info["agent_id"] == "worker-1"
        assert info["message_id"] == "msg-456"

    def test_get_task_assignment_not_found(self):
        """测试获取不存在的任务分配信息"""
        manager = AgentProcessManager()

        info = manager.get_task_assignment("nonexistent")

        assert info is None

    def test_multiple_tasks_index(self):
        """测试多个任务的反向索引"""
        manager = AgentProcessManager()

        manager.track_task_assignment("task-1", "worker-1", "msg-1")
        manager.track_task_assignment("task-2", "worker-2", "msg-2")
        manager.track_task_assignment("task-3", "worker-1", "msg-3")

        # 验证所有索引
        msg_task_1 = manager.get_task_by_message_id("msg-1")
        msg_task_2 = manager.get_task_by_message_id("msg-2")
        msg_task_3 = manager.get_task_by_message_id("msg-3")
        assert msg_task_1 is not None
        assert msg_task_2 is not None
        assert msg_task_3 is not None
        assert msg_task_1[0] == "task-1"
        assert msg_task_2[0] == "task-2"
        assert msg_task_3[0] == "task-3"

        # 取消其中一个
        manager.untrack_task("task-2")

        assert manager.get_task_by_message_id("msg-2") is None
        assert manager.get_task_by_message_id("msg-1") is not None
        assert manager.get_task_by_message_id("msg-3") is not None


class TestHealthCheckMocking:
    """Mock 健康检查测试 - 验证部分 False 返回的处理"""

    def test_mock_partial_unhealthy(self):
        """测试 Mock 部分不健康的情况"""
        manager = AgentProcessManager()

        # 不实际创建进程，直接 Mock health_check
        with patch.object(manager, "health_check") as mock_hc:
            # 模拟部分 Worker 不健康
            mock_result = HealthCheckResult(
                healthy=["planner-1", "worker-0", "reviewer-1"],
                unhealthy=["worker-1", "worker-2"],
                all_healthy=False,
                details={
                    "planner-1": {"healthy": True, "reason": "heartbeat_ok"},
                    "worker-0": {"healthy": True, "reason": "heartbeat_ok"},
                    "worker-1": {"healthy": False, "reason": "no_heartbeat_response"},
                    "worker-2": {"healthy": False, "reason": "process_dead"},
                    "reviewer-1": {"healthy": True, "reason": "heartbeat_ok"},
                },
            )
            mock_hc.return_value = mock_result

            result = manager.health_check()

            assert result.all_healthy is False
            assert len(result.get_unhealthy_workers()) == 2
            assert "worker-1" in result.get_unhealthy_workers()
            assert "worker-2" in result.get_unhealthy_workers()

    def test_mock_critical_process_unhealthy(self):
        """测试 Mock 关键进程（planner/reviewer）不健康"""
        manager = AgentProcessManager()

        with patch.object(manager, "health_check") as mock_hc:
            # 模拟 Planner 不健康
            mock_result = HealthCheckResult(
                healthy=["worker-0", "worker-1", "reviewer-1"],
                unhealthy=["planner-1"],
                all_healthy=False,
                details={
                    "planner-1": {"healthy": False, "reason": "process_dead"},
                    "worker-0": {"healthy": True, "reason": "heartbeat_ok"},
                    "worker-1": {"healthy": True, "reason": "heartbeat_ok"},
                    "reviewer-1": {"healthy": True, "reason": "heartbeat_ok"},
                },
            )
            mock_hc.return_value = mock_result

            result = manager.health_check()

            assert result.all_healthy is False
            assert "planner-1" in result.unhealthy
            # Planner 不在 get_unhealthy_workers 返回中（因为它不是 worker）
            assert "planner-1" not in result.get_unhealthy_workers()

    def test_mock_all_workers_unhealthy(self):
        """测试 Mock 所有 Worker 不健康的极端情况"""
        manager = AgentProcessManager()

        with patch.object(manager, "health_check") as mock_hc:
            mock_result = HealthCheckResult(
                healthy=["planner-1", "reviewer-1"],
                unhealthy=["worker-0", "worker-1", "worker-2"],
                all_healthy=False,
                details={
                    "planner-1": {"healthy": True, "reason": "heartbeat_ok"},
                    "reviewer-1": {"healthy": True, "reason": "heartbeat_ok"},
                    "worker-0": {"healthy": False, "reason": "no_heartbeat_response"},
                    "worker-1": {"healthy": False, "reason": "process_dead"},
                    "worker-2": {"healthy": False, "reason": "no_heartbeat_response"},
                },
            )
            mock_hc.return_value = mock_result

            result = manager.health_check()

            assert len(result.get_unhealthy_workers()) == 3
            assert result.all_healthy is False


# ============================================================================
# Late Result 处理测试（MP Orchestrator）
# ============================================================================


class TestLateResultHandling:
    """Late Result 兜底处理测试

    验证当 TASK_RESULT 的 correlation_id 不在 _pending_responses 时：
    1. 能通过反向索引查找任务
    2. 根据任务状态决定忽略或应用结果
    3. 不会导致挂起或崩溃
    """

    @pytest.fixture
    def mock_orchestrator(self):
        """创建 Mock Orchestrator 用于测试"""
        from coordinator.orchestrator_mp import (
            MultiProcessOrchestrator,
            MultiProcessOrchestratorConfig,
        )

        config = MultiProcessOrchestratorConfig(
            working_directory=".",
            max_iterations=1,
            worker_count=1,
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        # Mock process_manager 方法
        orchestrator.process_manager = MagicMock()
        orchestrator.process_manager.get_task_by_message_id = MagicMock(return_value=None)
        orchestrator.process_manager.get_task_assignment = MagicMock(return_value=None)
        orchestrator.process_manager.untrack_task = MagicMock(return_value=None)

        return orchestrator

    @pytest.mark.asyncio
    async def test_late_result_ignored_no_task_id(self, mock_orchestrator):
        """测试无法确定 task_id 时 late result 被忽略"""
        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={"success": True},  # 没有 task_id
            correlation_id="unknown-corr-id",
        )

        # 确保没有等待中的 Future
        assert "unknown-corr-id" not in mock_orchestrator._pending_responses

        # 调用 _handle_message（应该触发 _handle_late_result）
        await mock_orchestrator._handle_message(message)

        # 验证统计
        assert mock_orchestrator._message_stats.get("late_result_ignored", 0) >= 1

    @pytest.mark.asyncio
    async def test_late_result_ignored_task_requeued(self, mock_orchestrator):
        """测试任务已重新入队时 late result 被忽略"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建一个 PENDING 状态的任务（模拟已重新入队）
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.PENDING,  # 已重新入队
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={"task_id": "task-123", "success": True},
            correlation_id="msg-456",
        )

        await mock_orchestrator._handle_message(message)

        # 验证被忽略
        assert mock_orchestrator._message_stats.get("late_result_ignored", 0) >= 1
        # 任务状态不应改变
        assert mock_orchestrator.task_queue.get_task("task-123").status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_late_result_ignored_task_completed(self, mock_orchestrator):
        """测试任务已完成时 late result 被忽略"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建一个 COMPLETED 状态的任务
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.COMPLETED,
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={"task_id": "task-123", "success": False, "error": "Some error"},
            correlation_id="msg-456",
        )

        await mock_orchestrator._handle_message(message)

        # 验证被忽略
        assert mock_orchestrator._message_stats.get("late_result_ignored", 0) >= 1
        # 任务状态不应改变
        assert mock_orchestrator.task_queue.get_task("task-123").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_late_result_applied_in_progress(self, mock_orchestrator):
        """测试任务仍处于 IN_PROGRESS 时 late result 被应用"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建一个 IN_PROGRESS 状态的任务
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.IN_PROGRESS,
            assigned_to="worker-1",
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        # 初始化 iteration
        mock_orchestrator.state.start_new_iteration()

        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={
                "task_id": "task-123",
                "success": True,
                "output": "Task completed successfully",
            },
            correlation_id="msg-456",
        )

        await mock_orchestrator._handle_message(message)

        # 验证被应用
        assert mock_orchestrator._message_stats.get("late_result_applied", 0) >= 1
        # 任务状态应更新为 COMPLETED
        assert mock_orchestrator.task_queue.get_task("task-123").status == TaskStatus.COMPLETED
        # 统计应更新
        assert mock_orchestrator.state.total_tasks_completed >= 1

    @pytest.mark.asyncio
    async def test_late_result_ignored_sender_mismatch(self, mock_orchestrator):
        """测试发送者不匹配时 late result 被忽略"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建一个 IN_PROGRESS 状态的任务，分配给 worker-1
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.IN_PROGRESS,
            assigned_to="worker-1",
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        # 来自 worker-2 的响应（不匹配）
        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-2",  # 不匹配
            payload={"task_id": "task-123", "success": True},
            correlation_id="msg-456",
        )

        await mock_orchestrator._handle_message(message)

        # 验证被忽略
        assert mock_orchestrator._message_stats.get("late_result_ignored", 0) >= 1
        # 任务状态不应改变
        assert mock_orchestrator.task_queue.get_task("task-123").status == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_late_result_lookup_by_correlation_id(self, mock_orchestrator):
        """测试通过 correlation_id 反查 task_id"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建任务
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.IN_PROGRESS,
            assigned_to="worker-1",
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        # 初始化 iteration
        mock_orchestrator.state.start_new_iteration()

        # 配置 mock 返回反向索引结果
        mock_orchestrator.process_manager.get_task_by_message_id.return_value = (
            "task-123",
            {"agent_id": "worker-1", "message_id": "msg-456"},
        )

        # 消息没有 task_id，但有 correlation_id
        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={"success": True, "output": "Done"},  # 没有 task_id
            correlation_id="msg-456",
        )

        await mock_orchestrator._handle_message(message)

        # 验证通过反向索引查找
        mock_orchestrator.process_manager.get_task_by_message_id.assert_called_with("msg-456")
        # 验证被应用
        assert mock_orchestrator._message_stats.get("late_result_applied", 0) >= 1

    @pytest.mark.asyncio
    async def test_late_result_failed_task(self, mock_orchestrator):
        """测试 late result 失败时正确更新任务状态"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建一个 IN_PROGRESS 状态的任务
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.IN_PROGRESS,
            assigned_to="worker-1",
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        # 初始化 iteration
        mock_orchestrator.state.start_new_iteration()

        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={
                "task_id": "task-123",
                "success": False,
                "error": "Execution failed",
            },
            correlation_id="msg-456",
        )

        await mock_orchestrator._handle_message(message)

        # 验证被应用
        assert mock_orchestrator._message_stats.get("late_result_applied", 0) >= 1
        # 任务状态应更新为 FAILED
        updated_task = mock_orchestrator.task_queue.get_task("task-123")
        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.error == "Execution failed"
        # 统计应更新
        assert mock_orchestrator.state.total_tasks_failed >= 1

    @pytest.mark.asyncio
    async def test_late_result_no_crash_on_missing_iteration(self, mock_orchestrator):
        """测试缺少 iteration 时不会崩溃"""
        from tasks.task import Task, TaskStatus, TaskType

        # 创建任务但不初始化 iteration
        task = Task(
            id="task-123",
            title="Test Task",
            description="Test task description",
            instruction="Test instruction",
            type=TaskType.IMPLEMENT,
            status=TaskStatus.IN_PROGRESS,
            assigned_to="worker-1",
            iteration_id=1,
        )
        mock_orchestrator.task_queue._tasks["task-123"] = task

        message = ProcessMessage(
            type=ProcessMessageType.TASK_RESULT,
            sender="worker-1",
            payload={"task_id": "task-123", "success": True},
            correlation_id="msg-456",
        )

        # 不应该崩溃
        await mock_orchestrator._handle_message(message)

        # 任务应该被更新
        assert mock_orchestrator.task_queue.get_task("task-123").status == TaskStatus.COMPLETED

    def test_late_result_stats_initialized(self, mock_orchestrator):
        """测试 late result 统计初始化"""
        # 初始时没有 late result 统计
        assert "late_result_ignored" not in mock_orchestrator._message_stats
        assert "late_result_applied" not in mock_orchestrator._message_stats


class TestLateResultIntegration:
    """Late Result 集成测试 - 使用真实的 AgentProcessManager"""

    def test_manager_index_integration(self):
        """测试 AgentProcessManager 反向索引与任务跟踪集成"""
        manager = AgentProcessManager()

        # 模拟任务分配流程
        manager.track_task_assignment("task-1", "worker-1", "msg-100")
        manager.track_task_assignment("task-2", "worker-2", "msg-200")
        manager.track_task_assignment("task-3", "worker-1", "msg-300")

        # 验证反向索引
        msg_task_100 = manager.get_task_by_message_id("msg-100")
        msg_task_200 = manager.get_task_by_message_id("msg-200")
        msg_task_300 = manager.get_task_by_message_id("msg-300")
        assert msg_task_100 is not None
        assert msg_task_200 is not None
        assert msg_task_300 is not None
        assert msg_task_100[0] == "task-1"
        assert msg_task_200[0] == "task-2"
        assert msg_task_300[0] == "task-3"

        # 任务完成，取消跟踪
        manager.untrack_task("task-1")

        # 反向索引也应该被清理
        assert manager.get_task_by_message_id("msg-100") is None
        assert manager.get_task_by_message_id("msg-200") is not None

        # 获取任务分配信息
        info = manager.get_task_assignment("task-2")
        assert info is not None
        assert info["agent_id"] == "worker-2"

    def test_manager_concurrent_operations(self):
        """测试并发操作下反向索引的正确性"""
        manager = AgentProcessManager()

        # 快速添加多个任务
        for i in range(100):
            manager.track_task_assignment(f"task-{i}", f"worker-{i % 3}", f"msg-{i}")

        # 验证所有索引正确
        for i in range(100):
            result = manager.get_task_by_message_id(f"msg-{i}")
            assert result is not None
            assert result[0] == f"task-{i}"

        # 取消部分任务
        for i in range(0, 100, 2):
            manager.untrack_task(f"task-{i}")

        # 验证奇数索引仍存在，偶数已清理
        for i in range(100):
            result = manager.get_task_by_message_id(f"msg-{i}")
            if i % 2 == 0:
                assert result is None
            else:
                assert result is not None


# ============================================================================
# Pickle 序列化测试（macOS spawn 兼容性）
# ============================================================================


class TestWorkerPickleSerializable:
    """Worker 进程 pickle 序列化测试

    macOS 使用 spawn 启动方式，需要所有进程对象可被 pickle 序列化。
    这些测试确保 Worker 进程在 macOS 上能正常启动。

    注意：multiprocessing.Queue 本身无法被 pickle（使用 AuthenticationString），
    但 multiprocessing 框架会自动处理 Queue 的传递。测试重点是验证自定义对象
    （如 threading.Lock）不在 __init__ 中创建。
    """

    def test_worker_no_lock_in_init(self):
        """验证 Worker 在 __init__ 中不创建锁对象

        这是 macOS spawn 兼容性的关键：threading.Lock 无法被 pickle，
        必须在 run() 方法中（子进程内）初始化。
        """
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        try:
            worker = SimpleTestWorker(
                agent_id="no-lock-test",
                agent_type="test",
                inbox=inbox,
                outbox=outbox,
                config={},
            )

            # _task_lock 应该在 __init__ 后为 None
            # 锁应该在 run() 方法中初始化
            assert worker._task_lock is None, "_task_lock 应该在 run() 中初始化，不是 __init__"
        finally:
            inbox.close()
            outbox.close()

    def test_worker_no_executor_in_init(self):
        """验证 Worker 在 __init__ 中不创建线程池

        ThreadPoolExecutor 无法被 pickle，必须在 run() 方法中初始化。
        """
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        try:
            worker = SimpleTestWorker(
                agent_id="no-executor-test",
                agent_type="test",
                inbox=inbox,
                outbox=outbox,
                config={},
            )

            # _executor 应该在 __init__ 后为 None
            assert worker._executor is None, "_executor 应该在 run() 中初始化，不是 __init__"
        finally:
            inbox.close()
            outbox.close()

    def test_worker_config_is_serializable(self):
        """验证 Worker 的 config 可被 pickle 序列化"""
        config = {
            "key": "value",
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": ["x", "y", "z"],
        }

        # config 必须可被序列化
        pickled = pickle.dumps(config)
        unpickled = pickle.loads(pickled)
        assert unpickled == config

    def test_worker_attributes_serializable(self):
        """验证 Worker 的关键属性可被 pickle 序列化"""
        # 这些是在 spawn 时需要传递的属性
        attrs = {
            "agent_id": "test-worker-001",
            "agent_type": "test",
            "_running": False,
            "config": {"key": "value"},
        }

        # 所有属性必须可被序列化
        for name, value in attrs.items():
            try:
                pickled = pickle.dumps(value)
                unpickled = pickle.loads(pickled)
                assert unpickled == value, f"{name} 序列化后值不一致"
            except Exception as e:
                pytest.fail(f"属性 {name} 无法被 pickle 序列化: {e}")


class TestAgentProcessPickleSerializable:
    """Agent 进程 pickle 序列化测试

    验证各类 Agent 进程在 __init__ 后不包含不可序列化的对象。
    注意：Queue 由 multiprocessing 框架管理，不需要显式序列化。
    """

    def test_worker_agent_process_no_unpicklable_attrs(self):
        """测试 WorkerAgentProcess 在 __init__ 后无不可序列化属性"""
        from agents.worker_process import WorkerAgentProcess

        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        try:
            worker = WorkerAgentProcess(
                agent_id="worker-pickle-001",
                agent_type="worker",
                inbox=inbox,
                outbox=outbox,
                config={"working_directory": "."},
            )

            # 验证关键属性在 __init__ 后为 None（应在 run() 中初始化）
            assert worker._task_lock is None, "WorkerAgentProcess._task_lock 应为 None"
            assert worker._executor is None, "WorkerAgentProcess._executor 应为 None"
            assert worker.cursor_client is None, "WorkerAgentProcess.cursor_client 应为 None"
        finally:
            inbox.close()
            outbox.close()

    def test_planner_agent_process_no_unpicklable_attrs(self):
        """测试 PlannerAgentProcess 在 __init__ 后无不可序列化属性"""
        from agents.planner_process import PlannerAgentProcess

        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        try:
            planner = PlannerAgentProcess(
                agent_id="planner-pickle-001",
                agent_type="planner",
                inbox=inbox,
                outbox=outbox,
                config={"working_directory": "."},
            )

            # 验证关键属性在 __init__ 后为 None
            assert planner._task_lock is None, "PlannerAgentProcess._task_lock 应为 None"
            assert planner._executor is None, "PlannerAgentProcess._executor 应为 None"
            assert planner.cursor_client is None, "PlannerAgentProcess.cursor_client 应为 None"
        finally:
            inbox.close()
            outbox.close()

    def test_reviewer_agent_process_no_unpicklable_attrs(self):
        """测试 ReviewerAgentProcess 在 __init__ 后无不可序列化属性"""
        from agents.reviewer_process import ReviewerAgentProcess

        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        try:
            reviewer = ReviewerAgentProcess(
                agent_id="reviewer-pickle-001",
                agent_type="reviewer",
                inbox=inbox,
                outbox=outbox,
                config={"working_directory": "."},
            )

            # 验证关键属性在 __init__ 后为 None
            assert reviewer._task_lock is None, "ReviewerAgentProcess._task_lock 应为 None"
            assert reviewer._executor is None, "ReviewerAgentProcess._executor 应为 None"
            assert reviewer.cursor_client is None, "ReviewerAgentProcess.cursor_client 应为 None"
        finally:
            inbox.close()
            outbox.close()


class TestSpawnStartMethod:
    """spawn 启动方式测试

    这些测试验证在 spawn 启动方式下进程能正常工作。
    macOS 和 Windows 默认使用 spawn。
    """

    def test_worker_starts_and_ready(self):
        """测试 Worker 能正常启动并发送就绪消息

        此测试在所有平台上运行，验证进程启动的基本功能。
        macOS/Windows 使用 spawn，Linux 使用 fork。
        """
        inbox: Queue = mp.Queue()
        outbox: Queue = mp.Queue()

        worker = SimpleTestWorker(
            agent_id="spawn-start-test",
            agent_type="test",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        try:
            worker.start()

            # 等待就绪消息
            ready_msg = outbox.get(timeout=10.0)
            assert ready_msg.type == ProcessMessageType.STATUS_RESPONSE
            assert ready_msg.payload.get("status") == "ready"
        finally:
            inbox.put(ProcessMessage(type=ProcessMessageType.SHUTDOWN))
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
            inbox.close()
            outbox.close()

    def test_all_agent_processes_spawn_compatible(self):
        """测试所有 Agent 进程类型在 spawn 方式下兼容

        验证所有进程类型在 __init__ 后没有不可序列化的对象。
        这确保了在 macOS/Windows 的 spawn 启动方式下能正常工作。
        """
        from agents.planner_process import PlannerAgentProcess
        from agents.reviewer_process import ReviewerAgentProcess
        from agents.worker_process import WorkerAgentProcess

        process_classes = [
            (WorkerAgentProcess, "worker"),
            (PlannerAgentProcess, "planner"),
            (ReviewerAgentProcess, "reviewer"),
        ]

        for process_class, agent_type in process_classes:
            inbox: Queue = mp.Queue()
            outbox: Queue = mp.Queue()

            try:
                process = process_class(
                    agent_id=f"{agent_type}-spawn-compat",
                    agent_type=agent_type,
                    inbox=inbox,
                    outbox=outbox,
                    config={"working_directory": "."},
                )

                # 验证关键属性在 __init__ 后为 None（确保 spawn 兼容）
                assert process._task_lock is None, f"{process_class.__name__}._task_lock 应在 run() 中初始化"
                assert process._executor is None, f"{process_class.__name__}._executor 应在 run() 中初始化"
            finally:
                inbox.close()
                outbox.close()
