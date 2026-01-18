"""进程管理模块测试

测试 process/manager.py、process/message_queue.py、process/worker.py 的功能
"""
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from process.manager import AgentProcessManager
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

        reply = original.create_reply(
            ProcessMessageType.TASK_RESULT,
            {"status": "completed"}
        )

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
        inbox = mp.Queue()
        outbox = mp.Queue()

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
        inbox = mp.Queue()
        outbox = mp.Queue()

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
        inbox = mp.Queue()
        outbox = mp.Queue()

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
            inbox.put(ProcessMessage(
                type=ProcessMessageType.HEARTBEAT,
                sender="coordinator",
            ))

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
        inbox = mp.Queue()
        outbox = mp.Queue()

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
            inbox.put(ProcessMessage(
                type=ProcessMessageType.STATUS_REQUEST,
                sender="coordinator",
            ))

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
        inbox = mp.Queue()
        outbox = mp.Queue()

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
        inbox = mp.Queue()
        outbox = mp.Queue()

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
            manager.broadcast(ProcessMessage(
                type=ProcessMessageType.HEARTBEAT,
                sender="manager",
            ))

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

            results = manager.health_check()

            assert len(results) == 2
            assert results.get("health-0") is True
            assert results.get("health-1") is True
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
                manager.send_to_agent(f"workflow-{i}", ProcessMessage(
                    type=ProcessMessageType.TASK_ASSIGN,
                    sender="coordinator",
                    payload={"task_id": f"task-{i}", "content": f"task content {i}"},
                ))

            # 4. 收集任务结果
            results = []
            for _ in range(2):
                msg = manager.receive_message(timeout=5.0)
                if msg and msg.type == ProcessMessageType.TASK_RESULT:
                    results.append(msg)

            assert len(results) == 2

            # 5. 健康检查
            health = manager.health_check()
            assert all(health.values())

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
