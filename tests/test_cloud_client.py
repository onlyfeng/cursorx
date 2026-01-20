"""Cloud Client 测试模块

测试内容:
1. CloudAgentClient 初始化和配置
2. mock HTTP 请求，测试 create_task、get_task_status、execute 等方法
3. 认证流程（成功/失败）
4. 错误处理（限流、网络错误、认证失败）
5. 重试机制
6. AgentExecutorFactory 创建正确的 Executor
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 推荐: 从 cursor 顶层包导入（统一入口）
from cursor import (
    # 执行器
    AgentExecutorFactory,
    AgentResult,
    AskAgentExecutor,
    # 认证相关
    AuthError,
    AuthErrorCode,
    AuthStatus,
    AuthToken,
    AutoAgentExecutor,
    CLIAgentExecutor,
    CloudAgentExecutor,
    CloudAuthConfig,
    CloudAuthManager,
    # 任务管理
    CloudTask,
    CloudTaskClient,
    CloudTaskOptions,
    CursorCloudClient,
    ExecutionMode,
    PlanAgentExecutor,
    TaskResult,
    TaskStatus,
    get_api_key,
    require_auth,
)

# 兼容性: 也可以从子模块直接导入
# from cursor.cloud_client import AuthError, AuthErrorCode, ...
# from cursor.executor import AgentExecutorFactory, ...


# ========== 固定 Fixtures ==========


@pytest.fixture
def mock_api_key():
    """模拟 API Key"""
    return "test-api-key-12345"


@pytest.fixture
def mock_env_api_key(mock_api_key):
    """设置环境变量中的 API Key"""
    with patch.dict(os.environ, {"CURSOR_API_KEY": mock_api_key}):
        yield mock_api_key


@pytest.fixture
def auth_config():
    """认证配置"""
    return CloudAuthConfig(
        api_key="test-config-api-key",
        auth_timeout=10,
        max_retries=2,
    )


@pytest.fixture
def auth_manager(auth_config):
    """认证管理器"""
    return CloudAuthManager(config=auth_config)


@pytest.fixture
def cloud_task_client(auth_manager):
    """Cloud 任务客户端"""
    return CloudTaskClient(auth_manager=auth_manager)


@pytest.fixture
def cursor_cloud_client(auth_manager):
    """Cursor Cloud 客户端"""
    return CursorCloudClient(auth_manager=auth_manager)


# ========== AuthToken 测试 ==========


class TestAuthToken:
    """AuthToken 测试"""

    def test_token_not_expired_without_expiry(self):
        """无过期时间的 Token 不会过期"""
        token = AuthToken(access_token="test-token")
        assert token.is_expired is False
        assert token.expires_in_seconds is None

    def test_token_expired(self):
        """过期 Token 检测"""
        token = AuthToken(
            access_token="test-token",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert token.is_expired is True

    def test_token_not_expired(self):
        """未过期 Token 检测"""
        token = AuthToken(
            access_token="test-token",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert token.is_expired is False
        assert token.expires_in_seconds > 0

    def test_token_expiring_soon(self):
        """即将过期的 Token（5 分钟内）视为已过期"""
        token = AuthToken(
            access_token="test-token",
            expires_at=datetime.now() + timedelta(minutes=3),
        )
        assert token.is_expired is True

    def test_token_to_dict(self):
        """Token 转字典"""
        expires = datetime.now() + timedelta(hours=1)
        token = AuthToken(
            access_token="test-token",
            token_type="Bearer",
            expires_at=expires,
            refresh_token="refresh-123",
        )
        data = token.to_dict()
        assert data["access_token"] == "test-token"
        assert data["token_type"] == "Bearer"
        assert data["refresh_token"] == "refresh-123"
        assert data["expires_at"] is not None

    def test_token_from_dict(self):
        """从字典创建 Token"""
        data = {
            "access_token": "from-dict-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        token = AuthToken.from_dict(data)
        assert token.access_token == "from-dict-token"
        assert token.expires_at is not None


# ========== AuthStatus 测试 ==========


class TestAuthStatus:
    """AuthStatus 测试"""

    def test_needs_refresh_when_token_expired(self):
        """Token 过期时需要刷新"""
        token = AuthToken(
            access_token="test",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        status = AuthStatus(authenticated=True, token=token)
        assert status.needs_refresh is True

    def test_no_refresh_when_not_authenticated(self):
        """未认证时不需要刷新"""
        status = AuthStatus(authenticated=False)
        assert status.needs_refresh is False

    def test_no_refresh_when_token_valid(self):
        """Token 有效时不需要刷新"""
        token = AuthToken(
            access_token="test",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        status = AuthStatus(authenticated=True, token=token)
        assert status.needs_refresh is False

    def test_to_dict(self):
        """AuthStatus 转字典"""
        status = AuthStatus(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
            plan="pro",
        )
        data = status.to_dict()
        assert data["authenticated"] is True
        assert data["user_id"] == "user-123"
        assert data["email"] == "test@example.com"
        assert data["plan"] == "pro"


# ========== AuthError 测试 ==========


class TestAuthError:
    """AuthError 测试"""

    def test_error_str(self):
        """错误字符串格式"""
        error = AuthError("测试错误", AuthErrorCode.INVALID_API_KEY)
        assert "[invalid_api_key]" in str(error)
        assert "测试错误" in str(error)

    def test_user_friendly_message(self):
        """用户友好错误消息"""
        error = AuthError("", AuthErrorCode.INVALID_API_KEY)
        message = error.user_friendly_message
        assert "API Key" in message

    def test_rate_limited_message(self):
        """限流错误消息"""
        error = AuthError("", AuthErrorCode.RATE_LIMITED)
        message = error.user_friendly_message
        assert "限流" in message

    def test_network_error_message(self):
        """网络错误消息"""
        error = AuthError("", AuthErrorCode.NETWORK_ERROR)
        message = error.user_friendly_message
        assert "网络" in message


# ========== CloudAuthManager 测试 ==========


class TestCloudAuthManager:
    """CloudAuthManager 测试"""

    def test_init_default_config(self):
        """默认配置初始化"""
        manager = CloudAuthManager()
        assert manager.config is not None
        assert manager.is_authenticated is False

    def test_init_with_config(self, auth_config):
        """自定义配置初始化"""
        manager = CloudAuthManager(config=auth_config)
        assert manager.config.api_key == "test-config-api-key"
        assert manager.config.auth_timeout == 10

    def test_get_api_key_from_env(self, mock_env_api_key):
        """从环境变量获取 API Key"""
        manager = CloudAuthManager()
        key = manager.get_api_key()
        assert key == mock_env_api_key

    def test_get_api_key_from_config(self, auth_config):
        """从配置获取 API Key"""
        # 确保环境变量不干扰
        with patch.dict(os.environ, {}, clear=True):
            # 清除 CURSOR_API_KEY
            os.environ.pop("CURSOR_API_KEY", None)
            manager = CloudAuthManager(config=auth_config)
            key = manager.get_api_key()
            assert key == "test-config-api-key"

    def test_get_api_key_priority(self, auth_config, mock_env_api_key):
        """API Key 优先级: 配置对象 api_key > 环境变量 > config.yaml

        配置对象中的 api_key 被视为显式传入参数，优先级最高。
        """
        manager = CloudAuthManager(config=auth_config)
        key = manager.get_api_key()
        # 配置对象中的 api_key 优先（被视为显式传入）
        assert key == "test-config-api-key"

    @pytest.mark.asyncio
    async def test_authenticate_no_api_key(self):
        """无 API Key 时认证失败"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            manager = CloudAuthManager(config=CloudAuthConfig())
            status = await manager.authenticate()
            assert status.authenticated is False
            assert status.error is not None
            assert status.error.code == AuthErrorCode.CONFIG_NOT_FOUND

    @pytest.mark.asyncio
    async def test_authenticate_success(self, auth_config):
        """认证成功"""
        manager = CloudAuthManager(config=auth_config)

        # Mock CLI 验证
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Logged in as: test@example.com\nPlan: pro", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_find_agent_executable", return_value="agent"):
                status = await manager.authenticate()
                assert status.authenticated is True

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, auth_config):
        """认证失败"""
        manager = CloudAuthManager(config=auth_config)

        # Mock CLI 验证失败
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Invalid API key")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_find_agent_executable", return_value="agent"):
                status = await manager.authenticate()
                assert status.authenticated is False
                assert status.error is not None

    @pytest.mark.asyncio
    async def test_authenticate_timeout(self, auth_config):
        """认证超时"""
        manager = CloudAuthManager(config=auth_config)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=asyncio.TimeoutError(),
        ):
            status = await manager.authenticate()
            assert status.authenticated is False

    def test_on_auth_change_callback(self, auth_manager):
        """认证状态变化回调"""
        callback = MagicMock()
        auth_manager.on_auth_change(callback)
        auth_manager._notify_auth_change()
        callback.assert_called_once()

    def test_parse_auth_output_success(self, auth_manager):
        """解析认证输出 - 成功"""
        output = "Logged in as: test@example.com\nUser ID: user-123\nPlan: pro"
        status = auth_manager._parse_auth_output(output)
        assert status.authenticated is True
        assert "test@example.com" in (status.email or "")

    def test_parse_auth_output_not_logged_in(self, auth_manager):
        """解析认证输出 - 未登录"""
        output = "Not logged in"
        status = auth_manager._parse_auth_output(output)
        assert status.authenticated is False

    def test_detect_error_code_invalid_key(self, auth_manager):
        """检测错误代码 - 无效 Key"""
        code = auth_manager._detect_error_code("Invalid API key provided")
        assert code == AuthErrorCode.INVALID_API_KEY

    def test_detect_error_code_rate_limited(self, auth_manager):
        """检测错误代码 - 限流"""
        code = auth_manager._detect_error_code("Rate limit exceeded")
        assert code == AuthErrorCode.RATE_LIMITED

    def test_detect_error_code_network(self, auth_manager):
        """检测错误代码 - 网络错误"""
        code = auth_manager._detect_error_code("Network connection failed")
        assert code == AuthErrorCode.NETWORK_ERROR


# ========== TaskStatus 测试 ==========


class TestTaskStatus:
    """TaskStatus 测试"""

    def test_status_values(self):
        """状态值"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


# ========== TaskResult 测试 ==========


class TestTaskResult:
    """TaskResult 测试"""

    def test_is_success(self):
        """成功判断"""
        result = TaskResult(task_id="t1", status=TaskStatus.COMPLETED)
        assert result.is_success is True

        result_failed = TaskResult(
            task_id="t2", status=TaskStatus.COMPLETED, error="error"
        )
        assert result_failed.is_success is False

    def test_is_terminal(self):
        """终态判断"""
        for status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        ]:
            result = TaskResult(task_id="t", status=status)
            assert result.is_terminal is True

        for status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.QUEUED]:
            result = TaskResult(task_id="t", status=status)
            assert result.is_terminal is False

    def test_to_dict(self):
        """转字典"""
        result = TaskResult(
            task_id="task-123",
            status=TaskStatus.COMPLETED,
            result="success output",
            duration_ms=1234,
        )
        data = result.to_dict()
        assert data["task_id"] == "task-123"
        assert data["status"] == "completed"
        assert data["result"] == "success output"
        assert data["duration_ms"] == 1234


# ========== CloudTask 测试 ==========


class TestCloudTask:
    """CloudTask 测试"""

    def test_is_running(self):
        """运行中判断"""
        for status in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING]:
            task = CloudTask(task_id="t", status=status)
            assert task.is_running is True

    def test_is_completed(self):
        """完成判断"""
        task = CloudTask(task_id="t", status=TaskStatus.COMPLETED)
        assert task.is_completed is True
        assert task.is_success is True

    def test_duration(self):
        """耗时计算"""
        task = CloudTask(task_id="t")
        task.started_at = datetime.now() - timedelta(seconds=10)
        task.completed_at = datetime.now()
        assert task.duration is not None
        assert 9 <= task.duration <= 11

    def test_to_dict_and_from_dict(self):
        """序列化和反序列化"""
        options = CloudTaskOptions(model="gpt-5", timeout=300)
        task = CloudTask(
            task_id="task-abc",
            status=TaskStatus.RUNNING,
            prompt="test prompt",
            options=options,
        )

        data = task.to_dict()
        restored = CloudTask.from_dict(data)

        assert restored.task_id == "task-abc"
        assert restored.status == TaskStatus.RUNNING
        assert restored.prompt == "test prompt"
        assert restored.options.model == "gpt-5"


# ========== CloudTaskClient 测试 ==========


class TestCloudTaskClient:
    """CloudTaskClient 测试"""

    def test_init(self, cloud_task_client):
        """初始化"""
        assert cloud_task_client.auth_manager is not None
        assert cloud_task_client.api_base_url == "https://api.cursor.com"

    @pytest.mark.asyncio
    async def test_poll_task_status_timeout(self, cloud_task_client):
        """轮询超时"""
        # Mock 查询总是返回 RUNNING
        with patch.object(
            cloud_task_client,
            "_query_task_status",
            return_value=TaskResult(task_id="t", status=TaskStatus.RUNNING),
        ):
            result = await cloud_task_client.poll_task_status(
                "test-task",
                timeout=0.1,
                interval=0.05,
            )
            assert result.status == TaskStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_poll_task_status_success(self, cloud_task_client):
        """轮询成功"""
        call_count = 0

        async def mock_query(task_id):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result="done",
                )
            return TaskResult(task_id=task_id, status=TaskStatus.RUNNING)

        with patch.object(cloud_task_client, "_query_task_status", side_effect=mock_query):
            result = await cloud_task_client.poll_task_status(
                "test-task",
                timeout=5.0,
                interval=0.1,
            )
            assert result.status == TaskStatus.COMPLETED
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_poll_task_status_callback(self, cloud_task_client):
        """轮询状态回调"""
        status_changes = []

        async def mock_query(task_id):
            if len(status_changes) < 2:
                return TaskResult(task_id=task_id, status=TaskStatus.RUNNING)
            return TaskResult(task_id=task_id, status=TaskStatus.COMPLETED)

        with patch.object(cloud_task_client, "_query_task_status", side_effect=mock_query):
            await cloud_task_client.poll_task_status(
                "test-task",
                timeout=5.0,
                interval=0.1,
                on_status_change=lambda s: status_changes.append(s),
            )
            assert len(status_changes) >= 1

    def test_parse_task_status_output(self, cloud_task_client):
        """解析任务状态输出"""
        result = cloud_task_client._parse_task_status_output(
            "task-1", "Task completed successfully"
        )
        assert result.status == TaskStatus.COMPLETED

        result = cloud_task_client._parse_task_status_output("task-2", "Task failed with error")
        assert result.status == TaskStatus.FAILED

    def test_parse_task_api_response(self, cloud_task_client):
        """解析 API 响应"""
        data = {
            "status": "completed",
            "result": "output text",
            "duration_ms": 500,
        }
        result = cloud_task_client._parse_task_api_response("task-1", data)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "output text"
        assert result.duration_ms == 500


# ========== CursorCloudClient 测试 ==========


class TestCursorCloudClient:
    """CursorCloudClient 测试"""

    def test_is_cloud_request(self):
        """检测云端请求"""
        assert CursorCloudClient.is_cloud_request("& test prompt") is True
        assert CursorCloudClient.is_cloud_request("  & test prompt") is True
        assert CursorCloudClient.is_cloud_request("test prompt") is False

    def test_strip_cloud_prefix(self):
        """移除云端前缀"""
        assert CursorCloudClient.strip_cloud_prefix("& test") == "test"
        assert CursorCloudClient.strip_cloud_prefix("  &  test  ") == "test"
        assert CursorCloudClient.strip_cloud_prefix("no prefix") == "no prefix"

    @pytest.mark.asyncio
    async def test_submit_task_not_authenticated(self, cursor_cloud_client):
        """未认证时提交任务失败"""
        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(
                authenticated=False,
                error=AuthError("未认证", AuthErrorCode.INVALID_API_KEY),
            ),
        ):
            result = await cursor_cloud_client.submit_task("test prompt")
            assert result.success is False
            # 检查错误消息包含相关关键字
            assert any(kw in result.error for kw in ["认证", "未", "API Key", "无效"])

    @pytest.mark.asyncio
    async def test_submit_task_success(self, cursor_cloud_client):
        """提交任务成功"""
        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch.object(
                cursor_cloud_client.auth_manager,
                "get_api_key",
                return_value="test-key",
            ):
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate = AsyncMock(
                    return_value=(
                        b'{"task_id": "task-123", "status": "queued"}',
                        b"",
                    )
                )

                with patch(
                    "asyncio.create_subprocess_exec",
                    return_value=mock_process,
                ):
                    result = await cursor_cloud_client.submit_task("test prompt")
                    assert result.success is True
                    assert result.task is not None
                    assert result.task.task_id == "task-123"

    def test_parse_task_id_json(self, cursor_cloud_client):
        """解析 JSON 格式的 task_id"""
        output = '{"task_id": "abc-123"}'
        task_id = cursor_cloud_client._parse_task_id(output)
        assert task_id == "abc-123"

    def test_parse_task_id_session_id(self, cursor_cloud_client):
        """解析 session_id 格式"""
        output = '{"session_id": "sess-456"}'
        task_id = cursor_cloud_client._parse_task_id(output)
        assert task_id == "sess-456"

    def test_parse_task_id_uuid(self, cursor_cloud_client):
        """解析 UUID 格式"""
        output = "Task started: 550e8400-e29b-41d4-a716-446655440000"
        task_id = cursor_cloud_client._parse_task_id(output)
        assert task_id == "550e8400-e29b-41d4-a716-446655440000"

    @pytest.mark.asyncio
    async def test_get_task_status_cached(self, cursor_cloud_client):
        """获取缓存的任务状态"""
        # 预先添加任务到缓存
        task = CloudTask(
            task_id="cached-task",
            status=TaskStatus.COMPLETED,
            output="cached result",
        )
        cursor_cloud_client._tasks["cached-task"] = task

        result = await cursor_cloud_client.get_task_status("cached-task")
        assert result is not None
        assert result.task_id == "cached-task"
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_task(self, cursor_cloud_client):
        """取消任务"""
        task = CloudTask(task_id="to-cancel", status=TaskStatus.RUNNING)
        cursor_cloud_client._tasks["to-cancel"] = task

        success = await cursor_cloud_client.cancel_task("to-cancel")
        assert success is True
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, cursor_cloud_client):
        """无法取消已完成的任务"""
        task = CloudTask(task_id="completed", status=TaskStatus.COMPLETED)
        cursor_cloud_client._tasks["completed"] = task

        success = await cursor_cloud_client.cancel_task("completed")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_tasks(self, cursor_cloud_client):
        """列出任务"""
        cursor_cloud_client._tasks = {
            "t1": CloudTask(task_id="t1", status=TaskStatus.COMPLETED),
            "t2": CloudTask(task_id="t2", status=TaskStatus.RUNNING),
            "t3": CloudTask(task_id="t3", status=TaskStatus.COMPLETED),
        }

        # 列出所有任务
        all_tasks = await cursor_cloud_client.list_tasks()
        assert len(all_tasks) == 3

        # 按状态过滤
        completed = await cursor_cloud_client.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 2

    def test_parse_task_from_json_output_array(self, cursor_cloud_client):
        """从 JSON 数组格式解析任务"""
        output = json.dumps([
            {"session_id": "task-123", "status": "completed", "prompt": "test prompt"},
            {"session_id": "task-456", "status": "running", "prompt": "other prompt"},
        ])
        task = cursor_cloud_client._parse_task_from_json_output("task-123", output)
        assert task is not None
        assert task.task_id == "task-123"
        assert task.status == TaskStatus.COMPLETED
        assert task.prompt == "test prompt"

    def test_parse_task_from_json_output_object(self, cursor_cloud_client):
        """从 JSON 对象格式解析任务"""
        output = json.dumps({
            "sessions": [
                {"task_id": "task-789", "status": "running", "message": "doing work"},
            ]
        })
        task = cursor_cloud_client._parse_task_from_json_output("task-789", output)
        assert task is not None
        assert task.task_id == "task-789"
        assert task.status == TaskStatus.RUNNING

    def test_parse_task_from_json_output_ndjson(self, cursor_cloud_client):
        """从 NDJSON 格式解析任务"""
        output = '{"id": "task-111", "status": "failed", "error": "something wrong"}\n{"id": "task-222", "status": "completed"}'
        task = cursor_cloud_client._parse_task_from_json_output("task-111", output)
        assert task is not None
        assert task.task_id == "task-111"
        assert task.status == TaskStatus.FAILED

    def test_parse_task_from_json_output_not_found(self, cursor_cloud_client):
        """JSON 输出中未找到任务"""
        output = json.dumps([{"session_id": "other-task", "status": "running"}])
        task = cursor_cloud_client._parse_task_from_json_output("task-xxx", output)
        assert task is None

    def test_parse_task_from_text_output_table(self, cursor_cloud_client):
        """从表格格式文本解析任务"""
        output = """ID | STATUS | PROMPT
task-abc | running | implement feature
task-def | completed | fix bug"""
        task = cursor_cloud_client._parse_task_from_text_output("task-abc", output)
        assert task is not None
        assert task.task_id == "task-abc"
        assert task.status == TaskStatus.RUNNING

    def test_parse_task_from_text_output_status_keywords(self, cursor_cloud_client):
        """从文本解析各种状态关键词"""
        # Completed
        output = "task-a: completed successfully"
        task = cursor_cloud_client._parse_task_from_text_output("task-a", output)
        assert task.status == TaskStatus.COMPLETED

        # Failed
        output = "task-b: failed with error"
        task = cursor_cloud_client._parse_task_from_text_output("task-b", output)
        assert task.status == TaskStatus.FAILED

        # Cancelled
        output = "task-c: cancelled by user"
        task = cursor_cloud_client._parse_task_from_text_output("task-c", output)
        assert task.status == TaskStatus.CANCELLED

    def test_parse_task_from_text_output_not_found(self, cursor_cloud_client):
        """文本输出中未找到任务"""
        output = "task-other: running - some work"
        task = cursor_cloud_client._parse_task_from_text_output("task-missing", output)
        assert task is None

    def test_create_task_from_dict(self, cursor_cloud_client):
        """从字典创建任务"""
        data = {
            "status": "completed",
            "prompt": "test task",
            "output": "task output",
            "files_modified": ["file1.py", "file2.py"],
            "progress": 100,
        }
        task = cursor_cloud_client._create_task_from_dict("task-dict", data)
        assert task.task_id == "task-dict"
        assert task.status == TaskStatus.COMPLETED
        assert task.prompt == "test task"
        assert task.output == "task output"
        assert task.files_modified == ["file1.py", "file2.py"]
        assert task.progress == 100

    def test_create_task_from_dict_status_mapping(self, cursor_cloud_client):
        """从字典创建任务 - 状态映射"""
        # success -> COMPLETED
        data = {"status": "success"}
        task = cursor_cloud_client._create_task_from_dict("t1", data)
        assert task.status == TaskStatus.COMPLETED

        # error -> FAILED
        data = {"status": "error"}
        task = cursor_cloud_client._create_task_from_dict("t2", data)
        assert task.status == TaskStatus.FAILED

        # unknown -> RUNNING (default)
        data = {"status": "unknown_status"}
        task = cursor_cloud_client._create_task_from_dict("t3", data)
        assert task.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_fetch_task_from_cloud_empty_id(self, cursor_cloud_client):
        """空任务 ID 应返回 None"""
        task = await cursor_cloud_client._fetch_task_from_cloud("")
        assert task is None

        task = await cursor_cloud_client._fetch_task_from_cloud(None)
        assert task is None


# ========== ExecutionMode 测试 ==========


class TestExecutionMode:
    """ExecutionMode 测试"""

    def test_mode_values(self):
        """模式值"""
        assert ExecutionMode.CLI.value == "cli"
        assert ExecutionMode.CLOUD.value == "cloud"
        assert ExecutionMode.AUTO.value == "auto"
        assert ExecutionMode.PLAN.value == "plan"
        assert ExecutionMode.ASK.value == "ask"

    def test_mode_from_string(self):
        """从字符串创建模式"""
        assert ExecutionMode("cli") == ExecutionMode.CLI
        assert ExecutionMode("cloud") == ExecutionMode.CLOUD
        assert ExecutionMode("auto") == ExecutionMode.AUTO
        assert ExecutionMode("plan") == ExecutionMode.PLAN
        assert ExecutionMode("ask") == ExecutionMode.ASK

    def test_mode_is_string_enum(self):
        """ExecutionMode 是字符串枚举"""
        assert isinstance(ExecutionMode.CLI, str)
        assert ExecutionMode.PLAN == "plan"
        assert ExecutionMode.ASK == "ask"


# ========== AgentExecutorFactory 测试 ==========


class TestAgentExecutorFactory:
    """AgentExecutorFactory 测试"""

    def test_create_cli_executor(self):
        """创建 CLI 执行器"""
        executor = AgentExecutorFactory.create(mode=ExecutionMode.CLI)
        assert isinstance(executor, CLIAgentExecutor)
        assert executor.executor_type == "cli"

    def test_create_cloud_executor(self):
        """创建 Cloud 执行器"""
        executor = AgentExecutorFactory.create(mode=ExecutionMode.CLOUD)
        assert isinstance(executor, CloudAgentExecutor)
        assert executor.executor_type == "cloud"

    def test_create_auto_executor(self):
        """创建自动选择执行器"""
        executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)
        assert isinstance(executor, AutoAgentExecutor)
        assert executor.executor_type == "auto"

    def test_create_with_config(self):
        """使用配置创建执行器"""
        from cursor.client import CursorAgentConfig

        config = CursorAgentConfig(model="gpt-5.2-high", timeout=120)
        executor = AgentExecutorFactory.create(
            mode=ExecutionMode.CLI,
            cli_config=config,
        )
        assert isinstance(executor, CLIAgentExecutor)
        assert executor.config.model == "gpt-5.2-high"
        assert executor.config.timeout == 120

    def test_create_plan_executor(self):
        """创建规划模式执行器"""
        executor = AgentExecutorFactory.create(mode=ExecutionMode.PLAN)
        assert isinstance(executor, PlanAgentExecutor)
        assert executor.executor_type == "plan"
        assert executor.config.mode == "plan"
        assert executor.config.force_write is False

    def test_create_ask_executor(self):
        """创建问答模式执行器"""
        executor = AgentExecutorFactory.create(mode=ExecutionMode.ASK)
        assert isinstance(executor, AskAgentExecutor)
        assert executor.executor_type == "ask"
        assert executor.config.mode == "ask"
        assert executor.config.force_write is False

    def test_create_invalid_mode(self):
        """无效模式抛出异常"""
        with pytest.raises(ValueError):
            AgentExecutorFactory.create(mode="invalid")


# ========== CLIAgentExecutor 测试 ==========


class TestCLIAgentExecutor:
    """CLIAgentExecutor 测试"""

    def test_init(self):
        """初始化"""
        executor = CLIAgentExecutor()
        assert executor.executor_type == "cli"
        assert executor.config is not None

    def test_init_with_mode(self):
        """使用 mode 参数初始化"""
        executor = CLIAgentExecutor(mode="plan")
        assert executor.cli_mode == "plan"
        assert executor.config.mode == "plan"

    def test_init_mode_override_config(self):
        """mode 参数应该覆盖 config 中的 mode"""
        from cursor.client import CursorAgentConfig

        config = CursorAgentConfig(mode="agent")
        executor = CLIAgentExecutor(config=config, mode="plan")
        assert executor.cli_mode == "plan"
        assert executor.config.mode == "plan"

    @pytest.mark.asyncio
    async def test_execute(self):
        """执行任务"""
        executor = CLIAgentExecutor()

        # Mock CursorAgentClient.execute
        from cursor.client import CursorAgentResult

        mock_result = CursorAgentResult(
            success=True,
            output="test output",
            exit_code=0,
        )

        with patch.object(executor._client, "execute", return_value=mock_result):
            result = await executor.execute(prompt="test prompt")
            assert result.success is True
            assert result.output == "test output"
            assert result.executor_type == "cli"

    def test_check_available_sync(self):
        """同步检查可用性"""
        executor = CLIAgentExecutor()
        with patch.object(executor._client, "check_agent_available", return_value=True):
            assert executor.check_available_sync() is True


# ========== CloudAgentExecutor 测试 ==========


class TestCloudAgentExecutor:
    """CloudAgentExecutor 测试"""

    def test_init(self):
        """初始化"""
        executor = CloudAgentExecutor()
        assert executor.executor_type == "cloud"
        assert executor.auth_manager is not None

    @pytest.mark.asyncio
    async def test_execute_not_authenticated(self):
        """未认证时执行失败"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.cloud.client import CloudAgentResult

        executor = CloudAgentExecutor()

        # Mock CloudClientFactory.execute_task 返回认证失败
        mock_result = CloudAgentResult(
            success=False,
            error="Cloud 认证失败: 未认证",
        )

        with patch.object(
            CloudClientFactory,
            "execute_task",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await executor.execute(prompt="test")
            assert result.success is False
            assert "认证" in result.error or "未认证" in result.error

    @pytest.mark.asyncio
    async def test_check_available(self):
        """检查可用性"""
        executor = CloudAgentExecutor()

        # 认证成功时，Cloud API 可用
        with patch.object(
            executor._auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            available = await executor.check_available()
            # 认证成功即表示可用（使用 CursorCloudClient）
            assert available is True

        # 认证失败时，Cloud API 不可用
        executor2 = CloudAgentExecutor()
        with patch.object(
            executor2._auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=False, error="No API key"),
        ):
            available2 = await executor2.check_available()
            assert available2 is False

    @pytest.mark.asyncio
    async def test_execute_background_true_passes_wait_false(self):
        """测试 background=True 时，调用链传递 wait=False 并返回 session_id

        验证:
        1. CloudClientFactory.execute_task 被调用时 wait=False
        2. 返回结果包含 session_id
        3. 返回结果 files_modified 为空（后台任务尚未完成）
        """
        from cursor.cloud_client import CloudClientFactory
        from cursor.cloud.client import CloudAgentResult
        from cursor.cloud.task import CloudTask, TaskStatus

        executor = CloudAgentExecutor()

        # 创建 mock 的 CloudAgentResult（后台模式返回）
        mock_task = CloudTask(
            task_id="session-background-123",
            status=TaskStatus.QUEUED,
            prompt="后台任务",
        )
        mock_cloud_result = CloudAgentResult(
            success=True,
            output="任务已提交",
            task=mock_task,
        )

        with patch.object(
            CloudClientFactory,
            "execute_task",
            new_callable=AsyncMock,
            return_value=mock_cloud_result,
        ) as mock_execute:
            result = await executor.execute(
                prompt="后台执行任务",
                background=True,  # 关键：后台模式
            )

            # 验证 CloudClientFactory.execute_task 被调用时 wait=False
            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("wait") is False, (
                "background=True 时应传递 wait=False"
            )

            # 验证返回结果包含 session_id
            assert result.session_id == "session-background-123", (
                "返回结果应包含 session_id"
            )

            # 验证后台模式下 files_modified 为空（任务尚未完成）
            assert result.files_modified == [], (
                "后台模式下 files_modified 应为空"
            )

            # 验证 success 表示提交是否成功
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_background_false_passes_wait_true(self):
        """测试 background=False（默认）时，调用链传递 wait=True

        验证:
        1. CloudClientFactory.execute_task 被调用时 wait=True
        2. 返回完整的执行结果，包含 output 和 files_modified
        """
        from cursor.cloud_client import CloudClientFactory
        from cursor.cloud.client import CloudAgentResult
        from cursor.cloud.task import CloudTask, TaskStatus

        executor = CloudAgentExecutor()

        # 创建 mock 的 CloudAgentResult（前台模式返回完整结果）
        mock_task = CloudTask(
            task_id="session-foreground-456",
            status=TaskStatus.COMPLETED,
            prompt="前台任务",
        )
        mock_cloud_result = CloudAgentResult(
            success=True,
            output="执行完成，修改了 main.py",
            task=mock_task,
            files_modified=["main.py", "test.py"],
        )

        with patch.object(
            CloudClientFactory,
            "execute_task",
            new_callable=AsyncMock,
            return_value=mock_cloud_result,
        ) as mock_execute:
            result = await executor.execute(
                prompt="前台执行任务",
                background=False,  # 默认前台模式
            )

            # 验证 CloudClientFactory.execute_task 被调用时 wait=True
            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("wait") is True, (
                "background=False 时应传递 wait=True"
            )

            # 验证返回完整结果
            assert result.session_id == "session-foreground-456"
            assert result.output == "执行完成，修改了 main.py"
            assert result.files_modified == ["main.py", "test.py"]

    @pytest.mark.asyncio
    async def test_execute_default_background_is_false(self):
        """测试 execute 默认 background=False（前台模式）"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.cloud.client import CloudAgentResult
        from cursor.cloud.task import CloudTask, TaskStatus

        executor = CloudAgentExecutor()

        mock_task = CloudTask(
            task_id="session-default-789",
            status=TaskStatus.COMPLETED,
            prompt="默认模式任务",
        )
        mock_cloud_result = CloudAgentResult(
            success=True,
            output="默认模式执行完成",
            task=mock_task,
        )

        with patch.object(
            CloudClientFactory,
            "execute_task",
            new_callable=AsyncMock,
            return_value=mock_cloud_result,
        ) as mock_execute:
            # 不传递 background 参数，应默认为 False
            result = await executor.execute(prompt="默认模式任务")

            # 验证默认使用前台模式（wait=True）
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("wait") is True, (
                "默认应为前台模式（wait=True）"
            )


# ========== PlanAgentExecutor 测试 ==========


class TestPlanAgentExecutor:
    """PlanAgentExecutor 测试"""

    def test_init(self):
        """初始化"""
        executor = PlanAgentExecutor()
        assert executor.executor_type == "plan"
        assert executor.config is not None
        assert executor.config.mode == "plan"
        assert executor.config.force_write is False

    def test_init_with_config(self):
        """使用配置初始化"""
        from cursor.client import CursorAgentConfig

        config = CursorAgentConfig(model="gpt-5.2-high", timeout=180)
        executor = PlanAgentExecutor(config=config)
        assert executor.config.model == "gpt-5.2-high"
        assert executor.config.timeout == 180
        assert executor.config.mode == "plan"
        # force_write 应该被强制设为 False
        assert executor.config.force_write is False

    def test_init_force_write_override(self):
        """force_write 应该被强制覆盖为 False"""
        from cursor.client import CursorAgentConfig

        config = CursorAgentConfig(force_write=True)
        executor = PlanAgentExecutor(config=config)
        assert executor.config.force_write is False

    @pytest.mark.asyncio
    async def test_execute(self):
        """执行任务"""
        executor = PlanAgentExecutor()

        from cursor.client import CursorAgentResult

        mock_result = CursorAgentResult(
            success=True,
            output="plan output",
            exit_code=0,
        )

        with patch.object(executor._cli_executor._client, "execute", return_value=mock_result):
            result = await executor.execute(prompt="分析代码结构")
            assert result.success is True
            assert result.output == "plan output"
            assert result.executor_type == "plan"

    def test_check_available_sync(self):
        """同步检查可用性"""
        executor = PlanAgentExecutor()
        with patch.object(executor._cli_executor._client, "check_agent_available", return_value=True):
            assert executor.check_available_sync() is True


# ========== AskAgentExecutor 测试 ==========


class TestAskAgentExecutor:
    """AskAgentExecutor 测试"""

    def test_init(self):
        """初始化"""
        executor = AskAgentExecutor()
        assert executor.executor_type == "ask"
        assert executor.config is not None
        assert executor.config.mode == "ask"
        assert executor.config.force_write is False

    def test_init_with_config(self):
        """使用配置初始化"""
        from cursor.client import CursorAgentConfig

        config = CursorAgentConfig(model="gpt-5.2-high", timeout=120)
        executor = AskAgentExecutor(config=config)
        assert executor.config.model == "gpt-5.2-high"
        assert executor.config.timeout == 120
        assert executor.config.mode == "ask"
        # force_write 应该被强制设为 False
        assert executor.config.force_write is False

    def test_init_force_write_override(self):
        """force_write 应该被强制覆盖为 False"""
        from cursor.client import CursorAgentConfig

        config = CursorAgentConfig(force_write=True)
        executor = AskAgentExecutor(config=config)
        assert executor.config.force_write is False

    @pytest.mark.asyncio
    async def test_execute(self):
        """执行任务"""
        executor = AskAgentExecutor()

        from cursor.client import CursorAgentResult

        mock_result = CursorAgentResult(
            success=True,
            output="ask output",
            exit_code=0,
        )

        with patch.object(executor._cli_executor._client, "execute", return_value=mock_result):
            result = await executor.execute(prompt="这段代码是什么意思？")
            assert result.success is True
            assert result.output == "ask output"
            assert result.executor_type == "ask"

    def test_check_available_sync(self):
        """同步检查可用性"""
        executor = AskAgentExecutor()
        with patch.object(executor._cli_executor._client, "check_agent_available", return_value=True):
            assert executor.check_available_sync() is True


# ========== AutoAgentExecutor 测试 ==========


class TestAutoAgentExecutor:
    """AutoAgentExecutor 测试"""

    def test_init(self):
        """初始化"""
        executor = AutoAgentExecutor()
        assert executor.executor_type == "auto"
        assert executor.cli_executor is not None
        assert executor.cloud_executor is not None

    def test_init_with_cooldown_config(self):
        """使用冷却配置初始化"""
        executor = AutoAgentExecutor(
            cloud_cooldown_seconds=120,
            enable_cooldown=True,
        )
        assert executor.cloud_cooldown_seconds == 120
        assert executor._enable_cooldown is True
        assert executor.is_cloud_in_cooldown is False

    def test_init_cooldown_disabled(self):
        """禁用冷却策略"""
        executor = AutoAgentExecutor(enable_cooldown=False)
        assert executor._enable_cooldown is False
        # 即使设置冷却时间也不生效
        from datetime import datetime, timedelta
        executor._cloud_cooldown_until = datetime.now() + timedelta(hours=1)
        assert executor.is_cloud_in_cooldown is False

    @pytest.mark.asyncio
    async def test_fallback_to_cli(self):
        """Cloud 不可用时回退到 CLI"""
        executor = AutoAgentExecutor()

        # Mock Cloud 不可用
        with patch.object(executor._cloud_executor, "check_available", return_value=False):
            with patch.object(executor._cli_executor, "check_available", return_value=True):
                # Mock CLI 执行
                from cursor.client import CursorAgentResult

                mock_result = CursorAgentResult(
                    success=True,
                    output="cli output",
                    exit_code=0,
                )
                with patch.object(
                    executor._cli_executor._client, "execute", return_value=mock_result
                ):
                    result = await executor.execute(prompt="test")
                    assert result.success is True
                    assert result.executor_type == "cli"

    def test_reset_preference(self):
        """重置执行器偏好"""
        executor = AutoAgentExecutor()
        executor._preferred_executor = executor._cli_executor
        executor.reset_preference()
        assert executor._preferred_executor is None

    def test_reset_cooldown(self):
        """重置冷却状态"""
        from datetime import datetime, timedelta
        executor = AutoAgentExecutor()

        # 设置冷却状态
        executor._cloud_cooldown_until = datetime.now() + timedelta(hours=1)
        executor._cloud_failure_count = 3
        executor._preferred_executor = executor._cli_executor

        # 重置冷却
        with patch.object(executor._cloud_executor, "reset_availability_cache"):
            executor.reset_cooldown()

        # 验证所有状态已重置
        assert executor._cloud_cooldown_until is None
        assert executor._cloud_failure_count == 0
        assert executor._preferred_executor is None

    def test_cooldown_properties(self):
        """测试冷却相关属性"""
        from datetime import datetime, timedelta
        executor = AutoAgentExecutor(cloud_cooldown_seconds=60)

        # 初始状态
        assert executor.is_cloud_in_cooldown is False
        assert executor.cloud_cooldown_remaining is None

        # 设置冷却
        executor._cloud_cooldown_until = datetime.now() + timedelta(seconds=30)
        assert executor.is_cloud_in_cooldown is True
        remaining = executor.cloud_cooldown_remaining
        assert remaining is not None
        assert 0 < remaining <= 30

    @pytest.mark.asyncio
    async def test_cloud_failure_triggers_cooldown(self):
        """Cloud 失败触发冷却"""
        executor = AutoAgentExecutor(cloud_cooldown_seconds=60)

        cloud_fail_result = AgentResult(
            success=False,
            error="Cloud 失败",
            executor_type="cloud",
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI 成功",
            executor_type="cli",
        )

        with patch.object(
            executor._cloud_executor, "check_available",
            new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                executor._cloud_executor, "execute",
                new_callable=AsyncMock, return_value=cloud_fail_result
            ):
                with patch.object(
                    executor._cli_executor, "execute",
                    new_callable=AsyncMock, return_value=cli_success_result
                ):
                    result = await executor.execute(prompt="测试")

                    # 验证回退到 CLI
                    assert result.executor_type == "cli"
                    # 验证冷却已启动
                    assert executor.is_cloud_in_cooldown is True
                    assert executor._cloud_failure_count == 1

    @pytest.mark.asyncio
    async def test_cloud_success_resets_cooldown(self):
        """Cloud 成功重置冷却"""
        executor = AutoAgentExecutor()
        executor._cloud_failure_count = 2

        cloud_success_result = AgentResult(
            success=True,
            output="Cloud 成功",
            executor_type="cloud",
        )

        with patch.object(
            executor._cloud_executor, "check_available",
            new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                executor._cloud_executor, "execute",
                new_callable=AsyncMock, return_value=cloud_success_result
            ):
                await executor.execute(prompt="测试")

                # 验证冷却已重置
                assert executor._cloud_failure_count == 0
                assert executor._cloud_cooldown_until is None


# ========== 便捷函数测试 ==========


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_get_api_key_from_env(self, mock_env_api_key):
        """get_api_key 从环境变量获取"""
        key = get_api_key()
        assert key == mock_env_api_key

    @pytest.mark.asyncio
    async def test_require_auth_decorator_success(self, mock_env_api_key):
        """require_auth 装饰器 - 成功"""

        @require_auth
        async def test_func():
            return "success"

        with patch(
            "cursor.cloud_client.CloudAuthManager.authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            result = await test_func()
            assert result == "success"

    @pytest.mark.asyncio
    async def test_require_auth_decorator_failure(self):
        """require_auth 装饰器 - 失败"""

        @require_auth
        async def test_func():
            return "success"

        with patch(
            "cursor.cloud_client.CloudAuthManager.authenticate",
            return_value=AuthStatus(
                authenticated=False,
                error=AuthError("未认证", AuthErrorCode.INVALID_API_KEY),
            ),
        ):
            with pytest.raises(AuthError):
                await test_func()


# ========== 错误处理测试 ==========


class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """限流错误"""
        manager = CloudAuthManager()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Rate limit exceeded")
        )

        # 需要模拟 API key 存在，否则 authenticate 会先返回 CONFIG_NOT_FOUND
        with patch.object(manager, "get_api_key", return_value="test-key"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                status = await manager.authenticate()
                assert status.authenticated is False
                if status.error:
                    assert status.error.code == AuthErrorCode.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_network_error(self):
        """网络错误"""
        manager = CloudAuthManager()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("Network unreachable"),
        ):
            status = await manager.authenticate()
            assert status.authenticated is False

    @pytest.mark.asyncio
    async def test_cli_not_found_error(self):
        """CLI 找不到错误"""
        manager = CloudAuthManager()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("agent not found"),
        ):
            status = await manager.authenticate()
            assert status.authenticated is False


# ========== 重试机制测试 ==========


class TestRetryMechanism:
    """重试机制测试"""

    @pytest.mark.asyncio
    async def test_cli_executor_retry(self):
        """CLI 执行器重试"""
        executor = CLIAgentExecutor()

        from cursor.client import CursorAgentResult

        mock_result = CursorAgentResult(
            success=True,
            output="retry success",
            exit_code=0,
        )

        with patch.object(
            executor._client,
            "execute_with_retry",
            return_value=mock_result,
        ):
            result = await executor.execute_with_retry(
                prompt="test",
                max_retries=3,
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_auto_executor_cloud_failure_retry_cli(self):
        """自动执行器 Cloud 失败后重试 CLI"""
        executor = AutoAgentExecutor()

        # 首次选择 Cloud，但执行失败
        cloud_fail_result = AgentResult(
            success=False,
            error="Cloud API failed",
            executor_type="cloud",
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI success",
            executor_type="cli",
        )

        with patch.object(executor._cloud_executor, "check_available", return_value=True):
            with patch.object(
                executor._cloud_executor, "execute", return_value=cloud_fail_result
            ):
                with patch.object(
                    executor._cli_executor, "execute", return_value=cli_success_result
                ):
                    result = await executor.execute(prompt="test")
                    # 应该回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"


# ========== CloudAgentConfig 解析与注入测试 ==========


class TestCloudAgentConfigParsing:
    """测试 cloud_agent.api_base_url/max_retries 的解析与注入

    验证 CLI/env/config 的优先级:
    1. CLI 参数（最高）
    2. 环境变量
    3. config.yaml
    4. 代码默认值（最低）
    """

    def test_cloud_auth_config_default_api_base_url(self):
        """测试 CloudAuthConfig 默认 api_base_url"""
        config = CloudAuthConfig()
        assert config.api_base_url == "https://api.cursor.com"

    def test_cloud_auth_config_default_max_retries(self):
        """测试 CloudAuthConfig 默认 max_retries"""
        config = CloudAuthConfig()
        assert config.max_retries == 3

    def test_cloud_auth_config_custom_api_base_url(self):
        """测试自定义 api_base_url"""
        custom_url = "https://custom.api.example.com"
        config = CloudAuthConfig(api_base_url=custom_url)
        assert config.api_base_url == custom_url

    def test_cloud_auth_config_custom_max_retries(self):
        """测试自定义 max_retries"""
        config = CloudAuthConfig(max_retries=5)
        assert config.max_retries == 5

    def test_cloud_task_client_inherits_api_base_url(self):
        """测试 CloudTaskClient 继承 api_base_url"""
        custom_url = "https://custom.cursor.com"
        auth_config = CloudAuthConfig(api_key="test-key", api_base_url=custom_url)
        auth_manager = CloudAuthManager(config=auth_config)
        task_client = CloudTaskClient(
            auth_manager=auth_manager,
            api_base_url=custom_url,
        )

        assert task_client.api_base_url == custom_url

    def test_cloud_task_client_default_api_base_url(self):
        """测试 CloudTaskClient 默认 api_base_url"""
        auth_manager = CloudAuthManager()
        task_client = CloudTaskClient(auth_manager=auth_manager)

        assert task_client.api_base_url == "https://api.cursor.com"


class TestCloudConfigFromYaml:
    """测试从 config.yaml 加载 cloud_agent 配置

    验证 core.config.CloudAgentConfig 能正确加载 config.yaml 中的值
    """

    def test_config_cloud_agent_api_base_url(self):
        """测试从 config.yaml 加载 api_base_url"""
        from core.config import get_config

        config = get_config()
        # 验证 api_base_url 有值（不为空）
        assert config.cloud_agent.api_base_url is not None
        assert len(config.cloud_agent.api_base_url) > 0
        # 验证是有效的 URL 格式
        assert config.cloud_agent.api_base_url.startswith("http")

    def test_config_cloud_agent_max_retries(self):
        """测试从 config.yaml 加载 max_retries"""
        from core.config import get_config

        config = get_config()
        # 验证 max_retries 是正整数
        assert config.cloud_agent.max_retries >= 0
        assert isinstance(config.cloud_agent.max_retries, int)

    def test_config_cloud_agent_timeout(self):
        """测试从 config.yaml 加载 timeout"""
        from core.config import get_config

        config = get_config()
        # 验证 timeout 是正数
        assert config.cloud_agent.timeout > 0

    def test_config_cloud_agent_auth_timeout(self):
        """测试从 config.yaml 加载 auth_timeout"""
        from core.config import get_config

        config = get_config()
        # 验证 auth_timeout 是正数
        assert config.cloud_agent.auth_timeout > 0


class TestCloudConfigPriority:
    """测试 Cloud 配置优先级

    验证优先级正确：
    1. 显式设置的 api_key（最高）
    2. 环境变量 CURSOR_API_KEY
    3. config.yaml
    4. 代码默认值（最低）
    """

    def test_explicit_api_key_highest_priority(self, mock_env_api_key):
        """测试显式设置的 API Key 优先级最高

        当在 CloudAuthConfig 中显式设置 api_key 时，
        即使环境变量存在，也使用显式设置的值。
        """
        auth_config = CloudAuthConfig(api_key="explicit-api-key")
        auth_manager = CloudAuthManager(config=auth_config)

        # 显式设置的 api_key 优先级最高
        api_key = auth_manager.get_api_key()
        assert api_key == "explicit-api-key"

    def test_env_api_key_overrides_config_file_lookup(self, mock_env_api_key):
        """测试环境变量 API Key 优先于 config.yaml 文件查找

        当 CloudAuthConfig 未显式设置 api_key 时，
        环境变量优先于从 config.yaml 文件读取。
        """
        # 不设置 api_key，让它从环境变量或配置文件读取
        auth_config = CloudAuthConfig()
        auth_manager = CloudAuthManager(config=auth_config)

        # 环境变量优先于配置文件查找
        api_key = auth_manager.get_api_key()
        assert api_key == mock_env_api_key

    def test_config_api_key_when_no_env(self):
        """测试无环境变量时使用显式设置的 config API Key"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            auth_config = CloudAuthConfig(api_key="config-only-key")
            auth_manager = CloudAuthManager(config=auth_config)

            api_key = auth_manager.get_api_key()
            assert api_key == "config-only-key"

    def test_cloud_auth_config_from_core_config(self):
        """测试从 core.config 加载 CloudAgentConfig 并注入到 CloudAuthConfig"""
        from core.config import get_config

        config = get_config()

        # 使用 config.yaml 中的值构建 CloudAuthConfig
        auth_config = CloudAuthConfig(
            api_base_url=config.cloud_agent.api_base_url,
            auth_timeout=config.cloud_agent.auth_timeout,
            max_retries=config.cloud_agent.max_retries,
        )

        # 验证值正确注入
        assert auth_config.api_base_url == config.cloud_agent.api_base_url
        assert auth_config.auth_timeout == config.cloud_agent.auth_timeout
        assert auth_config.max_retries == config.cloud_agent.max_retries


class TestCloudTaskClientConfigInjection:
    """测试 CloudTaskClient 配置注入

    验证 api_base_url/max_retries 正确传递到 CloudTaskClient
    """

    def test_task_client_custom_config_injection(self):
        """测试 CloudTaskClient 自定义配置注入"""
        from cursor.cloud.retry import RetryConfig

        custom_api_url = "https://custom.api.com"
        custom_max_retries = 5

        auth_config = CloudAuthConfig(
            api_key="test-key",
            api_base_url=custom_api_url,
        )
        auth_manager = CloudAuthManager(config=auth_config)
        retry_config = RetryConfig(max_retries=custom_max_retries)

        task_client = CloudTaskClient(
            auth_manager=auth_manager,
            api_base_url=custom_api_url,
            retry_config=retry_config,
        )

        assert task_client.api_base_url == custom_api_url
        assert task_client.retry_config.max_retries == custom_max_retries

    def test_task_client_default_config(self):
        """测试 CloudTaskClient 默认配置"""
        auth_manager = CloudAuthManager()
        task_client = CloudTaskClient(auth_manager=auth_manager)

        # 验证默认值
        assert task_client.api_base_url == "https://api.cursor.com"
        assert task_client.retry_config is not None
        assert task_client.retry_config.max_retries == 3  # RetryConfig 默认值


class TestCloudExecutorConfigInjection:
    """测试 CloudAgentExecutor 配置注入

    验证执行器正确使用配置值
    """

    def test_cloud_executor_uses_auth_manager(self):
        """测试 CloudAgentExecutor 使用认证管理器"""
        executor = CloudAgentExecutor()
        assert executor.auth_manager is not None
        assert isinstance(executor.auth_manager, CloudAuthManager)

    def test_cloud_executor_custom_auth_config(self):
        """测试 CloudAgentExecutor 使用自定义认证配置

        CloudAgentExecutor 接受 auth_config 参数（不是 auth_manager）
        """
        custom_config = CloudAuthConfig(
            api_key="custom-api-key",
            api_base_url="https://custom.api.com",
            max_retries=5,
        )
        executor = CloudAgentExecutor(auth_config=custom_config)

        # 验证认证配置被正确注入
        assert executor._auth_config.api_base_url == "https://custom.api.com"
        assert executor._auth_config.max_retries == 5
        assert executor._auth_config.api_key == "custom-api-key"


# ========== CURSOR_CLOUD_API_KEY 环境变量测试 ==========


class TestCursorCloudApiKeyEnvVar:
    """测试 CURSOR_CLOUD_API_KEY 环境变量支持

    验证优先级：
    1. 显式参数
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
    4. config.yaml
    """

    def test_cloud_auth_manager_uses_cursor_cloud_api_key(self):
        """测试 CloudAuthManager 支持 CURSOR_CLOUD_API_KEY 环境变量"""
        with patch.dict(os.environ, {}, clear=True):
            # 只设置 CURSOR_CLOUD_API_KEY
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ["CURSOR_CLOUD_API_KEY"] = "cloud-api-key-from-env"

            manager = CloudAuthManager(config=CloudAuthConfig())
            api_key = manager.get_api_key()

            assert api_key == "cloud-api-key-from-env"

    def test_cursor_api_key_priority_over_cursor_cloud_api_key(self):
        """测试 CURSOR_API_KEY 优先于 CURSOR_CLOUD_API_KEY"""
        with patch.dict(os.environ, {}, clear=True):
            # 同时设置两个环境变量
            os.environ["CURSOR_API_KEY"] = "cursor-api-key"
            os.environ["CURSOR_CLOUD_API_KEY"] = "cursor-cloud-api-key"

            manager = CloudAuthManager(config=CloudAuthConfig())
            api_key = manager.get_api_key()

            # CURSOR_API_KEY 应该优先
            assert api_key == "cursor-api-key"

    def test_config_api_key_priority_over_env(self):
        """测试配置中的 api_key 优先于所有环境变量"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ["CURSOR_API_KEY"] = "cursor-api-key"
            os.environ["CURSOR_CLOUD_API_KEY"] = "cursor-cloud-api-key"

            config = CloudAuthConfig(api_key="config-api-key")
            manager = CloudAuthManager(config=config)
            api_key = manager.get_api_key()

            # 配置中的 api_key 应该优先
            assert api_key == "config-api-key"

    def test_cloud_client_factory_resolve_api_key_cursor_cloud_api_key(self):
        """测试 CloudClientFactory.resolve_api_key 支持 CURSOR_CLOUD_API_KEY"""
        from cursor.cloud_client import CloudClientFactory

        with patch.dict(os.environ, {}, clear=True):
            # 只设置 CURSOR_CLOUD_API_KEY
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ["CURSOR_CLOUD_API_KEY"] = "factory-cloud-api-key"

            api_key = CloudClientFactory.resolve_api_key()

            assert api_key == "factory-cloud-api-key"

    def test_cloud_client_factory_resolve_api_key_priority(self):
        """测试 CloudClientFactory.resolve_api_key 优先级"""
        from cursor.cloud_client import CloudClientFactory

        with patch.dict(os.environ, {}, clear=True):
            os.environ["CURSOR_API_KEY"] = "cursor-api-key"
            os.environ["CURSOR_CLOUD_API_KEY"] = "cursor-cloud-api-key"

            # 显式参数优先级最高
            api_key = CloudClientFactory.resolve_api_key(
                explicit_api_key="explicit-api-key"
            )
            assert api_key == "explicit-api-key"

            # CURSOR_API_KEY 优先于 CURSOR_CLOUD_API_KEY
            api_key = CloudClientFactory.resolve_api_key()
            assert api_key == "cursor-api-key"

            # 清除 CURSOR_API_KEY，CURSOR_CLOUD_API_KEY 应生效
            os.environ.pop("CURSOR_API_KEY", None)
            api_key = CloudClientFactory.resolve_api_key()
            assert api_key == "cursor-cloud-api-key"

    def test_build_cloud_client_config_cursor_cloud_api_key(self):
        """测试 build_cloud_client_config 支持 CURSOR_CLOUD_API_KEY"""
        from core.config import build_cloud_client_config, ConfigManager

        with patch.dict(os.environ, {}, clear=True):
            # 只设置 CURSOR_CLOUD_API_KEY
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ["CURSOR_CLOUD_API_KEY"] = "build-cloud-api-key"

            # 重置 ConfigManager 以清除缓存
            ConfigManager.reset_instance()

            config = build_cloud_client_config()

            assert config["api_key"] == "build-cloud-api-key"


# ========== CloudAuthManager 从 config.yaml 读取 api_key 测试 ==========


class TestCloudAuthManagerApiKeyFromConfigYaml:
    """测试 CloudAuthManager 仅设置 cloud_agent.api_key 时可解析到 key

    验证：
    1. 无环境变量时，CloudAuthManager 从 config.yaml 的 cloud_agent.api_key 读取
    2. 优先级正确：显式参数 > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml
    """

    def test_cloud_auth_manager_reads_api_key_from_config_yaml(self, tmp_path):
        """测试 CloudAuthManager 从 config.yaml 的 cloud_agent.api_key 读取 API Key"""
        import yaml
        from pathlib import Path
        from core.config import ConfigManager

        # 创建包含 cloud_agent.api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-cloud-api-key-12345",
                "timeout": 300,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        # 清除环境变量
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            # 重置 ConfigManager 并设置工作目录
            ConfigManager.reset_instance()

            # 使用不设置 api_key 的 CloudAuthConfig
            config = CloudAuthConfig(config_file=str(config_path))
            manager = CloudAuthManager(config=config)

            # 验证能从 config.yaml 读取 api_key
            api_key = manager.get_api_key()

            # 注意：get_api_key 会尝试多个路径，包括 config.yaml
            # 由于我们明确指定了 config_file，应该能读取到
            assert api_key == "yaml-cloud-api-key-12345", (
                f"应从 config.yaml 读取 api_key，实际值: {api_key}"
            )

    def test_cloud_auth_manager_explicit_api_key_priority(self, tmp_path):
        """测试显式传入的 api_key 优先于 config.yaml"""
        import yaml

        # 创建包含 cloud_agent.api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-api-key",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            # 显式设置 api_key
            config = CloudAuthConfig(
                api_key="explicit-api-key",
                config_file=str(config_path),
            )
            manager = CloudAuthManager(config=config)

            api_key = manager.get_api_key()

            # 显式设置的 api_key 应该优先
            assert api_key == "explicit-api-key", (
                f"显式设置的 api_key 应优先，实际值: {api_key}"
            )

    def test_cloud_auth_manager_env_priority_over_config_yaml(self, tmp_path):
        """测试环境变量优先于 config.yaml"""
        import yaml

        # 创建包含 cloud_agent.api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-api-key",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.dict(os.environ, {}, clear=True):
            # 设置环境变量
            os.environ["CURSOR_API_KEY"] = "env-api-key"

            config = CloudAuthConfig(config_file=str(config_path))
            manager = CloudAuthManager(config=config)

            api_key = manager.get_api_key()

            # 环境变量应该优先于 config.yaml
            assert api_key == "env-api-key", (
                f"环境变量应优先于 config.yaml，实际值: {api_key}"
            )

    def test_cloud_auth_manager_no_api_key_returns_none(self, tmp_path):
        """测试无任何 api_key 配置时返回 None"""
        import yaml

        # 创建不包含 api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "timeout": 300,
                # 注意：不设置 api_key
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CURSOR_API_KEY", None)
            os.environ.pop("CURSOR_CLOUD_API_KEY", None)

            config = CloudAuthConfig(config_file=str(config_path))
            manager = CloudAuthManager(config=config)

            api_key = manager.get_api_key()

            # 无任何 api_key 配置时应返回 None
            assert api_key is None, (
                f"无 api_key 配置时应返回 None，实际值: {api_key}"
            )
