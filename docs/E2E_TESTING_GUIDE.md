# E2E 测试指南

本文档描述了 CursorX 多 Agent 系统的端到端 (E2E) 测试框架、测试编写方法和最佳实践。

## 1. E2E 测试概述

### 1.1 测试目标和范围

E2E 测试验证多 Agent 系统的完整工作流程，确保各组件协同工作正常。主要测试目标包括：

- **工作流验证**：验证 Planner → Worker → Reviewer 的完整迭代循环
- **协作机制**：验证多 Agent 之间的消息传递和状态同步
- **错误恢复**：验证系统在各种错误场景下的容错能力
- **状态管理**：验证迭代状态、任务状态的正确转换

### 1.2 测试分类

E2E 测试按功能领域分为以下类别：

| 测试文件 | 类别 | 描述 |
|---------|------|------|
| `test_e2e_workflow.py` | workflow | 单/多迭代工作流、状态转换 |
| `test_e2e_agent_collaboration.py` | collaboration | Planner-Worker-Reviewer 协作 |
| `test_e2e_error_handling.py` | error | 规划/执行/评审阶段的错误处理 |
| `test_e2e_execution_modes.py` | execution | 不同执行模式验证 |
| `test_e2e_streaming.py` | streaming | 流式输出测试 |
| `test_e2e_knowledge_integration.py` | integration | 知识库集成测试 |
| `test_e2e_run_modes.py` | modes | 运行模式 (plan/ask/agent) |
| `test_e2e_basic.py` | basic | 基础功能验证 |
| `test_e2e_imports.py` | imports | 模块导入测试 |

## 2. 运行测试

### 2.1 本地运行命令

```bash
# 运行所有 E2E 测试
pytest tests/test_e2e_*.py -v

# 运行特定类别的测试
pytest tests/test_e2e_workflow.py -v
pytest tests/test_e2e_error_handling.py -v
pytest tests/test_e2e_agent_collaboration.py -v

# 按标记运行测试
pytest -m e2e -v                    # 所有 E2E 测试
pytest -m "e2e and not slow" -v     # 排除慢速测试
pytest -m integration -v            # 仅集成测试

# 运行单个测试类或方法
pytest tests/test_e2e_workflow.py::TestSingleIterationWorkflow -v
pytest tests/test_e2e_workflow.py::TestSingleIterationWorkflow::test_simple_task_completion -v

# 并行运行（需安装 pytest-xdist）
pytest tests/test_e2e_*.py -n auto -v

# 生成覆盖率报告
pytest tests/test_e2e_*.py --cov=. --cov-report=html
```

### 2.2 CI 集成说明

E2E 测试在 CI 流水线中自动运行。配置位于 `.github/workflows/ci.yml`：

```yaml
# CI 中的 E2E 测试配置示例
- name: Run E2E Tests
  run: |
    pytest tests/test_e2e_*.py -v --tb=short
  env:
    CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
```

**CI 注意事项**：
- E2E 测试使用 Mock 替代真实 CLI 调用，无需实际 API Key
- 慢速测试可通过 `-m "not slow"` 在快速检查中跳过
- 完整测试在 PR 合并前运行

### 2.3 环境变量配置

| 环境变量 | 描述 | 优先级 | 默认值 |
|---------|------|--------|--------|
| `CURSOR_API_KEY` | 主要 API 密钥（测试中通常使用 mock） | 高 | - |
| `CURSOR_CLOUD_API_KEY` | 备选 API 密钥（仅当 `CURSOR_API_KEY` 未设置时使用） | 低 | - |
| `E2E_TEST_TIMEOUT` | 测试超时时间（秒） | - | 300 |
| `E2E_VERBOSE` | 详细日志输出 | - | false |

**API Key 优先级**: `CURSOR_API_KEY` 优先于 `CURSOR_CLOUD_API_KEY`。后者仅作为备选，当前者未设置时才会使用。

测试中使用 `env_with_api_key` fixture 设置测试用 API Key：

```python
def test_with_api_key(env_with_api_key):
    # 测试环境已设置 CURSOR_API_KEY="test-api-key-12345"
    ...
```

## 3. 编写新测试

### 3.1 使用 Fixtures 的示例

#### 基础 Fixtures

```python
import pytest
from tests.conftest import (
    mock_executor,
    mock_task_queue,
    mock_knowledge_manager,
    temp_workspace,
    sample_tasks,
    create_test_task,
)

class TestMyFeature:
    """功能测试类"""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """创建测试用 Orchestrator"""
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=3,
            worker_pool_size=2,
            enable_auto_commit=False,
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_feature(
        self,
        orchestrator: Orchestrator,
        temp_workspace: Path,
    ) -> None:
        """测试功能"""
        # 使用临时工作空间
        assert temp_workspace.exists()
        ...
```

#### 使用 MockAgentExecutor

```python
from tests.conftest import MockAgentExecutor, MockAgentResult

@pytest.fixture
def configured_executor() -> MockAgentExecutor:
    """配置好响应的执行器"""
    executor = MockAgentExecutor()

    # 配置单个响应
    executor.configure_response(
        success=True,
        output="任务完成",
        files_modified=["main.py"],
    )

    # 配置多个响应（按顺序返回）
    executor.configure_responses([
        {"success": True, "output": "第一次执行"},
        {"success": True, "output": "第二次执行"},
        {"success": False, "error": "第三次失败"},
    ])

    # 配置特定 prompt 的响应
    executor.configure_response_for_prompt(
        prompt_contains="分析代码",
        response={"success": True, "output": "代码分析完成"},
    )

    return executor
```

#### 使用 orchestrator_factory

```python
from tests.conftest import orchestrator_factory

@pytest.mark.asyncio
async def test_with_factory(orchestrator_factory):
    """使用工厂创建自定义配置的 Orchestrator"""
    orchestrator = orchestrator_factory(
        max_iterations=5,
        worker_pool_size=3,
        strict_review=True,
        enable_auto_commit=True,
    )

    result = await orchestrator.run("测试目标")
    assert result["success"] is True
```

### 3.2 Mock 策略

E2E 测试使用 Mock 替代真实 Cursor CLI 调用，遵循以下策略：

#### Mock Planner 执行

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_planner_mock(self, orchestrator: Orchestrator):
    mock_plan_result = {
        "success": True,
        "analysis": "分析结果",
        "tasks": [
            {
                "type": "implement",
                "title": "实现功能",
                "description": "功能描述",
                "instruction": "执行指令",
                "target_files": ["main.py"],
                "priority": "high",
            },
        ],
        "sub_planners_needed": [],
    }

    with patch.object(
        orchestrator.planner, "execute", new_callable=AsyncMock
    ) as mock_planner:
        mock_planner.return_value = mock_plan_result
        result = await orchestrator.run("目标")
        mock_planner.assert_called_once()
```

#### Mock Worker 执行

```python
@pytest.mark.asyncio
async def test_worker_mock(self, orchestrator: Orchestrator):
    async def simulate_task_completion(queue: Any, iteration_id: int) -> None:
        tasks = queue.get_tasks_by_iteration(iteration_id)
        for task in tasks:
            # 模拟任务执行
            task.start()
            await asyncio.sleep(0.01)  # 模拟执行时间
            task.complete({
                "output": f"{task.title} 完成",
                "files_modified": task.target_files,
            })

    with patch.object(
        orchestrator.worker_pool, "start", new_callable=AsyncMock
    ) as mock_workers:
        mock_workers.side_effect = simulate_task_completion
        ...
```

#### Mock Reviewer 评审

```python
from agents.reviewer import ReviewDecision

@pytest.mark.asyncio
async def test_reviewer_mock(self, orchestrator: Orchestrator):
    review_count = 0

    def get_review_result(*args, **kwargs):
        nonlocal review_count
        review_count += 1

        if review_count < 2:
            return {
                "success": True,
                "decision": ReviewDecision.CONTINUE,
                "score": 60,
                "summary": "需要继续迭代",
                "suggestions": ["改进建议1", "改进建议2"],
                "next_iteration_focus": "重点关注",
            }
        else:
            return {
                "success": True,
                "decision": ReviewDecision.COMPLETE,
                "score": 95,
                "summary": "目标完成",
            }

    with patch.object(
        orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
    ) as mock_reviewer:
        mock_reviewer.side_effect = get_review_result
        ...
```

#### 完整的 Mock 组合

```python
@pytest.mark.asyncio
async def test_full_workflow(self, orchestrator: Orchestrator):
    """完整工作流测试模板"""
    mock_plan_result = {...}
    mock_review_result = {...}

    with patch.object(
        orchestrator.planner, "execute", new_callable=AsyncMock
    ) as mock_planner:
        with patch.object(
            orchestrator.worker_pool, "start", new_callable=AsyncMock
        ) as mock_workers:
            with patch.object(
                orchestrator.reviewer, "review_iteration", new_callable=AsyncMock
            ) as mock_reviewer:
                mock_planner.return_value = mock_plan_result
                mock_reviewer.return_value = mock_review_result

                async def simulate_complete(queue, iteration_id):
                    tasks = queue.get_tasks_by_iteration(iteration_id)
                    for task in tasks:
                        task.complete({"output": "完成"})

                mock_workers.side_effect = simulate_complete

                result = await orchestrator.run("测试目标")

                # 验证结果
                assert result["success"] is True
                mock_planner.assert_called_once()
                mock_workers.assert_called_once()
                mock_reviewer.assert_called_once()
```

### 3.3 断言最佳实践

使用 `conftest_e2e.py` 提供的断言助手：

```python
from tests.conftest import (
    assert_iteration_success,
    assert_iteration_failed,
    assert_task_completed,
    assert_task_failed,
    assert_tasks_in_order,
    assert_executor_called_with,
)

@pytest.mark.asyncio
async def test_with_assertions(self):
    result = await orchestrator.run("目标")

    # 验证迭代成功
    assert_iteration_success(
        result,
        expected_iterations=2,
        min_tasks_completed=3,
    )

    # 验证任务完成
    task = queue.get_task("task-id")
    assert_task_completed(task, expected_result_contains="成功")

    # 验证任务失败
    assert_task_failed(task, expected_error_contains="超时")

    # 验证执行器调用
    assert_executor_called_with(
        executor,
        prompt_contains="分析代码",
        times=1,
    )

    # 验证任务执行顺序
    assert_tasks_in_order(tasks, ["任务1", "任务2", "任务3"])
```

#### 自定义断言示例

```python
def assert_workflow_stats(result: dict, expected: dict) -> None:
    """验证工作流统计信息"""
    assert result["total_tasks_created"] == expected.get("created", 0)
    assert result["total_tasks_completed"] == expected.get("completed", 0)
    assert result["total_tasks_failed"] == expected.get("failed", 0)
    assert result["iterations_completed"] == expected.get("iterations", 1)

# 使用
assert_workflow_stats(result, {
    "created": 5,
    "completed": 4,
    "failed": 1,
    "iterations": 2,
})
```

## 4. 测试标记

### 4.1 @pytest.mark.e2e

标记端到端测试，可能需要完整环境：

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_workflow():
    """完整工作流测试"""
    ...
```

### 4.2 @pytest.mark.slow

标记执行时间较长的测试：

```python
@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_many_iterations():
    """多迭代测试（耗时）"""
    config = OrchestratorConfig(max_iterations=10, ...)
    ...
```

运行时排除慢速测试：
```bash
pytest -m "e2e and not slow" -v
```

### 4.3 @pytest.mark.integration

标记需要外部依赖的集成测试：

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_knowledge_base_integration():
    """知识库集成测试"""
    ...
```

### 4.4 组合标记

```python
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_system_integration():
    """完整系统集成测试（慢速）"""
    ...
```

## 5. 故障排查

### 5.1 常见问题和解决方案

#### 问题：测试超时

**症状**：测试长时间无响应或超时失败

**解决方案**：
```python
# 1. 设置合理的超时时间
@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 60 秒超时
async def test_with_timeout():
    ...

# 2. 检查 Mock 是否正确配置
async def simulate_complete(queue, iteration_id):
    tasks = queue.get_tasks_by_iteration(iteration_id)
    for task in tasks:
        task.complete({"output": "完成"})  # 确保任务被标记为完成

# 3. 使用 wait_for_condition 替代固定等待
from tests.conftest import wait_for_condition

await wait_for_condition(
    lambda: queue.is_iteration_complete(1),
    timeout=5.0,
    interval=0.1,
)
```

#### 问题：Mock 未生效

**症状**：实际方法被调用而非 Mock

**解决方案**：
```python
# 确保 patch 目标正确
# 错误：patch 导入位置
with patch("agents.planner.PlannerAgent.execute"):  # 可能不生效
    ...

# 正确：patch 实例方法
with patch.object(orchestrator.planner, "execute", new_callable=AsyncMock):
    ...
```

#### 问题：异步测试失败

**症状**：`RuntimeError: no running event loop`

**解决方案**：
```python
# 确保使用 @pytest.mark.asyncio 装饰器
@pytest.mark.asyncio
async def test_async_feature():
    ...

# 或在 pytest.ini 中配置
# [pytest]
# asyncio_mode = auto
```

#### 问题：状态污染

**症状**：测试之间相互影响

**解决方案**：
```python
# 使用 fixture 隔离状态
@pytest.fixture
def fresh_orchestrator():
    """每个测试获得全新的 Orchestrator"""
    config = OrchestratorConfig(...)
    return Orchestrator(config)

# 使用 MockAgentExecutor.reset()
def test_multiple_executions(mock_executor):
    mock_executor.configure_response(success=True, output="第一次")
    # ... 第一次测试 ...

    mock_executor.reset()  # 重置状态

    mock_executor.configure_response(success=False, error="第二次")
    # ... 第二次测试 ...
```

### 5.2 日志分析

#### 启用详细日志

```python
import logging

# 在测试中启用日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cursorx")
logger.setLevel(logging.DEBUG)

# 或使用 pytest 的 caplog fixture
def test_with_logging(caplog):
    with caplog.at_level(logging.DEBUG):
        result = await orchestrator.run("目标")

    # 检查日志内容
    assert "规划阶段完成" in caplog.text
    assert "ERROR" not in caplog.text
```

#### 检查执行历史

```python
def test_execution_history(mock_executor):
    # ... 执行测试 ...

    # 分析执行历史
    history = mock_executor.execution_history
    for i, call in enumerate(history):
        print(f"调用 {i + 1}:")
        print(f"  Prompt: {call['prompt'][:100]}...")
        print(f"  Context: {call.get('context', {})}")
        print(f"  Timestamp: {call['timestamp']}")

    # 验证调用次数
    assert mock_executor.execution_count == 3
```

#### 调试技巧

```python
# 1. 打印任务状态
def debug_tasks(queue, iteration_id):
    tasks = queue.get_tasks_by_iteration(iteration_id)
    for task in tasks:
        print(f"Task: {task.title}")
        print(f"  Status: {task.status}")
        print(f"  Error: {task.error}")
        print(f"  Result: {task.result}")

# 2. 使用 pytest --pdb 进入调试器
# pytest tests/test_e2e_workflow.py -v --pdb

# 3. 使用 breakpoint() 设置断点
async def test_debug():
    result = await orchestrator.run("目标")
    breakpoint()  # 在此暂停
    assert result["success"]
```

### 5.3 测试隔离

确保测试独立运行，不依赖执行顺序：

```python
# 使用 tmp_path 进行文件隔离
def test_file_operations(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("# test")
    ...

# 使用 monkeypatch 进行环境隔离
def test_env_isolation(monkeypatch):
    monkeypatch.setenv("CUSTOM_VAR", "test_value")
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    ...

# 使用 temp_workspace fixture
def test_workspace_isolation(temp_workspace):
    # temp_workspace 是临时目录，测试后自动清理
    config_file = temp_workspace / "config.yaml"
    config_file.write_text("key: value")
    ...
```

## 附录

### A. 测试文件结构

```
tests/
├── conftest.py           # 主配置，导入 E2E fixtures
├── conftest_e2e.py       # E2E 测试专用配置
├── test_e2e_basic.py
├── test_e2e_workflow.py
├── test_e2e_error_handling.py
├── test_e2e_agent_collaboration.py
├── test_e2e_execution_modes.py
├── test_e2e_streaming.py
├── test_e2e_knowledge_integration.py
├── test_e2e_run_modes.py
└── test_e2e_imports.py
```

### B. 常用 Fixtures 速查

| Fixture | 描述 | Scope |
|---------|------|-------|
| `temp_workspace` | 临时工作目录（含 git 初始化） | function |
| `mock_executor` | MockAgentExecutor 实例 | function |
| `mock_task_queue` | MockTaskQueue 实例 | function |
| `mock_knowledge_manager` | MockKnowledgeManager 实例 | function |
| `sample_tasks` | 示例任务列表 | function |
| `orchestrator_factory` | Orchestrator 工厂函数 | function |
| `project_root` | 项目根目录路径 | session |
| `env_with_api_key` | 设置测试 API Key | function |
| `clean_env` | 清理环境变量 | function |

### C. ReviewDecision 值

| 值 | 描述 | 后续行为 |
|----|------|---------|
| `COMPLETE` | 目标完成 | 终止迭代 |
| `CONTINUE` | 继续迭代 | 开始下一轮 |
| `ADJUST` | 调整方向 | 重新规划 |
| `ABORT` | 终止任务 | 终止并标记失败 |
