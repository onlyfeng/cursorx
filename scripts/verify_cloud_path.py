#!/usr/bin/env python3
"""云端提交路径验证脚本

验证云端提交的各个组件是否正确连接。

用法:
    python scripts/verify_cloud_path.py
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_check(name: str, passed: bool, detail: str = ""):
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}")
    if detail:
        print(f"      {detail}")


def verify_cloud_utils():
    """验证 core/cloud_utils.py 模块"""
    print_header("1. 验证 Cloud 请求检测 (core/cloud_utils.py)")
    
    try:
        from core.cloud_utils import is_cloud_request, strip_cloud_prefix, CLOUD_PREFIX
        
        # 测试 CLOUD_PREFIX
        print_check("CLOUD_PREFIX 常量", CLOUD_PREFIX == "&", f"值: '{CLOUD_PREFIX}'")
        
        # 测试 is_cloud_request
        test_cases = [
            ("& 任务", True),
            ("&任务", True),
            ("& ", False),  # 只有空白
            ("&", False),   # 只有 &
            ("任务 & 描述", False),  # & 不在开头
            ("", False),
            (None, False),
        ]
        all_passed = True
        for prompt, expected in test_cases:
            result = is_cloud_request(prompt)
            if result != expected:
                all_passed = False
                print_check(f"is_cloud_request('{prompt}')", False, f"期望 {expected}, 实际 {result}")
        print_check("is_cloud_request() 边界测试", all_passed)
        
        # 测试 strip_cloud_prefix
        strip_cases = [
            ("& 任务", "任务"),
            ("&任务", "任务"),
            ("普通任务", "普通任务"),
        ]
        strip_passed = True
        for prompt, expected in strip_cases:
            result = strip_cloud_prefix(prompt)
            if result != expected:
                strip_passed = False
        print_check("strip_cloud_prefix() 测试", strip_passed)
        
        return True
    except Exception as e:
        print_check("导入 core/cloud_utils", False, str(e))
        return False


def verify_cloud_client_factory():
    """验证 CloudClientFactory"""
    print_header("2. 验证 CloudClientFactory (cursor/cloud_client.py)")
    
    try:
        from cursor.cloud_client import (
            CloudClientFactory,
            CloudAuthConfig,
            CloudAuthManager,
            CursorCloudClient,
        )
        
        print_check("导入 CloudClientFactory", True)
        
        # 验证 resolve_api_key 方法
        api_key = CloudClientFactory.resolve_api_key()
        has_key = bool(api_key)
        print_check(
            "resolve_api_key()",
            True,
            f"{'已配置 API Key' if has_key else '未配置 API Key'}"
        )
        
        # 验证 create_auth_config 方法
        auth_config = CloudClientFactory.create_auth_config()
        print_check(
            "create_auth_config()",
            isinstance(auth_config, CloudAuthConfig),
            f"api_base_url: {auth_config.api_base_url}"
        )
        
        # 验证 create 方法
        client, manager = CloudClientFactory.create()
        print_check(
            "create()",
            isinstance(client, CursorCloudClient) and isinstance(manager, CloudAuthManager),
        )
        
        return True
    except Exception as e:
        print_check("CloudClientFactory 验证", False, str(e))
        return False


def verify_cursor_cloud_client():
    """验证 CursorCloudClient"""
    print_header("3. 验证 CursorCloudClient (cursor/cloud/client.py)")
    
    try:
        from cursor.cloud.client import CursorCloudClient, CloudAgentResult
        from cursor.cloud.task import CloudTask, CloudTaskOptions, TaskStatus
        
        print_check("导入 CursorCloudClient", True)
        print_check("导入 CloudAgentResult", True)
        print_check("导入 CloudTask/CloudTaskOptions/TaskStatus", True)
        
        # 验证 is_cloud_request 代理方法
        is_cloud = CursorCloudClient.is_cloud_request("& 任务")
        print_check(
            "CursorCloudClient.is_cloud_request() 代理",
            is_cloud == True,
        )
        
        # 验证 strip_cloud_prefix 代理方法
        stripped = CursorCloudClient.strip_cloud_prefix("& 任务")
        print_check(
            "CursorCloudClient.strip_cloud_prefix() 代理",
            stripped == "任务",
        )
        
        return True
    except Exception as e:
        print_check("CursorCloudClient 验证", False, str(e))
        return False


def verify_executor():
    """验证执行器抽象层"""
    print_header("4. 验证 Executor 抽象层 (cursor/executor.py)")
    
    try:
        from cursor.executor import (
            ExecutionMode,
            AgentExecutorFactory,
            CLIAgentExecutor,
            CloudAgentExecutor,
            AutoAgentExecutor,
            PlanAgentExecutor,
            AskAgentExecutor,
            CloudExecutionPolicy,
        )
        
        print_check("导入 ExecutionMode", True)
        
        # 验证执行模式枚举
        modes = [m.value for m in ExecutionMode]
        print_check(
            "ExecutionMode 枚举",
            set(modes) == {"cli", "cloud", "auto", "plan", "ask"},
            f"模式: {modes}"
        )
        
        # 验证工厂方法
        cli_exec = AgentExecutorFactory.create(mode=ExecutionMode.CLI)
        print_check(
            "AgentExecutorFactory.create(CLI)",
            isinstance(cli_exec, CLIAgentExecutor),
        )
        
        cloud_exec = AgentExecutorFactory.create(mode=ExecutionMode.CLOUD)
        print_check(
            "AgentExecutorFactory.create(CLOUD)",
            isinstance(cloud_exec, CloudAgentExecutor),
        )
        
        auto_exec = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)
        print_check(
            "AgentExecutorFactory.create(AUTO)",
            isinstance(auto_exec, AutoAgentExecutor),
        )
        
        plan_exec = AgentExecutorFactory.create(mode=ExecutionMode.PLAN)
        print_check(
            "AgentExecutorFactory.create(PLAN)",
            isinstance(plan_exec, PlanAgentExecutor),
        )
        
        # 验证 CloudExecutionPolicy
        policy = CloudExecutionPolicy()
        should_try, reason = policy.should_try_cloud(cloud_enabled=True)
        print_check(
            "CloudExecutionPolicy.should_try_cloud()",
            should_try == True,
        )
        
        return True
    except Exception as e:
        print_check("Executor 验证", False, str(e))
        return False


def verify_execution_policy():
    """验证执行策略模块"""
    print_header("5. 验证 ExecutionPolicy (core/execution_policy.py)")
    
    try:
        from core.execution_policy import (
            CloudFailureKind,
            CloudFailureInfo,
            classify_cloud_failure,
            resolve_effective_execution_mode,
            should_route_ampersand_to_cloud,
            sanitize_prompt_for_cli_fallback,
            build_user_facing_fallback_message,
        )
        
        print_check("导入 execution_policy 模块", True)
        
        # 验证 CloudFailureKind 枚举
        kinds = [k.value for k in CloudFailureKind]
        print_check(
            "CloudFailureKind 枚举",
            "auth" in kinds and "rate_limit" in kinds,
            f"类型: {kinds}"
        )
        
        # 验证 should_route_ampersand_to_cloud
        should_route = should_route_ampersand_to_cloud(
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
            has_api_key=True,
        )
        print_check(
            "should_route_ampersand_to_cloud()",
            should_route == True,
        )
        
        # 验证 resolve_effective_execution_mode
        mode, reason = resolve_effective_execution_mode(
            requested_mode="cloud",
            triggered_by_prefix=False,
            cloud_enabled=True,
            has_api_key=True,
        )
        print_check(
            "resolve_effective_execution_mode()",
            mode == "cloud",
            f"模式: {mode}, 原因: {reason}"
        )
        
        # 验证 sanitize_prompt_for_cli_fallback
        sanitized = sanitize_prompt_for_cli_fallback("& 任务")
        print_check(
            "sanitize_prompt_for_cli_fallback()",
            sanitized == "任务",
        )
        
        # 验证 classify_cloud_failure
        info = classify_cloud_failure("认证失败 unauthorized")
        print_check(
            "classify_cloud_failure()",
            info.kind == CloudFailureKind.AUTH,
            f"分类结果: {info.kind.value}"
        )
        
        return True
    except Exception as e:
        print_check("ExecutionPolicy 验证", False, str(e))
        return False


def verify_client_routing():
    """验证 CursorAgentClient 路由逻辑"""
    print_header("6. 验证 CursorAgentClient 路由 (cursor/client.py)")
    
    try:
        from cursor.client import CursorAgentClient, CursorAgentConfig
        
        print_check("导入 CursorAgentClient", True)
        
        # 验证 _is_cloud_request 代理方法
        is_cloud = CursorAgentClient._is_cloud_request("& 任务")
        print_check(
            "CursorAgentClient._is_cloud_request() 代理",
            is_cloud == True,
        )
        
        # 验证 _strip_cloud_prefix 代理方法
        stripped = CursorAgentClient._strip_cloud_prefix("& 任务")
        print_check(
            "CursorAgentClient._strip_cloud_prefix() 代理",
            stripped == "任务",
        )
        
        # 验证 _should_route_to_cloud 逻辑
        # 当 cloud_enabled=False 时，应该返回 False
        config = CursorAgentConfig(cloud_enabled=False)
        client = CursorAgentClient(config)
        should_route = client._should_route_to_cloud("& 任务")
        print_check(
            "_should_route_to_cloud() (cloud_enabled=False)",
            should_route == False,
        )
        
        return True
    except Exception as e:
        print_check("CursorAgentClient 路由验证", False, str(e))
        return False


async def verify_cloud_authentication():
    """验证云端认证"""
    print_header("7. 验证 Cloud 认证状态")
    
    try:
        from cursor.cloud_client import CloudClientFactory
        
        # 检查 API Key
        api_key = CloudClientFactory.resolve_api_key()
        if not api_key:
            print_check("Cloud API Key", False, "未配置 CURSOR_API_KEY 环境变量")
            print("      提示: export CURSOR_API_KEY=your_api_key")
            return False
        
        print_check("Cloud API Key", True, "已配置")
        
        # 尝试认证
        _, auth_manager = CloudClientFactory.create()
        auth_status = await auth_manager.authenticate()
        
        if auth_status.authenticated:
            print_check("Cloud 认证", True, "认证成功")
            return True
        else:
            error_msg = auth_status.error.user_friendly_message if auth_status.error else "未知错误"
            print_check("Cloud 认证", False, error_msg)
            return False
            
    except Exception as e:
        print_check("Cloud 认证验证", False, str(e))
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  云端提交路径验证")
    print("=" * 60)
    
    results = []
    
    # 同步验证
    results.append(("Cloud 请求检测", verify_cloud_utils()))
    results.append(("CloudClientFactory", verify_cloud_client_factory()))
    results.append(("CursorCloudClient", verify_cursor_cloud_client()))
    results.append(("Executor 抽象层", verify_executor()))
    results.append(("ExecutionPolicy", verify_execution_policy()))
    results.append(("Client 路由逻辑", verify_client_routing()))
    
    # 异步验证
    auth_result = asyncio.run(verify_cloud_authentication())
    results.append(("Cloud 认证", auth_result))
    
    # 总结
    print_header("验证总结")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} {name}")
    
    print()
    print(f"  总计: {passed}/{total} 通过")
    print()
    
    if passed == total:
        print("  \033[92m所有验证通过！云端提交路径配置正确。\033[0m")
        return 0
    else:
        print("  \033[93m部分验证失败，请检查上述错误。\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
