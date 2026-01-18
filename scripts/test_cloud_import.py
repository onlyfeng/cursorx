#!/usr/bin/env python3
"""验证 Cloud Agent 导入和配置"""

import sys

sys.path.insert(0, "/mnt/e/QianFeng/ai/cursorx")

try:
    from cursor import (
        CursorAgentConfig,
        CursorCloudClient,
        TaskStatus,
    )

    # 测试 Cloud Agent 配置项
    config = CursorAgentConfig()
    print("Cloud Agent 配置测试:")
    print(f"  cloud_enabled: {config.cloud_enabled}")
    print(f"  cloud_api_base: {config.cloud_api_base}")
    print(f"  cloud_agents_endpoint: {config.cloud_agents_endpoint}")
    print(f"  cloud_timeout: {config.cloud_timeout}")
    print(f"  cloud_poll_interval: {config.cloud_poll_interval}")
    print(f"  auto_detect_cloud_prefix: {config.auto_detect_cloud_prefix}")

    # 测试 Cloud 前缀检测
    print()
    print("Cloud 前缀检测测试:")
    print(f"  is_cloud('& test'): {CursorCloudClient.is_cloud_request('& test')}")
    print(f"  is_cloud('test'): {CursorCloudClient.is_cloud_request('test')}")
    print(f"  strip_prefix('& hello'): '{CursorCloudClient.strip_cloud_prefix('& hello')}'")

    # 测试 TaskStatus
    print()
    print("TaskStatus 枚举测试:")
    for status in TaskStatus:
        print(f"  {status.name}: {status.value}")

    print()
    print("所有导入和配置测试通过!")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
