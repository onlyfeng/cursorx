#!/bin/bash
# beforeCommand hook - 命令执行前触发
#
# 可用于：
# - 验证命令安全性
# - 记录命令日志
# - 设置环境变量

# 示例：记录命令
echo "[Hook] 准备执行命令: $*" >> logs/commands.log
