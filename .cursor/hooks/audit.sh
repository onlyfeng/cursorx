#!/bin/bash
# audit.sh - 通用审计钩子脚本
# 将所有 JSON 输入写入审计日志
# 用法: 被 Cursor hooks 系统调用

# 从 stdin 读取 JSON 输入
json_input=$(cat)

# 创建时间戳
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# 确保日志目录存在
LOG_DIR="${HOME}/.cursor/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/agent-audit.log"

# 提取事件类型（如果有的话）
event_type=$(echo "$json_input" | jq -r '.type // "unknown"' 2>/dev/null)

# 将带时间戳的条目写入审计日志
echo "[$timestamp] [$event_type] $json_input" >> "$LOG_FILE"

# 正常退出，允许操作继续
exit 0
