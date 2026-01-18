#!/bin/bash
# implement_task.sh - 任务实现脚本
# 使用 Cursor Agent 实现具体任务

set -e

TASK="$1"

if [ -z "$TASK" ]; then
  echo "用法: $0 \"任务描述\""
  exit 1
fi

echo "🔧 开始实现任务: $TASK"

# 使用 --force 允许修改文件
agent -p --force --output-format text \
  "$TASK

  请完成这个任务并确保：
  - 代码可以正常运行
  - 遵循现有代码风格
  - 添加必要的注释"

if [ $? -eq 0 ]; then
  echo "✅ 任务完成"
else
  echo "❌ 任务失败"
  exit 1
fi
