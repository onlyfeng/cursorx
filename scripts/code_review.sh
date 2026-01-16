#!/bin/bash
# code_review.sh - 代码审查脚本
# 使用 Cursor Agent 进行自动化代码审查

set -e

echo "🔍 开始代码审查..."

# 审查最近的更改
agent -p --output-format text \
  "审查最近的代码更改并提供以下反馈：
  - 代码质量和可读性
  - 潜在的 bug 或问题
  - 安全性考虑
  - 最佳实践合规性
  
  注意：不要编写任何代码，只提供审查意见。"

if [ $? -eq 0 ]; then
  echo "✅ 代码审查已完成"
else
  echo "❌ 代码审查失败"
  exit 1
fi
