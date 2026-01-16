#!/bin/bash
# batch_refactor.sh - 批量重构脚本
# 使用 Cursor Agent 批量重构文件

set -e

PATTERN="${1:-*.py}"
REFACTOR_PROMPT="${2:-添加类型注解}"

echo "🔄 批量重构: $PATTERN"
echo "📝 重构内容: $REFACTOR_PROMPT"

find . -name "$PATTERN" -type f | while read file; do
  echo "处理: $file"
  
  agent -p --force --output-format text \
    "对文件 $file 进行以下重构: $REFACTOR_PROMPT"
  
  if [ $? -eq 0 ]; then
    echo "  ✅ $file 完成"
  else
    echo "  ❌ $file 失败"
  fi
done

echo "🎉 批量重构完成"
