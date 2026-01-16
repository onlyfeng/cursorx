#!/bin/bash
# analyze_codebase.sh - 代码库分析脚本
# 使用 Cursor Agent 分析代码库结构

set -e

OUTPUT_FORMAT="${1:-json}"

echo "📊 分析代码库结构..."

agent -p --output-format "$OUTPUT_FORMAT" \
  "分析这个代码库并提供：
  - 项目结构概述
  - 主要模块和组件
  - 技术栈和依赖
  - 架构模式
  
  注意：不要编写任何代码，只提供分析。
  
  以 JSON 格式输出：
  {
    \"structure\": \"项目结构描述\",
    \"modules\": [\"模块列表\"],
    \"tech_stack\": [\"技术栈\"],
    \"patterns\": [\"架构模式\"]
  }"

if [ $? -eq 0 ]; then
  echo "✅ 分析完成"
else
  echo "❌ 分析失败"
  exit 1
fi
