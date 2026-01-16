#!/bin/bash
# analyze-image.sh - 使用无头 CLI 分析图像
#
# 用法: ./analyze_image.sh <图像路径>
# 示例: ./analyze_image.sh ./screenshots/ui-mockup.png

IMAGE_PATH="${1:-./screenshot.png}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "❌ 图像文件不存在: $IMAGE_PATH"
  exit 1
fi

echo "🖼️ 分析图像: $IMAGE_PATH"

agent -p --output-format json \
  "分析此图像并提供详细说明: $IMAGE_PATH" | \
  jq -r '.result // .'

echo ""
echo "✅ 分析完成"
