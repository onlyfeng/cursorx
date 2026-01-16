#!/bin/bash
# browse_web.sh - 使用 Agent 浏览网页
#
# 前提: 需要安装 MCP Playwright 服务器
# npm install -g @anthropic/mcp-server-playwright
#
# 用法: ./browse_web.sh <URL> [操作]
# 示例: ./browse_web.sh "https://example.com" "截取屏幕截图"

URL="${1:-https://example.com}"
ACTION="${2:-截取屏幕截图并描述页面内容}"

echo "🌐 浏览网页: $URL"
echo "📋 操作: $ACTION"
echo ""

# 检查 MCP 服务器
if ! agent mcp list 2>/dev/null | grep -q "playwright"; then
  echo "⚠️ 未找到 playwright MCP 服务器"
  echo "请安装: npm install -g @anthropic/mcp-server-playwright"
  exit 1
fi

agent -p --force --output-format text \
  "导航到 $URL 并执行以下操作: $ACTION"

echo ""
echo "✅ 完成"
