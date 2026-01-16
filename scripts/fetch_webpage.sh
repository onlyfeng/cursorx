#!/bin/bash
# fetch_webpage.sh - 获取网页内容（无需 GUI）
#
# 用法: ./fetch_webpage.sh <URL> [格式]
# 格式: text (默认), html, links
#
# 示例:
#   ./fetch_webpage.sh https://example.com
#   ./fetch_webpage.sh https://example.com links

URL="${1:-https://example.com}"
FORMAT="${2:-text}"

echo "🌐 获取网页: $URL"
echo "📋 格式: $FORMAT"
echo ""

case "$FORMAT" in
  "text")
    # 使用 lynx 获取纯文本
    lynx -dump -nolist "$URL"
    ;;
  "html")
    # 使用 curl 获取原始 HTML
    curl -s "$URL"
    ;;
  "links")
    # 使用 lynx 提取链接
    lynx -dump -listonly "$URL"
    ;;
  *)
    echo "未知格式: $FORMAT"
    echo "可用格式: text, html, links"
    exit 1
    ;;
esac

echo ""
echo "✅ 完成"
