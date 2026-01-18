#!/bin/bash
# stream-progress.sh - 实时跟踪进度
# 使用 stream-json 格式实时显示 Agent 执行进度

echo "🚀 开始流式处理..."

# 实时跟踪进度
accumulated_text=""
tool_count=0
start_time=$(date +%s)

PROMPT="${1:-分析项目结构并在 analysis.txt 中生成摘要报告}"

agent -p --force --output-format stream-json --stream-partial-output \
  "$PROMPT" | \
  while IFS= read -r line; do

    type=$(echo "$line" | jq -r '.type // empty')
    subtype=$(echo "$line" | jq -r '.subtype // empty')

    case "$type" in
      "system")
        if [ "$subtype" = "init" ]; then
          model=$(echo "$line" | jq -r '.model // "unknown"')
          echo "🤖 使用模型: $model"
        fi
        ;;

      "assistant")
        # 累积增量文本以实现流畅的进度显示
        content=$(echo "$line" | jq -r '.message.content[0].text // empty')
        accumulated_text="$accumulated_text$content"

        # 显示实时进度(每次字符增量时更新)
        printf "\r📝 生成中: %d 字符" ${#accumulated_text}
        ;;

      "tool_call")
        if [ "$subtype" = "started" ]; then
          tool_count=$((tool_count + 1))
          # 提取工具信息
          if echo "$line" | jq -e '.tool_call.writeToolCall' > /dev/null 2>&1; then
            path=$(echo "$line" | jq -r '.tool_call.writeToolCall.args.path // "unknown"')
            echo -e "\n🔧 工具 #$tool_count: 创建 $path"
          elif echo "$line" | jq -e '.tool_call.readToolCall' > /dev/null 2>&1; then
            path=$(echo "$line" | jq -r '.tool_call.readToolCall.args.path // "unknown"')
            echo -e "\n📖 工具 #$tool_count: 读取 $path"
          elif echo "$line" | jq -e '.tool_call.shellToolCall' > /dev/null 2>&1; then
            cmd=$(echo "$line" | jq -r '.tool_call.shellToolCall.args.command // "unknown"')
            echo -e "\n💻 工具 #$tool_count: 执行 $cmd"
          fi
        elif [ "$subtype" = "completed" ]; then
          # 提取并显示工具结果
          if echo "$line" | jq -e '.tool_call.writeToolCall.result.success' > /dev/null 2>&1; then
            lines=$(echo "$line" | jq -r '.tool_call.writeToolCall.result.success.linesCreated // 0')
            size=$(echo "$line" | jq -r '.tool_call.writeToolCall.result.success.fileSize // 0')
            echo "   ✅ 已创建 $lines 行 ($size 字节)"
          elif echo "$line" | jq -e '.tool_call.readToolCall.result.success' > /dev/null 2>&1; then
            lines=$(echo "$line" | jq -r '.tool_call.readToolCall.result.success.totalLines // 0')
            echo "   ✅ 已读取 $lines 行"
          elif echo "$line" | jq -e '.tool_call.shellToolCall.result.success' > /dev/null 2>&1; then
            echo "   ✅ 命令执行成功"
          fi
        fi
        ;;

      "result")
        duration=$(echo "$line" | jq -r '.duration_ms // 0')
        end_time=$(date +%s)
        total_time=$((end_time - start_time))
        echo -e "\n\n🎯 完成, 耗时 ${duration}ms (总计 ${total_time}s)"
        echo "📊 最终统计: $tool_count 个工具, 生成 ${#accumulated_text} 字符"
        ;;

      "error")
        error_msg=$(echo "$line" | jq -r '.error // "未知错误"')
        echo -e "\n❌ 错误: $error_msg"
        ;;
    esac
  done

echo ""
echo "✨ 流式处理完成"
