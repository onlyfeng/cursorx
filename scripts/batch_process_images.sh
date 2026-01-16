#!/bin/bash
# batch_process_images.sh - 批量处理媒体文件
#
# 用法: ./batch_process_images.sh <目录> [扩展名]
# 示例: ./batch_process_images.sh ./images png

DIR="${1:-.}"
EXT="${2:-png}"

echo "🖼️ 批量处理图像: $DIR/*.$EXT"
echo ""

count=0
for image in "$DIR"/*."$EXT"; do
  if [ -f "$image" ]; then
    count=$((count + 1))
    echo "[$count] 正在处理: $image"
    
    # 生成描述文件
    output_file="${image%.$EXT}.description.txt"
    
    agent -p --output-format text \
      "描述图像内容: $image" > "$output_file"
    
    echo "    ✅ 已生成: $output_file"
  fi
done

echo ""
echo "🎉 完成! 共处理 $count 个文件"
