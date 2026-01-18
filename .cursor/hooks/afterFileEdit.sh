#!/bin/bash
# afterFileEdit hook - 文件编辑后触发
#
# 可用变量:
# - $file_path: 被编辑的文件路径
# - $old_string: 文件修改前的内容（用于 diff）
# - $new_string: 文件修改后的内容
#
# Hooks 特性:
# - 并行执行，合并响应
# - 执行延迟降低 10 倍

# 示例：记录文件变更
if [ -n "$file_path" ]; then
    echo "[Hook] 文件已编辑: $file_path" >> logs/file_changes.log
fi
