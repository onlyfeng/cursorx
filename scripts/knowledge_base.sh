#!/bin/bash
# knowledge_base.sh - 知识库管理 CLI 包装脚本
#
# 用法:
#   ./knowledge_base.sh add <url>           添加单个 URL
#   ./knowledge_base.sh import <file>       批量导入 URL
#   ./knowledge_base.sh list [-v]           列出所有文档
#   ./knowledge_base.sh search <query>      搜索文档
#   ./knowledge_base.sh remove <doc_id>     删除文档
#   ./knowledge_base.sh refresh [doc_id]    刷新文档
#   ./knowledge_base.sh refresh --all       刷新所有文档
#   ./knowledge_base.sh stats               显示统计信息
#
# 示例:
#   ./knowledge_base.sh add https://example.com
#   ./knowledge_base.sh import urls.txt
#   ./knowledge_base.sh search "Python"
#   ./knowledge_base.sh refresh --all

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 执行 Python CLI
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/knowledge_cli.py" "$@"
