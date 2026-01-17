#!/bin/bash
# 代码索引管理脚本
# 
# 提供便捷的命令行接口来管理代码索引
#
# 使用方法:
#   ./scripts/manage_index.sh build [--full]
#   ./scripts/manage_index.sh update
#   ./scripts/manage_index.sh search <query> [--top-k N]
#   ./scripts/manage_index.sh status
#   ./scripts/manage_index.sh clear [--confirm]
#   ./scripts/manage_index.sh info

set -e

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    cat << EOF
代码索引管理工具

用法: $0 <command> [options]

命令:
  build     构建代码索引
  update    增量更新索引
  search    语义搜索代码
  status    显示索引状态
  clear     清除所有索引
  info      显示索引系统信息

选项:
  -h, --help    显示此帮助信息
  -p, --path    指定代码库路径 (默认: 当前目录)
  -c, --config  指定配置文件路径
  -v, --verbose 显示详细日志

示例:
  $0 build                      # 增量构建索引
  $0 build --full               # 全量重建索引
  $0 update                     # 增量更新
  $0 search "用户认证"           # 语义搜索
  $0 search "login" --top-k 5   # 返回前 5 个结果
  $0 status                     # 查看索引状态
  $0 info                       # 显示系统信息
  $0 clear --confirm            # 清除所有索引

更多信息请参考: python -m indexing.cli --help
EOF
}

# 检查 Python 环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到 python3${NC}"
        echo "请安装 Python 3.8+"
        exit 1
    fi
    
    # 检查必要的依赖
    python3 -c "import sentence_transformers" 2>/dev/null || {
        echo -e "${YELLOW}警告: sentence-transformers 未安装${NC}"
        echo "运行: pip install sentence-transformers"
        exit 1
    }
    
    python3 -c "import chromadb" 2>/dev/null || {
        echo -e "${YELLOW}警告: chromadb 未安装${NC}"
        echo "运行: pip install chromadb"
        exit 1
    }
}

# 运行 CLI
run_cli() {
    cd "$PROJECT_ROOT"
    python3 -m indexing.cli "$@"
}

# 主函数
main() {
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 处理帮助选项
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
    esac
    
    # 检查环境
    check_python
    
    # 运行命令
    local command="$1"
    shift
    
    case "$command" in
        build)
            echo -e "${BLUE}构建代码索引...${NC}"
            run_cli build "$@"
            ;;
        update)
            echo -e "${BLUE}增量更新索引...${NC}"
            run_cli update "$@"
            ;;
        search)
            if [ $# -eq 0 ]; then
                echo -e "${RED}错误: 请提供搜索查询${NC}"
                echo "用法: $0 search <query>"
                exit 1
            fi
            run_cli search "$@"
            ;;
        status)
            echo -e "${BLUE}索引状态${NC}"
            run_cli status "$@"
            ;;
        clear)
            echo -e "${YELLOW}清除索引${NC}"
            run_cli clear "$@"
            ;;
        info)
            echo -e "${BLUE}索引系统信息${NC}"
            run_cli info "$@"
            ;;
        *)
            echo -e "${RED}错误: 未知命令 '$command'${NC}"
            echo "运行 '$0 --help' 查看可用命令"
            exit 1
            ;;
    esac
}

# 快捷命令别名
# 这些函数可以直接在 shell 中使用
index_build() {
    "$SCRIPT_DIR/manage_index.sh" build "$@"
}

index_update() {
    "$SCRIPT_DIR/manage_index.sh" update "$@"
}

index_search() {
    "$SCRIPT_DIR/manage_index.sh" search "$@"
}

index_status() {
    "$SCRIPT_DIR/manage_index.sh" status "$@"
}

index_clear() {
    "$SCRIPT_DIR/manage_index.sh" clear "$@"
}

index_info() {
    "$SCRIPT_DIR/manage_index.sh" info "$@"
}

# 运行主函数
main "$@"
