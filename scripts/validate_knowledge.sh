#!/bin/bash
# validate_knowledge.sh - 知识库验证脚本
#
# 用法:
#   ./validate_knowledge.sh              # 快速验证
#   ./validate_knowledge.sh --full       # 完整测试 (pytest)
#   ./validate_knowledge.sh --unit       # 单元测试
#   ./validate_knowledge.sh --verbose    # 详细输出
#   ./validate_knowledge.sh --help       # 帮助信息
#
# 返回码:
#   0 - 所有验证通过
#   1 - 存在验证失败
#
# 示例:
#   ./scripts/validate_knowledge.sh           # 快速验证知识库
#   ./scripts/validate_knowledge.sh --full    # 运行所有知识库测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo "知识库验证脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  (无参数)       快速验证 - 运行基本功能检查"
    echo "  --full         完整测试 - 使用 pytest 运行所有知识库测试"
    echo "  --unit         单元测试 - 仅运行 test_knowledge.py"
    echo "  --vector       向量测试 - 运行向量搜索相关测试"
    echo "  --validation   验证测试 - 运行验证测试模块"
    echo "  --all          全部测试 - 运行所有测试模块"
    echo "  --verbose, -v  详细输出"
    echo "  --help, -h     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 快速验证"
    echo "  $0 --full             # 完整测试"
    echo "  $0 --unit -v          # 详细单元测试"
}

# 检查 Python 环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到 python3${NC}"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}检查依赖...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # 检查核心依赖
    python3 -c "import knowledge" 2>/dev/null || {
        echo -e "${YELLOW}警告: knowledge 模块导入失败，尝试安装依赖...${NC}"
        pip install -r requirements.txt -q
    }
}

# 快速验证
quick_validation() {
    echo -e "${BLUE}运行快速验证...${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    PYTHONPATH=. python3 tests/test_knowledge_validation.py
    
    return $?
}

# 运行 pytest 测试
run_pytest() {
    local test_file="$1"
    local verbose="$2"
    
    cd "$PROJECT_ROOT"
    export PYTHONPATH=.
    
    if [ "$verbose" = "true" ]; then
        python3 -m pytest $test_file -v --tb=short
    else
        python3 -m pytest $test_file --tb=short
    fi
    
    return $?
}

# 主函数
main() {
    local mode="quick"
    local verbose="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                mode="full"
                shift
                ;;
            --unit)
                mode="unit"
                shift
                ;;
            --vector)
                mode="vector"
                shift
                ;;
            --validation)
                mode="validation"
                shift
                ;;
            --all)
                mode="all"
                shift
                ;;
            -v|--verbose)
                verbose="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    echo "=============================================="
    echo -e "${BLUE}知识库验证${NC}"
    echo "=============================================="
    echo ""
    
    check_python
    check_dependencies
    
    local exit_code=0
    
    case $mode in
        quick)
            quick_validation || exit_code=$?
            ;;
        full)
            echo -e "${BLUE}运行完整测试套件...${NC}"
            echo ""
            run_pytest "tests/test_knowledge.py tests/test_knowledge_vector.py tests/test_knowledge_validation.py" "$verbose" || exit_code=$?
            ;;
        unit)
            echo -e "${BLUE}运行单元测试...${NC}"
            echo ""
            run_pytest "tests/test_knowledge.py" "$verbose" || exit_code=$?
            ;;
        vector)
            echo -e "${BLUE}运行向量搜索测试...${NC}"
            echo ""
            run_pytest "tests/test_knowledge_vector.py" "$verbose" || exit_code=$?
            ;;
        validation)
            echo -e "${BLUE}运行验证测试...${NC}"
            echo ""
            run_pytest "tests/test_knowledge_validation.py" "$verbose" || exit_code=$?
            ;;
        all)
            echo -e "${BLUE}运行所有测试...${NC}"
            echo ""
            run_pytest "tests/" "$verbose" || exit_code=$?
            ;;
    esac
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}=============================================="
        echo -e "验证完成 - 所有测试通过"
        echo -e "==============================================${NC}"
    else
        echo -e "${RED}=============================================="
        echo -e "验证完成 - 存在失败的测试"
        echo -e "==============================================${NC}"
    fi
    
    return $exit_code
}

main "$@"
