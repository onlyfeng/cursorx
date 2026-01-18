#!/bin/bash
# ============================================================
# 依赖同步与检测脚本
# 功能:
#   1. 编译 .in 文件生成锁定的 .txt 文件
#   2. 检测依赖不一致问题
#   3. 验证 pyproject.toml 与 requirements.in 同步
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 帮助信息
show_help() {
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  compile     编译 .in 文件生成锁定的 .txt 文件"
    echo "  check       检查依赖一致性"
    echo "  upgrade     升级所有依赖到最新版本"
    echo "  sync        同步安装依赖到当前环境"
    echo "  audit       运行安全审计"
    echo "  help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 compile   # 编译锁定文件"
    echo "  $0 check     # 检查依赖一致性"
    echo "  $0 upgrade   # 升级依赖"
}

# 检查 pip-tools 是否安装
check_pip_tools() {
    if ! command -v pip-compile &> /dev/null; then
        echo -e "${YELLOW}[警告] pip-tools 未安装，正在安装...${NC}"
        pip install pip-tools>=7.3.0
    fi
}

# 编译锁定文件
compile_requirements() {
    echo -e "${BLUE}[INFO] 编译依赖锁定文件...${NC}"
    
    check_pip_tools
    
    # 编译核心依赖
    echo -e "${BLUE}[INFO] 编译 requirements.txt...${NC}"
    pip-compile requirements.in -o requirements.txt --quiet --strip-extras
    
    # 编译开发依赖
    echo -e "${BLUE}[INFO] 编译 requirements-dev.txt...${NC}"
    pip-compile requirements-dev.in -o requirements-dev.txt --quiet --strip-extras
    
    # 编译测试依赖
    echo -e "${BLUE}[INFO] 编译 requirements-test.txt...${NC}"
    pip-compile requirements-test.in -o requirements-test.txt --quiet --strip-extras
    
    echo -e "${GREEN}[成功] 所有锁定文件已生成${NC}"
}

# 升级依赖
upgrade_requirements() {
    echo -e "${BLUE}[INFO] 升级依赖到最新版本...${NC}"
    
    check_pip_tools
    
    # 升级核心依赖
    echo -e "${BLUE}[INFO] 升级 requirements.txt...${NC}"
    pip-compile requirements.in -o requirements.txt --upgrade --quiet --strip-extras
    
    # 升级开发依赖
    echo -e "${BLUE}[INFO] 升级 requirements-dev.txt...${NC}"
    pip-compile requirements-dev.in -o requirements-dev.txt --upgrade --quiet --strip-extras
    
    # 升级测试依赖
    echo -e "${BLUE}[INFO] 升级 requirements-test.txt...${NC}"
    pip-compile requirements-test.in -o requirements-test.txt --upgrade --quiet --strip-extras
    
    echo -e "${GREEN}[成功] 所有依赖已升级${NC}"
}

# 同步安装
sync_requirements() {
    echo -e "${BLUE}[INFO] 同步安装依赖...${NC}"
    
    if ! command -v pip-sync &> /dev/null; then
        echo -e "${YELLOW}[警告] pip-tools 未安装，正在安装...${NC}"
        pip install pip-tools>=7.3.0
    fi
    
    # 安装所有依赖
    pip-sync requirements.txt requirements-dev.txt requirements-test.txt
    
    echo -e "${GREEN}[成功] 依赖同步完成${NC}"
}

# 提取 pyproject.toml 中的依赖（简化版本，去除版本约束）
extract_pyproject_deps() {
    local section=$1
    python3 -c "
import re
import sys

with open('pyproject.toml', 'r') as f:
    content = f.read()

# 根据 section 提取不同部分
if '$section' == 'core':
    # 提取 [project] dependencies
    match = re.search(r'\[project\]\s*.*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
elif '$section' == 'dev':
    # 提取 [project.optional-dependencies] dev
    match = re.search(r'\[project\.optional-dependencies\].*?dev\s*=\s*\[(.*?)\]', content, re.DOTALL)
elif '$section' == 'test':
    # 提取 [project.optional-dependencies] test
    match = re.search(r'\[project\.optional-dependencies\].*?test\s*=\s*\[(.*?)\]', content, re.DOTALL)
else:
    sys.exit(0)

if match:
    deps_str = match.group(1)
    # 提取包名（去除版本约束）
    deps = re.findall(r'\"([a-zA-Z0-9_-]+)', deps_str)
    for dep in sorted(set(deps)):
        print(dep.lower())
"
}

# 提取 .in 文件中的依赖
extract_in_deps() {
    local file=$1
    grep -E '^[a-zA-Z0-9_-]+' "$file" 2>/dev/null | \
        grep -v '^#' | \
        grep -v '^-r' | \
        sed 's/[>=<].*//; s/\[.*\]//' | \
        tr '[:upper:]' '[:lower:]' | \
        sort -u
}

# 检查依赖一致性
check_consistency() {
    echo -e "${BLUE}[INFO] 检查依赖一致性...${NC}"
    
    local has_error=0
    
    # 检查文件是否存在
    for file in requirements.in requirements-dev.in requirements-test.in; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}[错误] 文件不存在: $file${NC}"
            has_error=1
        fi
    done
    
    if [[ $has_error -eq 1 ]]; then
        return 1
    fi
    
    # 检查锁定文件是否存在
    echo -e "${BLUE}[INFO] 检查锁定文件...${NC}"
    for file in requirements.txt requirements-dev.txt requirements-test.txt; do
        if [[ ! -f "$file" ]]; then
            echo -e "${YELLOW}[警告] 锁定文件不存在: $file (运行 '$0 compile' 生成)${NC}"
        fi
    done
    
    # 检查 pyproject.toml 与 requirements.in 的核心依赖
    echo -e "${BLUE}[INFO] 检查 pyproject.toml 与 requirements.in 同步...${NC}"
    
    local pyproject_deps=$(extract_pyproject_deps "core")
    local in_deps=$(extract_in_deps "requirements.in")
    
    # 找出 pyproject.toml 中有但 requirements.in 中没有的
    local missing_in_req=""
    for dep in $pyproject_deps; do
        if ! echo "$in_deps" | grep -q "^${dep}$"; then
            missing_in_req="$missing_in_req $dep"
        fi
    done
    
    # 找出 requirements.in 中有但 pyproject.toml 中没有的
    local extra_in_req=""
    for dep in $in_deps; do
        if ! echo "$pyproject_deps" | grep -q "^${dep}$"; then
            extra_in_req="$extra_in_req $dep"
        fi
    done
    
    if [[ -n "$missing_in_req" ]]; then
        echo -e "${YELLOW}[警告] pyproject.toml 中有，但 requirements.in 中缺少:${NC}"
        for dep in $missing_in_req; do
            echo -e "  - $dep"
        done
        has_error=1
    fi
    
    if [[ -n "$extra_in_req" ]]; then
        echo -e "${YELLOW}[信息] requirements.in 中有，但 pyproject.toml 中没有:${NC}"
        for dep in $extra_in_req; do
            echo -e "  - $dep"
        done
        # 这不一定是错误，可能是额外添加的依赖
    fi
    
    # 检查开发依赖
    echo -e "${BLUE}[INFO] 检查开发依赖...${NC}"
    local pyproject_dev=$(extract_pyproject_deps "dev")
    local dev_deps=$(extract_in_deps "requirements-dev.in")
    
    local missing_dev=""
    for dep in $pyproject_dev; do
        if ! echo "$dev_deps" | grep -q "^${dep}$"; then
            missing_dev="$missing_dev $dep"
        fi
    done
    
    if [[ -n "$missing_dev" ]]; then
        echo -e "${YELLOW}[警告] pyproject.toml [dev] 中有，但 requirements-dev.in 中缺少:${NC}"
        for dep in $missing_dev; do
            echo -e "  - $dep"
        done
        has_error=1
    fi
    
    # 检查测试依赖
    echo -e "${BLUE}[INFO] 检查测试依赖...${NC}"
    local pyproject_test=$(extract_pyproject_deps "test")
    local test_deps=$(extract_in_deps "requirements-test.in")
    
    local missing_test=""
    for dep in $pyproject_test; do
        if ! echo "$test_deps" | grep -q "^${dep}$"; then
            missing_test="$missing_test $dep"
        fi
    done
    
    if [[ -n "$missing_test" ]]; then
        echo -e "${YELLOW}[警告] pyproject.toml [test] 中有，但 requirements-test.in 中缺少:${NC}"
        for dep in $missing_test; do
            echo -e "  - $dep"
        done
        has_error=1
    fi
    
    # 检查锁定文件是否过期
    echo -e "${BLUE}[INFO] 检查锁定文件是否过期...${NC}"
    for in_file in requirements.in requirements-dev.in requirements-test.in; do
        local txt_file="${in_file%.in}.txt"
        if [[ -f "$txt_file" ]]; then
            if [[ "$in_file" -nt "$txt_file" ]]; then
                echo -e "${YELLOW}[警告] $txt_file 可能已过期 (源文件 $in_file 更新)${NC}"
                has_error=1
            fi
        fi
    done
    
    if [[ $has_error -eq 0 ]]; then
        echo -e "${GREEN}[成功] 依赖一致性检查通过${NC}"
        return 0
    else
        echo -e "${YELLOW}[警告] 发现依赖不一致，请检查上述问题${NC}"
        return 1
    fi
}

# 安全审计
run_audit() {
    echo -e "${BLUE}[INFO] 运行安全审计...${NC}"
    
    if ! command -v pip-audit &> /dev/null; then
        echo -e "${YELLOW}[警告] pip-audit 未安装，正在安装...${NC}"
        pip install pip-audit>=2.6.0
    fi
    
    echo -e "${BLUE}[INFO] 审计 requirements.txt...${NC}"
    pip-audit -r requirements.txt || true
    
    echo -e "${BLUE}[INFO] 审计完成${NC}"
}

# 主函数
main() {
    case "${1:-help}" in
        compile)
            compile_requirements
            ;;
        check)
            check_consistency
            ;;
        upgrade)
            upgrade_requirements
            ;;
        sync)
            sync_requirements
            ;;
        audit)
            run_audit
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}[错误] 未知命令: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
