#!/usr/bin/env python3
"""Fix escaped backticks in reviewer.mdc"""

file_path = ".cursor/rules/reviewer.mdc"

with open(file_path, encoding="utf-8") as f:
    content = f.read()

# Replace escaped backticks with normal backticks
content = content.replace("\\`\\`\\`", "```")
content = content.replace("\\`", "`")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed backticks successfully")
