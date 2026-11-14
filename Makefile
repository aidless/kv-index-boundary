# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 本仓无既有单元测试目录；一键入口就是可执行验收题库。
.PHONY: eval

eval:
	python3 eval/run_cases.py
