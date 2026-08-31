.PHONY: install test check-config status

install:
	python -m pip install -r requirements.txt
	python -m pip install -e . --no-deps

test:
	python -m pytest

check-config:
	PYTHONPATH=src:. python -c "from pathlib import Path; from credit_risk.config import load_project_configs; load_project_configs(Path.cwd()); print('configuration contract: OK')"

status:
	git status --short --branch
