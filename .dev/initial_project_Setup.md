# 🛠️ scaffold_readmitrx.sh

``` bash
#!/bin/bash

# Assumes you're inside an empty GitHub repo directory like ReadmitRx/

# Top-level files
touch README.md requirements.txt requirements.lock.txt pyproject.toml setup.py pytest.ini
touch .pre-commit-config.yaml Makefile

# CI/CD
mkdir -p .github/workflows
touch .github/workflows/ci.yml

# Core project folders
mkdir -p app
mkdir -p readmitrx/pipeline
mkdir -p readmitrx/scoring
mkdir -p readmitrx/cluster
mkdir -p readmitrx/routing
mkdir -p readmitrx/utils

mkdir -p models
mkdir -p scripts
mkdir -p tests
mkdir -p docs

# Entry point (optional)
touch streamlit_app.py

# Init Python modules
touch readmitrx/__init__.py
touch readmitrx/pipeline/__init__.py
touch readmitrx/scoring/__init__.py
touch readmitrx/cluster/__init__.py
touch readmitrx/routing/__init__.py
touch readmitrx/utils/__init__.py

```
🧪 2. Set Up Pre-commit Hooks

``` yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy

  - repo: https://github.com/pre-commit/mirrors-pytest
    rev: v8.1.1
    hooks:
      - id: pytest
```

```bash
pip install pre-commit
pre-commit install
```

# 🔒 3. Dependency Management

Add these to requirements.txt
```bash
# Core ML + Viz
pandas==2.2.2
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
matplotlib==3.8.4
seaborn==0.13.2

# Dev Tools
black==24.3.0
ruff==0.4.4
mypy==1.10.0
pytest==8.1.1
pre-commit==3.7.0
```

```bash
pip install -r requirements.txt
pip freeze > requirements.lock.txt
```


# 🛠️ 4. GitHub Actions CI
Create .github/workflows/ci.yml:

```yaml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: 3.10

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run pre-commit checks
        run: pre-commit run --all-files
```


```bash
tree readmitrx/ -L 2
```
