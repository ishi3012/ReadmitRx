#!/bin/bash

# Assumes you're inside an empty GitHub repo directory like ReadmitRx/

# Top-level files
touch README.md requirements.txt requirements.lock.txt pyproject.toml setup.py pytest.ini
touch .pre-commit-config.yaml Makefile
touch .gitignore

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



echo "Project scaffold created."
