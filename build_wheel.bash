#!/bin/bash

source venv_py3_9/bin/activate
uv pip install poetry
poetry build -v

source venv_py3_10/bin/activate
uv pip install poetry
poetry build -v

source venv_py3_11/bin/activate
uv pip install poetry
poetry build -v

source venv_py3_12/bin/activate
uv pip install poetry
poetry build -v

source venv_py3_13/bin/activate
uv pip install poetry
poetry build -v