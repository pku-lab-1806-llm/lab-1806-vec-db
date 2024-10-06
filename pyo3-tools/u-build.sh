#!/bin/bash

if [ ! -d ".venv" ]; then
    uv venv -p 3.10
fi

uv pip install -r pyo3-tools/requirements.txt

source .venv/bin/activate

maturin build --release --out dist
