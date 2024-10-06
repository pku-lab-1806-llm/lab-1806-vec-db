if (-not (Test-Path .venv)) {
    uv venv -p 3.10
}

uv pip install -r pyo3-tools/requirements.txt

.venv\Scripts\activate

maturin build --release --out dist
