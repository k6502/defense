
# defense

Aircraft recognition pipeline (training + runtime) for research and evaluation.

## Repository layout

- training/ — Python training, data tooling, and tests
  - training/src/ — Python source: data.py, neural.py, export.py, utils.py
  - training/data/fgvc-aircraft-2013b/ — example dataset manifests and eval scripts
  - training/requirements.txt — Python dependencies
- runtime/ — Rust runtime crate for inference (Cargo.toml, src/)
- .venv/ — recommended Python virtual environment (local)

Put your own input images/samples in examples/ at repo root.

## Quickstart (Linux)

1. Create and activate venv:
   - python3 -m venv .venv
   - source .venv/bin/activate
2. Install Python deps:
   - pip install --upgrade pip
   - pip install -r training/requirements.txt
3. Provide dataset or sample images under examples/ or training/data/.

## Python — training & export

- Run training / experiments (from repo root, venv active):
  - python training/src/neural.py
- Export trained model:
  - python training/src/export.py
- Run Python tests:
  - pip install pytest
  - pytest -q training/src/tests

Inspect training/src/ for data loading and model code.

## Rust — runtime / inference

- Build:
  - cd runtime
  - cargo build --release
- Run tests:
  - cargo test
- Binary output:
  - runtime/target/(debug|release)

Use the Rust runtime for deployment or fast local inference.

## Notes

- Keep large datasets/artifacts out of version control.
- training/ contains the authoritative Python model/dev code; runtime/ contains deployment code.
- Use .venv for reproducible Python environment.
