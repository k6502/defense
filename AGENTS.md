# defense — Aircraft recognition (agents)

## Purpose

This repository implements an aircraft recognition pipeline for defense use:

- Python-based training and data tooling (training/).
- Rust-based runtime for inference/production (runtime/).

## Repository layout

- training/ — Python code, data, requirements, and tests.
  - training/src/ — Python source: data.py, neural.py, export.py, utils.py
  - training/data/fgvc-aircraft-2013b/ — dataset and evaluation helpers
  - training/requirements.txt — Python dependencies
- runtime/ — Rust crate for runtime/inference.
  - runtime/src/, Cargo.toml — Rust source and build config
- .venv/ — recommended Python virtual environment (local, gitignored)

## Environment (recommended)

Use a Python virtual environment in the project root.

Commands (Linux):

- Create venv:
  - python3 -m venv .venv
- Activate:
  - source .venv/bin/activate
- Install Python deps:
  - pip install --upgrade pip
  - pip install -r training/requirements.txt

## Python (training) usage

- Data and training code live in training/src/.
- Typical commands (inside activated venv):
  - Run training or experiments:
    - python training/src/neural.py
  - Export models:
    - python training/src/export.py
  - Run tests:
    - pip install pytest
    - pytest -q training/src/tests

Inspect training/data/ for dataset manifests and the fgvc-aircraft-2013b folder for example data and evaluation scripts.

## Rust (runtime) usage

- Build runtime (Linux):
  - cd runtime
  - cargo build --release
- Run tests/examples:
  - cargo test
  - cargo run --release --example <name>  (if examples exist)

Binary output is under runtime/target/.

## Notes

- Keep dataset files and large artifacts out of version control.
- Use the venv in project root for all Python work to keep environments reproducible.
- Python source is authoritative for model development; Rust runtime is for deployment/inference.

```// filepath: /home/kptr/Projects/defense/AGENTS.md
# defense — Aircraft recognition (agents)

## Purpose
This repository implements an aircraft recognition pipeline for defense use:
- Python-based training and data tooling (training/).
- Rust-based runtime for inference/production (runtime/).

## Repository layout
- training/ — Python code, data, requirements, and tests.
  - training/src/ — Python source: data.py, neural.py, export.py, utils.py
  - training/data/fgvc-aircraft-2013b/ — dataset and evaluation helpers
  - training/requirements.txt — Python dependencies
- runtime/ — Rust crate for runtime/inference.
  - runtime/src/, Cargo.toml — Rust source and build config
- .venv/ — recommended Python virtual environment (local, gitignored)

## Environment (recommended)
Use a Python virtual environment in the project root.

Commands (Linux):
- Create venv:
  - python3 -m venv .venv
- Activate:
  - source .venv/bin/activate
- Install Python deps:
  - pip install --upgrade pip
  - pip install -r training/requirements.txt

## Python (training) usage
- Data and training code live in training/src/.
- Typical commands (inside activated venv):
  - Run training or experiments:
    - python training/src/neural.py
  - Export models:
    - python training/src/export.py
  - Run tests:
    - pip install pytest
    - pytest -q training/src/tests

Inspect training/data/ for dataset manifests and the fgvc-aircraft-2013b folder for example data and evaluation scripts.

## Rust (runtime) usage
- Build runtime (Linux):
  - cd runtime
  - cargo build --release
- Run tests/examples:
  - cargo test
  - cargo run --release --example <name>  (if examples exist)

Binary output is under runtime/target/.

## Notes
- Keep dataset files and large artifacts out of version control.
- Use the venv in project root for all Python work to keep environments reproducible.
- Python source is authoritative for model development; Rust runtime is for deployment/inference.
