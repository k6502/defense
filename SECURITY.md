# Security Policy

This repository provides an aircraft recognition pipeline (Python training + Rust runtime). Security is taken seriously. If you discover a vulnerability, please follow the process below.

## Reporting a Vulnerability

- Preferred: Create a private GitHub Security Advisory for this repository.
- Alternative: Open a GitHub Issue and mark it as confidential if available, or contact the maintainers via their GitHub accounts (owner: k6502).
- Do not publish details publicly until the issue is resolved.

When reporting include:

- A clear summary of the issue and impact.
- Steps to reproduce (environment, commands).
- A minimal proof‑of‑concept if available.
- Affected component(s) and version(s) (e.g., training dependencies, runtime crate version).
- Logs, stack traces, and config snippets as needed.

## Response & Disclosure Timeline

- Acknowledgement: within 3 business days.
- Triage and classification: as soon as possible.
- Fix or mitigation plan: targeted within 30 days depending on severity.
- Full resolution / public disclosure: coordinated; target within 90 days for non-critical issues (may be shorter for critical vulnerabilities).

## Fixing & Patching

- Maintainers will publish fixes as commits/PRs and, where applicable, release tagged versions.
- For Python dependency issues, maintainers will update training/requirements.txt and recommend repro steps.
- For Rust/runtime issues, maintainers will update Cargo.toml and publish new crate releases or runtime binaries.

Recommended local checks:

- Python: pip-audit or safety (install in venv)
  - python3 -m venv .venv && source .venv/bin/activate
  - pip install pip-audit
  - pip-audit -r training/requirements.txt
- Rust: cargo-audit
  - cargo install cargo-audit
  - cd runtime && cargo audit

## Disclosure Policy

- Coordinated disclosure only: do not publicly disclose vulnerabilities until a fix or mitigation is available or maintainers agree to disclosure.
- If a reporter requests anonymity, maintainers will respect reasonable requests.

## Sensitive Data

- This project does not include production secrets. Do not commit private keys, credentials, or sensitive datasets to the repository.
- If you find leaked secrets, report them via the above channels.

## Contact & Maintainers

- Repository owner: k6502 (use GitHub Security Advisory or repo contact mechanisms).
- If necessary, attach an encrypted report using PGP; include a public key on the advisory or issue if requested.

Thank you for helping keep this project secure.
