---
name: cybersecurity-skills
description: Structured cybersecurity workflows, vulnerability auditing, secret hygiene, and secure ML model serialization mapped to MITRE ATT&CK and NIST frameworks.
---

# Cybersecurity Skills — Institutional MLOps Security Protocol

Use this skill to audit codebases, verify zero credential leakage, audit dependencies, and prevent code injection or unsafe deserialization vulnerabilities.

## 1. Security Invariants
- **Zero Deserialization Vulnerabilities (CWE-502)**: Strict ban on `pickle`, `joblib`, `shelve`, or untrusted `yaml.load`. Always use native JSON (`model.save_model()`).
- **Secret & API Key Hygiene**: Verify all API tokens (Alpaca, NewsAPI, Hugging Face) are loaded exclusively via `os.getenv()` or `.env`, never hardcoded in source.
- **Dependency Audit**: Regular automated scanning with `pip-audit` and `bandit -r src -ll`.
- **Input Sanitization**: Sanitize all ticker symbols and user inputs with `sanitize_filename()` and strict allowlist regexes.
