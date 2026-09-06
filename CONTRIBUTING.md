# Contributing to Sentilyze

Thank you for your interest in contributing to **Sentilyze**! Sentilyze is an experimental, data-driven multi-agent quantitative trading research platform combining transformer NLP, walk-forward machine learning, and multi-agent quorums.

To maintain institutional code quality, reproducible research, and deterministic safety, please review the following guidelines before submitting contributions.

---

## 🏛️ Core Project Architecture & Contribution Rules

All contributions must adhere to the fundamental architectural principles of Sentilyze:

1. **Primary Foundation & Polyglot Extensibility**:
   - Python is the primary core language for Sentilyze pipelines, ML models, and backtesting.
   - However, **we welcome polyglot innovation**: developers are encouraged to propose or contribute high-performance modules, low-latency execution kernels, or analytical extensions in other languages (such as **Rust, C++, Go, CUDA, or Mojo**).
   - Non-Python modules and extensions will be reviewed, benchmarked for stability/safety, and tested collaboratively by maintainers before integration into the core repository.
2. **Frontend & Dashboard Interface**:
   - The primary reference dashboard is built with **Streamlit** for seamless cloud deployment and research exploration.
   - Complementary client tools, headless API endpoints, terminal UIs, or external visualization bridges are welcome for review.
3. **Secure Model Serialization**:
   - **NEVER use `pickle` or `joblib`** to save or load machine learning models.
   - All XGBoost models must use native `.json` format (`model.save_model()` / `model.load_model()`) in compliance with CodeQL `py/unsafe-deserialization` security policies.
   - PyTorch models must use `torch.load(..., weights_only=True)`.
4. **Live Paper Portfolio Preservation**:
   - `results/paper_portfolio.json` and `results/executed_trades.csv` represent the live paper trading state.
   - **NEVER** reset, wipe, overwrite with dummy mock data, or commit test artifacts over these live state files. All automated tests must use mock brokers or temporary directories (`tmp_path`).
5. **Deterministic Results Source of Truth**:
   - The Streamlit Cloud application reads metrics, portfolios, and SHAP data from `results/`.
   - Local training runs (`mlruns/`) and cached raw data (`data/`) must remain local and ignored by Git.

---

## 🛠️ Development Environment Setup

### 1. Fork & Clone
```bash
git clone https://github.com/yash6810/Sentilyze.git
cd Sentilyze
```

### 2. Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install runtime and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 🧪 Quality Gates & Testing Standards

Before opening a pull request, verify that all quality, security, and unit test gates pass locally:

### 1. Code Formatting (Black)
Sentilyze enforces `black==25.1.0`:
```bash
black .
```

### 2. Linting (Flake8)
Verify that there are no syntax errors or undefined symbols:
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### 3. Security Auditing (Bandit)
Run static application security testing:
```bash
bandit -r src/ -ll
```

### 4. Unit Test Suite (Pytest)
Ensure all tests are passing with zero regressions:
```bash
pytest tests/ -v
```

---

## 🚀 Pull Request Workflow

1. **Branch Naming**: Use descriptive branch names:
   - `feat/add-kalman-filter`
   - `fix/hrp-weight-normalization`
   - `perf/quantize-deep-model`
   - `docs/update-methodology`
2. **Atomic Commits**: Keep commits focused and use Conventional Commit messages:
   - `feat(risk): implement downside semivariance metric`
   - `fix(portfolio): resolve float precision in rebalancer`
   - `test(copilot): add integration tests for multi-turn assistant`
3. **Documentation & Graph Sync**:
   - If adding new modules or algorithms, include docstrings and update tests in `tests/`.
   - Run `graphify update .` after code modifications to refresh the knowledge graph.
4. **Submit PR**: Open a Pull Request against `main` on GitHub detailing your changes, motivation, and test verification results.

---

## 💬 Questions & Community Support

If you have questions, bug reports, or proposals:
- Open an issue in our [GitHub Issue Tracker](https://github.com/yash6810/Sentilyze/issues).
- Connect via Discord alerts or reach out to the maintainer at `yashupadhyay481@gmail.com`.
