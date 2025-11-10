# ACE + Browser-Use Demos

Two demos comparing automation **without** vs **with** ACE learning.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   uv sync --extra demos
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Run demos:**

### 🌐 Domain Checker
   ```bash
   # Baseline (no learning)
   uv run python examples/browser-use/baseline_domain_checker.py

   # With ACE (learns and improves)
   uv run python examples/browser-use/ace_domain_checker.py
   ```

### 📝 Form Filler
   ```bash
   # Baseline (no learning)
   uv run python examples/browser-use/baseline_form_filler.py

   # With ACE (learns and improves)
   uv run python examples/browser-use/ace_form_filler.py
   ```

## 📊 What You'll See

**Baseline:** Same approach every time, repeats mistakes

**ACE:** Learns after each task, gets more efficient, builds strategies

## 🎯 Key Point

ACE transforms static automation into intelligent, learning systems.