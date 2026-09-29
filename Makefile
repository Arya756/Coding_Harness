.PHONY: setup run test clean

# =============================================================================
# AI Harness Hackathon 2026 — Standardised Makefile
# =============================================================================
# Evaluation flow (do not modify the interface):
#
#   git clone <this-repo>
#   cd <this-repo>
#   export AI_API_KEY="<provided-key>"
#   make setup
#   make run            # → interactive: model → target repo → issue → run
#   make test           # → unit/integration tests (no live API calls)
#   make clean          # → remove generated artefacts
#
# API key MUST come from the AI_API_KEY environment variable.
# Source code, Makefile, and .env files must never contain a real key.
# =============================================================================

HARNESS_DIR := harness_hackathon

setup:
	pip install -r $(HARNESS_DIR)/requirements.txt
	python -c "import nltk" 2>/dev/null && python -c "nltk.download('stopwords', quiet=True)" || true

run:
	cd $(HARNESS_DIR) && python -m harness.main

test:
	cd $(HARNESS_DIR) && python -m pytest tests/ -m "not e2e"

clean:
	rm -rf .harness/ __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf $(HARNESS_DIR)/.harness/ $(HARNESS_DIR)/__pycache__/ $(HARNESS_DIR)/*/__pycache__/
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
