.PHONY: setup run test test-e2e clean

setup:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords', quiet=True)" || true

run:
	python harness/main.py --repo . \
		--issue "$$( [ -f issue.txt ] && cat issue.txt || ([ -f ISSUE.md ] && cat ISSUE.md) || echo "$$ISSUE" )"

test:
	python -m pytest tests/ -m "not e2e"

test-e2e:
	python -m pytest tests/e2e/

clean:
	rm -rf .harness/ __pycache__/ */__pycache__/ */*/__pycache__/ *.pyc .pytest_cache
