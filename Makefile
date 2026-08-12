PYTHON ?= python

.PHONY: setup format-check lint typecheck test test-slow verify-cpu

setup:
	$(PYTHON) -m pip install -e .[cpu,dev,export]

format-check:
	$(PYTHON) -m screen2action.tools.verify format

lint:
	$(PYTHON) -m screen2action.tools.verify lint

typecheck:
	$(PYTHON) -m screen2action.tools.verify typecheck

test:
	$(PYTHON) -m screen2action.tools.verify test

test-slow:
	$(PYTHON) -m pytest -m slow

verify-cpu:
	$(PYTHON) -m screen2action.tools.verify all

train-tiny:
	$(PYTHON) -m screen2action.tools.train --stage tiny_cpu --device cpu
