# Usage:
#   make paper-assets   build every figure and table used in the paper into paper/assets/
#                       (figures/*.pdf + table_*.tex, same layout as the paper repo)

PYTHON ?= python
ASSETS := paper/assets

.PHONY: help paper-assets
.DEFAULT_GOAL := help

help:
	@echo "make paper-assets   build all paper figures and tables into $(ASSETS)/"

paper-assets:
	rm -rf $(ASSETS)
	cd paper && $(PYTHON) build_paper_assets.py
	@echo "Done: $$(ls $(ASSETS)/figures | wc -l | tr -d " ") figures, $$(ls $(ASSETS)/table_*.tex | wc -l | tr -d " ") tables in $(ASSETS)/"
