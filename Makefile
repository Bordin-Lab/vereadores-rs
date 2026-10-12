.PHONY: validate figures reproduce paper clean

validate:
	python scripts/validate_repository.py

figures:
	python paper/code/build_figures.py
	python scripts/build_integrated_figure.py

reproduce:
	python scripts/reproduce_from_processed.py

paper:
	bash scripts/compile_paper.sh

clean:
	find paper -maxdepth 1 -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' \) -delete
