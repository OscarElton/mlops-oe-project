#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-oe-project
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Generar data/processed/rentas_cdmx_limpio.csv a partir de data/raw
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m rentas_cdmx.dataset


## Entrenar con los hiperparámetros por defecto y registrar en MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m rentas_cdmx.modeling.train


## Correr las 10 corridas del experimento de la semana 4
TRAIN = $(PYTHON_INTERPRETER) -m rentas_cdmx.modeling.train
.PHONY: experiments
experiments:
	$(TRAIN) --run-name 01_baseline
	$(TRAIN) --run-name 02_n_estimators_100  --n-estimators 100
	$(TRAIN) --run-name 03_n_estimators_500  --n-estimators 500
	$(TRAIN) --run-name 04_min_leaf_2        --min-samples-leaf 2
	$(TRAIN) --run-name 05_min_leaf_20       --min-samples-leaf 20
	$(TRAIN) --run-name 06_max_depth_10      --max-depth 10
	$(TRAIN) --run-name 07_max_depth_20      --max-depth 20
	$(TRAIN) --run-name 08_max_features_035  --max-features 0.35
	$(TRAIN) --run-name 09_max_features_07   --max-features 0.7
	$(TRAIN) --run-name 10_combinada         --n-estimators 500 --min-samples-leaf 2 --max-features 0.7


## Abrir la interfaz de MLflow con las corridas registradas
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
