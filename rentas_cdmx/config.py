from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / 'rentas_cdmx_limpio.csv'

MLFLOW_TRACKING_URI = f'sqlite:///{PROJ_ROOT / "mlflow.db"}'
MLFLOW_EXPERIMENT_NAME = 'TC5061_RandomForest_Rentas_CDMX'
