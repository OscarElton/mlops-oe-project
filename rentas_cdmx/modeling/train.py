"""
Entrena un Random Forest para estimar la renta mensual (MXN) de departamentos en CDMX
y registra la corrida en MLflow (parámetros, métricas, artefactos y modelo).

Para modificar alguno de los hiperparámetros, se puede correr de la siguiente manera desde la raíz
del proyecto (todos son opcionales):
    python -m rentas_cdmx.modeling.train --n-estimators 200 --min-samples-leaf 5 --max-depth 20 --max-features 1.0 --seed 42
"""

import argparse
import json
import random
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from rentas_cdmx.config import CLEAN_DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

NUM_FEATS = ['m2_built', 'no_bedrooms', 'no_full_bathrooms', 'no_parking_spaces', 'lat', 'lon']
TARGET = 'asking_price'
STRATIFY_COL = 'del_mun'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Entrena un Random Forest de renta de departamentos en CDMX y lo registra en MLflow.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Número de árboles del bosque')
    parser.add_argument('--min-samples-leaf', type=int, default=5,
                        help='Mínimo de observaciones por hoja; valores mayores regularizan el modelo')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Profundidad máxima de cada árbol (None = sin límite)')
    parser.add_argument('--max-features', type=float, default=1.0,
                        help='Fracción de variables evaluadas en cada split (0, 1]')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para el split y el modelo')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Proporción de datos reservada para prueba')
    parser.add_argument('--data-path', type=Path, default=CLEAN_DATA_PATH,
                        help='CSV limpio generado por rentas_cdmx/dataset.py')
    parser.add_argument('--experiment-name', default=MLFLOW_EXPERIMENT_NAME,
                        help='Nombre del experimento en MLflow')
    parser.add_argument('--run-name', default=None,
                        help='Nombre de la corrida en MLflow (por defecto se genera con los hiperparámetros)')
    parser.add_argument('--tracking-uri', default=MLFLOW_TRACKING_URI,
                        help='URI del tracking server de MLflow')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    return pd.read_csv(path)


def split_data(df, test_size, seed):
    X = df[NUM_FEATS]
    y = np.log(df[TARGET])
    return train_test_split(X, y, test_size=test_size, stratify=df[STRATIFY_COL], random_state=seed)


def build_model(args):
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=-1,
    )
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('reg', rf)])


def regression_metrics(y_log, pred_log):
    real, pred = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(pred - real) / real
    return {
        'MdAPE': float(np.median(ape)),
        'MAPE': float(mean_absolute_percentage_error(real, pred)),
        'PPE20': float(np.mean(ape <= 0.20)),
        'MAE': float(mean_absolute_error(real, pred)),
    }


def evaluate(model, X_train, y_train, X_test, y_test):
    metrics = regression_metrics(y_test, model.predict(X_test))
    metrics['MdAPE_train'] = regression_metrics(y_train, model.predict(X_train))['MdAPE']
    metrics['gap_MdAPE'] = metrics['MdAPE'] - metrics['MdAPE_train']
    return metrics


def plot_real_vs_pred(y_test, pred_log):
    real, pred = np.exp(y_test), np.exp(pred_log)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real, pred, alpha=0.2, s=8)
    ax.plot([3_000, 100_000], [3_000, 100_000], 'r--', label='predicción perfecta')
    ax.set(xlabel='Renta real (MXN)', ylabel='Renta predicha (MXN)', title='Real vs predicho (test)')
    ax.legend()
    return fig


def error_by_rent_range(y_test, pred_log):
    real, pred = np.exp(y_test), np.exp(pred_log)
    errores = pd.DataFrame({'real': real, 'error_pct': np.abs(pred - real) / real})
    errores['rango_renta'] = pd.qcut(errores['real'], 5).astype(str)
    return errores.groupby('rango_renta', sort=False)['error_pct'].agg(['count', 'median', 'mean'])


def log_artifacts(model, X_train, X_test, y_test, metrics):
    pred_log = model.predict(X_test)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp/'metrics.json').write_text(json.dumps(metrics, indent=2))
        error_by_rent_range(y_test, pred_log).to_csv(tmp/'error_por_rango_renta.csv')
        mlflow.log_artifact(tmp/'metrics.json')
        mlflow.log_artifact(tmp/'error_por_rango_renta.csv')

    fig = plot_real_vs_pred(y_test, pred_log)
    mlflow.log_figure(fig, 'real_vs_predicho.png')
    plt.close(fig)

    mlflow.sklearn.log_model(
        model,
        name='modelo',
        input_example=X_train.head(5),
        serialization_format='pickle',
    )


def default_run_name(args):
    return (f'rf_n{args.n_estimators}_leaf{args.min_samples_leaf}'
            f'_depth{args.max_depth}_feat{args.max_features}')


def main():
    args = parse_args()
    set_seed(args.seed)

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or default_run_name(args)):
        params = {
            'n_estimators': args.n_estimators,
            'min_samples_leaf': args.min_samples_leaf,
            'max_depth': args.max_depth,
            'max_features': args.max_features,
            'seed': args.seed,
            'test_size': args.test_size,
            'n_registros': len(df),
            'features': ','.join(NUM_FEATS),
            'target': f'log({TARGET})',
        }
        for key, value in params.items():
            mlflow.log_param(key, value)

        model = build_model(args).fit(X_train, y_train)
        metrics = evaluate(model, X_train, y_train, X_test, y_test)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        log_artifacts(model, X_train, X_test, y_test, metrics)

    print(json.dumps({'params': params, 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
