# mlops-oe-project

![](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)

Proyecto de la materia TC5061 (MLOps) · Oscar Elton (A01840474)

Este proyecto contiene un modelo que estima la **renta mensual (MXN)** de departamentos en la Ciudad de México a partir de características de las propiedades. Más adelante se probará incluir otras variables de contexto geográfico, como número de escuelas, áreas verdes, transporte, airbnbs, estrato socioeconómico, etc. Cada corrida se registra en MLflow con sus parámetros, métricas, artefactos y el modelo entrenado.

## Estructura

```
├── Makefile           <- Comandos: `make data`, `make train`, `make experiments`, `make mlflow-ui`
├── README.md
├── data
│   ├── processed      <- data limpia, ignoara por tamaño del archivo
│   └── raw            <- Scrapings crudos de cada portal (~660 MB, no se incluyen en el repo)
├── models             <- (vacío ahora) los modelos se están guardando en MLflow
├── notebooks          <- notebooks de explotación de datos
├── reports            <- reportes para mostrar a otros
├── pyproject.toml     <- Metadata del paquete rentas_cdmx
├── requirements.txt   <- Dependencias con versiones fijas
└── rentas_cdmx        <- Código fuente del proyecto
    ├── config.py      <- Rutas del proyecto y configuración de MLflow
    ├── dataset.py     <- Limpieza: data/raw -> data/processed/
    └── modeling
        └── train.py   <- Entrenamiento parametrizado y registro en MLflow
```



## Instalación

Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` instala el módulo `rentas_cdmx` como paquete editable (`-e .`), lo que permite importarlo desde los notebooks.

## Ejecución

Todos los comandos se corren **desde la raíz del proyecto**, porque MLflow guarda los artefactos en `./mlruns`.

Con valores por defecto (reproduce el baseline de la semana 3):

```bash
python -m rentas_cdmx.modeling.train      # o: make train
```

Cambiando hiperparámetros sin editar el código:

```bash
python -m rentas_cdmx.modeling.train --n-estimators 500 --min-samples-leaf 2 --max-features 0.7 --seed 42
```

Para reproducir las 10 corridas registradas en el experimento:

```bash
make experiments
```

`python -m rentas_cdmx.modeling.train --help` muestra todos los argumentos:


| Argumento            | Default                                  | Descripción                                          |
| -------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `--n-estimators`     | 200                                      | Número de árboles                                    |
| `--min-samples-leaf` | 5                                        | Mínimo de observaciones por hoja                     |
| `--max-depth`        | None                                     | Profundidad máxima de cada árbol (None = sin límite) |
| `--max-features`     | 1.0                                      | Fracción de variables evaluadas en cada split        |
| `--seed`             | 42                                       | Semilla para el split y el modelo                    |
| `--test-size`        | 0.3                                      | Proporción de datos de prueba                        |
| `--data-path`        | `data/processed/rentas_cdmx_limpio.csv`  | CSV de entrada                                       |
| `--experiment-name`  | `TC5061_RandomForest_Rentas_CDMX`        | Experimento en MLflow                                |
| `--run-name`         | generado a partir de los hiperparámetros | Nombre de la corrida                                 |
| `--tracking-uri`     | `sqlite:///mlflow.db`                    | Backend de MLflow                                    |




## Ver los resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # o: make mlflow-ui
```

Abre [http://127.0.0.1:5000](http://127.0.0.1:5000), entra al experimento `TC5061_RandomForest_Rentas_CDMX`, selecciona las corridas y da clic en **Compare**.

En cada corrida se registra lo siguiente:

- **Parámetros**: los hiperparámetros, la semilla, `test_size`, el número de registros, las features y el target.
- **Métricas**: con las rentas ya regresadas a pesos (porque el modelo predice `log(renta)`):
  - `MdAPE`: error porcentual absoluto mediano. Es la métrica principal.
  - `MAPE`: error porcentual absoluto promedio.
  - `PPE20`: proporción de predicciones que caen a ±20% de la renta real.
  - `MAE`: error absoluto promedio, en pesos.
  - `MdAPE_train` y `gap_MdAPE`: sirven para medir el sobreajuste. Más de ~3% se podría considerar sobreajuste según lo visto en la materia de Aprendizaje Automático.
- **Artefactos**: `metrics.json`, `real_vs_predicho.png` y `error_por_rango_renta.csv`.
- **Modelo**: el pipeline completo (Imputer + Random Forest), guardado con `mlflow.sklearn.log_model`.



## Reproducibilidad

- La semilla (`--seed`) se fija en tres lugares: `random` y `numpy`, en el `train_test_split` estratificado por alcaldía y en el `random_state` del Random Forest.
- `requirements.txt` fija las versiones exactas con las que se generaron los resultados en Python 3.13.



## Datos

Los datos crudos vienen de algunos sitios inmobiliarios de MX, que pesan unos 660 MB, así que no se incluyen en el repositorio. 

Si tienes los archivos crudos en `data/raw/`, puedes regenerar el CSV con:

```bash
python -m rentas_cdmx.dataset      # o: make data
```

