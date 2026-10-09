"""
Script para generar el dataset limpio de rentas de departamentos en CDMX a partir de los zips de scrapings crudos.

Replica la limpieza del notebook de la semana 3. Solo hace falta correrlo si se quiere
regenerar data/rentas_cdmx_limpio.csv desde los archivos originales (no incluidos en el repo.).

Para usarlo, se usa el Makefile con el comando make data.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from rentas_cdmx.config import CLEAN_DATA_PATH, RAW_DATA_DIR

FUENTES = ['i24', 'lamudi', 'propiedades', 'remax']

COLS = ['id', 'source_type', 'operation_type', 'property_type', 'currency', 'asking_price',
        'm2_built', 'no_bedrooms', 'no_full_bathrooms', 'no_half_bathrooms', 'no_parking_spaces',
        'lat', 'lon', 'state', 'del_mun', 'neighborhood']

NUM_COLS = ['asking_price', 'm2_built', 'no_bedrooms', 'no_full_bathrooms',
            'no_half_bathrooms', 'no_parking_spaces', 'lat', 'lon']

TIPOS_DEPTO = ['Departamento', 'Loft', 'Estudio', 'Penthouse', 'Dúplex']

TIPO_CAMBIO = {'USD': 17, 'EUR': 21}

COLS_SALIDA = ['asking_price', 'm2_built', 'no_bedrooms', 'no_full_bathrooms',
               'no_parking_spaces', 'lat', 'lon', 'del_mun', 'property_type', 'source_type']


def parse_args():
    parser = argparse.ArgumentParser(description='Limpia los scrapings crudos y genera el CSV para entrenamiento.')
    parser.add_argument('--raw-dir', type=Path, default=RAW_DATA_DIR,
                        help='Carpeta con los zips crudos de cada portal (default: data/raw)')
    parser.add_argument('--output', type=Path, default=CLEAN_DATA_PATH,
                        help='Ruta del CSV limpio')
    return parser.parse_args()


def load_raw(raw_dir):
    return pd.concat(
        [pd.read_csv(raw_dir / f'{f}.zip', usecols=COLS, low_memory=False) for f in FUENTES],
        ignore_index=True,
    )


def filter_rentals(df):
    df = df[
        df['operation_type'].str.lower().isin(['rent', 'renta'])
        & df['property_type'].isin(TIPOS_DEPTO)
        & df['state'].isin(['Ciudad de México', 'DF / CDMX'])
    ].copy()

    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors='coerce')
    return df


def clean(df):
    # Convertir dolares y euros a pesos mexicanos
    for moneda, tc in TIPO_CAMBIO.items():
        df['asking_price'] = np.where(df['currency'] == moneda, df['asking_price'] * tc, df['asking_price'])

    # Limpiar nombre de alcaldías y quitar fuera de CDMX
    df['del_mun'] = df['del_mun'].replace({'Alvaro Obregón': 'Álvaro Obregón'})
    df = df[~df['del_mun'].isin(['Ciudad de México', 'otra'])]

    # Quitar potenciales duplicados según features importantes
    cols_dup = ['asking_price', 'm2_built', 'no_bedrooms', 'no_full_bathrooms',
                'no_parking_spaces', 'del_mun', 'neighborhood']
    df = df.drop_duplicates(['source_type', 'id']).drop_duplicates(cols_dup)

    # Quitar nulos en features importantes
    df = df.dropna(subset=['asking_price', 'm2_built', 'no_bedrooms', 'no_full_bathrooms', 'lat', 'lon'])

    # Quitar outliers en features importantes
    df = df[
        df['asking_price'].between(3_000, 100_000)
        & df['m2_built'].between(20, 300)
        & (df['no_bedrooms'] <= 4)
        & df['no_full_bathrooms'].between(1, 3)
        & ~(df['no_parking_spaces'] > 4)
    ]

    precio_m2 = df['asking_price'] / df['m2_built']
    lim_inf, lim_sup = precio_m2.quantile([0.01, 0.99])
    df = df[precio_m2.between(lim_inf, lim_sup)]
    df = df[df['lat'].between(19.0, 19.6) & df['lon'].between(-99.4, -98.9)]

    return df


def main():
    args = parse_args()
    df = clean(filter_rentals(load_raw(args.raw_dir)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[COLS_SALIDA].to_csv(args.output, index=False)
    print(f'{len(df):,} registros guardados en {args.output}')


if __name__ == '__main__':
    main()
