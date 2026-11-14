
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import (
    BREAST_CANCER_FILE,
    DIABETES_FILE
)


BREAST_CANCER_COLUMNS = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

DIABETES_COLUMNS_WITH_ZERO_AS_MISSING = {
    "glucose",
    "bloodpressure",
    "skinthickness",
    "insulin",
    "bmi"
}

def load_breast_cancer():
    

    if not BREAST_CANCER_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {BREAST_CANCER_FILE}"
        )

    df = pd.read_csv(
        BREAST_CANCER_FILE,
        header=None,
        names=BREAST_CANCER_COLUMNS
    )

    
    X = df.drop(columns=["id", "diagnosis"])

    
    y = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    if y.isna().any():
        raise ValueError("Foram encontrados diagnósticos inválidos.")

    return X, y.astype(int)



def load_diabetes():


    if not DIABETES_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DIABETES_FILE}"
        )

    df = pd.read_excel(DIABETES_FILE)

    
    
    column_mapping = {
        column: "".join(
            character.lower()
            for character in str(column)
            if character.isalnum()
        )
        for column in df.columns
    }

    df = df.rename(columns=column_mapping)

    if "outcome" not in df.columns:
        raise ValueError(
            "A coluna Outcome não foi encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    X = df.drop(columns=["outcome"]).copy()

    y = df["outcome"].astype(int)

    
    for column in X.columns:

        if column in DIABETES_COLUMNS_WITH_ZERO_AS_MISSING:

            X[column] = X[column].replace(0, np.nan)

    return X, y



def load_all_datasets():
    
    datasets = {
        "breast_cancer": load_breast_cancer(),
        "diabetes": load_diabetes()
    }

    return datasets


def preprocess_fold(X_train, X_validation):


    imputer = SimpleImputer(strategy="median")

    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)

    X_validation_imputed = imputer.transform(X_validation)

    X_train_scaled = scaler.fit_transform(X_train_imputed)

    X_validation_scaled = scaler.transform(
        X_validation_imputed
    )

    return X_train_scaled, X_validation_scaled


def describe_dataset(name, X, y):
    

    print(f"\n{'=' * 50}")
    print(f"BASE: {name.upper()}")
    print(f"{'=' * 50}")

    print(f"Quantidade de registros: {len(X)}")

    print(f"Quantidade de atributos: {X.shape[1]}")

    print("\nDistribuição das classes:")

    print(y.value_counts().sort_index())

    print(f"\nValores ausentes: {X.isna().sum().sum()}")

    print(f"Registros duplicados: {X.duplicated().sum()}")
