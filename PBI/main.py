import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_countries(filename="countries.csv", base_dir=BASE_DIR):
    csv_path = os.path.join(base_dir, filename)
    print(f"[INFO] Cargando datos desde: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normaliza nombres de columnas (a veces vienen con espacios raros)
    df.columns = [c.strip() for c in df.columns]
    return df


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Convierte una serie (posiblemente con $, comas, espacios) a numérica."""
    if s.dtype == "O":
        s = (
            s.astype(str)
             .str.replace(r"[\$,]", "", regex=True)   # quita $ y comas
             .str.strip()
        )
    return pd.to_numeric(s, errors="coerce")


def prepare_data(df, target_col="GDP ($ per capita)", drop_col="Region"):
    df = df.copy()

    # 1) Eliminar Region si existe
    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])

    # 2) Comprobar target
    if target_col not in df.columns:
        # Mensaje útil para depurar si el nombre exacto difiere
        cols_preview = ", ".join(df.columns[:15])
        raise ValueError(
            f"No existe la columna target '{target_col}'. "
            f"Primeras columnas detectadas: {cols_preview} ..."
        )

    # 3) Convertir target a numérico (por si viene como texto)
    df[target_col] = _to_numeric_series(df[target_col])

    # 4) Eliminar filas con NaN (tal como pide el enunciado)
    df = df.dropna().copy()

    # 5) Separar X, y
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 6) Quedarse SOLO con features numéricas (descarta Country u otras categóricas)
    #    Si alguna numérica venía como texto, la intentamos convertir.
    for col in X.columns:
        if X[col].dtype == "O":
            X[col] = _to_numeric_series(X[col])

    X = X.select_dtypes(include=[np.number]).copy()

    # 7) Asegurar que aún hay datos
    if len(X) == 0 or len(y) == 0:
        raise ValueError(
            "Tras limpiar datos, no quedan filas. "
            "Revisa si el CSV tiene muchos textos/no-numéricos o el target no se pudo convertir."
        )
    if X.shape[1] == 0:
        raise ValueError(
            "Tras filtrar features numéricas, no queda ninguna columna en X. "
            "En ese caso habría que codificar variables categóricas (OneHotEncoder)."
        )

    return X, y


def standardize_features(X: pd.DataFrame):
    # Para MLP es MUY recomendable estandarizar (todas las features en escala similar)
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def build_models(alpha=0.1, max_iter=30000, random_state=42):
    common = dict(
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )

    mlp1 = MLPRegressor(hidden_layer_sizes=(200,), **common)
    mlp2 = MLPRegressor(hidden_layer_sizes=(50, 50), **common)
    return mlp1, mlp2



def evaluate_models(X_scaled, y, k=5, random_state=42, alpha=0.1, max_iter=30000):
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    mlp1, mlp2 = build_models(alpha=alpha, max_iter=max_iter, random_state=random_state)

    # neg_mean_squared_error -> luego lo pasamos a MSE positivo
    s1 = cross_val_score(mlp1, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")
    s2 = cross_val_score(mlp2, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")

    mse1 = -np.mean(s1)
    mse2 = -np.mean(s2)
    return mse1, mse2, -s1, -s2  # devolvemos MSE por fold ya en positivo


def main():
    df = load_countries("countries.csv")

    X, y = prepare_data(df, target_col="GDP ($ per capita)", drop_col="Region")

    print(f"[INFO] Filas finales (sin NaN): {len(y)}")
    print(f"[INFO] Nº de features numéricas usadas: {X.shape[1]}")

    X_scaled = standardize_features(X)

    mse1, mse2, folds1, folds2 = evaluate_models(
        X_scaled, y, k=5, random_state=42, alpha=0.1, max_iter=30000
    )

    print("\n[RESULT] MLP1 (1 capa oculta de 200 neuronas)")
    print(f"  MSE medio (k=5): {mse1:.4f}")
    print(f"  MSE por fold: {folds1}")

    print("\n[RESULT] MLP2 (2 capas ocultas de 50 y 50 neuronas)")
    print(f"  MSE medio (k=5): {mse2:.4f}")
    print(f"  MSE por fold: {folds2}")

    if mse1 < mse2:
        print("\n[CONCLUSION] MLP1 domina (menor MSE medio).")
    elif mse2 < mse1:
        print("\n[CONCLUSION] MLP2 domina (menor MSE medio).")
    else:
        print("\n[CONCLUSION] Empate (mismo MSE medio).")


if __name__ == "__main__":
    main()
