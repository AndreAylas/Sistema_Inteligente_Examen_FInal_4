import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_countries(filename="countries.csv",base_dir=BASE_DIR):
    csv_path=os.path.join(base_dir,filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontrol el archivo: {csv_path}")
    return pd.read_csv(csv_path)

def prepare_data(df,target_col="GDP",drop_col="Region"):
    
    # 1.- eliminar la columa region
    if drop_col in df.columns:
        df=df.drop(columns=[drop_col])
    
    # 2.- eliminar filas con NaN
    df=df.dropna().copy()
    
    # 3.-  comprobar que este la columna GDP
    if target_col not in df.columns:
        raise ValueError("No existe la columna PBI dentro del archivo csv .....")
    
    #4 Separar X,y
    X=df.drop(columns=[target_col])
    y=df[target_col]
    
    # 5.- Convertir todo X a numerico
    X_num=X.apply(pd.to_numeric,errors="coerce")
    tmp=pd.concat([X_num,y],axis=1).dropna()
    X_num=tmp.drop(columns=[target_col])
    y=tmp[target_col]
    return X_num,y


def standarize_features(x):
    scaler=StandardScaler()
    return scaler.fit_transform(x)

def build_models(alpha=0.1, max_itera=3000, random_state=42):
    mlp1=MLPRegressor(
        hidden_layer_sizes=(200,),
        alpha= alpha,
        max_iter=max_itera,
        random_state=random_state
    )
    mlp2=MLPRegressor(
        hidden_layer_sizes=(50,50),
        alpha=alpha,
        max_iter=max_itera,
        random_state=random_state
    )
    return mlp1, mlp2

def evaluate_models(X_scaled,y,k=5,random_state=42):
    cv=KFold(n_splits=k,shuffle=True,random_state=random_state)
    mlp1,mlp2=build_models()
    # scoring: neg_mean_squared_error -> luego lo pasamos a MSE positivo
    scores_mlp1 = cross_val_score(mlp1, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")
    scores_mlp2 = cross_val_score(mlp2, X_scaled, y, cv=cv, scoring="neg_mean_squared_error")

    mse_mlp1 = -np.mean(scores_mlp1)
    mse_mlp2 = -np.mean(scores_mlp2)

    return mse_mlp1, mse_mlp2, scores_mlp1, scores_mlp2

def main():
    df=load_countries("countries.csv")
    
    X, y= prepare_data(df)
        
    X_scaled=standarize_features(X)
    
    # Evaluación con validación cruzada k=5
    mse1, mse2, s1, s2 = evaluate_models(X_scaled, y, k=5, random_state=42)

    print("[INFO] Filas finales (sin NaN):", len(y))
    print("[INFO] Nº de variables (features):", X.shape[1])

    print("\n[RESULT] MLP1 (1 capa oculta de 200 neuronas)")
    print("  MSE medio (k=5):", mse1)
    print("  MSE por fold:", (-s1))

    print("\n[RESULT] MLP2 (2 capas ocultas de 50 y 50 neuronas)")
    print("  MSE medio (k=5):", mse2)
    print("  MSE por fold:", (-s2))

    if mse1 < mse2:
        print("\n[CONCLUSION] MLP1 domina (menor MSE medio).")
    elif mse2 < mse1:
        print("\n[CONCLUSION] MLP2 domina (menor MSE medio).")
    else:
        print("\n[CONCLUSION] Empate (mismo MSE medio).")
        
        
if __name__=="__main__":
    main()