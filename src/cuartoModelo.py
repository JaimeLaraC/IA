#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Sklearn, Imblearn, etc.
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Model Selection y Métricas
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

# XGBoost y Optuna
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback

SEED = 42
np.random.seed(SEED)

# ============================================
# 1. Selección del archivo CSV
# ============================================
def seleccionar_archivo_csv(data_dir='data'):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No se encontraron archivos CSV en '{data_dir}'.")
        exit(1)

    print(f"Archivos CSV en '{data_dir}':")
    for i, cf in enumerate(csv_files, 1):
        print(f"{i}. {cf}")
    while True:
        sel = input(f"Seleccione (1-{len(csv_files)}): ")
        try:
            sel = int(sel)
            if 1 <= sel <= len(csv_files):
                return os.path.join(data_dir, csv_files[sel - 1]), csv_files[sel - 1]
        except:
            pass
        print("Entrada inválida.")

data_dir = 'data'
ruta_csv, archivo_seleccionado = seleccionar_archivo_csv(data_dir)

# ============================================
# 2. Crear Directorio del Modelo
# ============================================
def crear_directorio_modelo():
    model_name = input("Nombre del modelo (sin extensión): ").strip()
    if not model_name:
        model_name = 'modelo_xgboost_binario_sin_empate'
        print(f"No se ingresó nombre. Usando: {model_name}")
    modelo_dir = os.path.join('models', model_name)
    os.makedirs(modelo_dir, exist_ok=True)
    return model_name, modelo_dir

model_name, modelo_dir = crear_directorio_modelo()

# ============================================
# 3. Carga y Limpieza de Datos
# ============================================
df = pd.read_csv(ruta_csv)
print("Columnas del DataFrame:", df.columns)

# -- Convertir columnas con '%' a float (si existen)
for col in ['Ball Possession_local','Passes %_local','Ball Possession_visitante','Passes %_visitante']:
    if col in df.columns:
        df[col] = df[col].replace(r'^\s*$', '0%', regex=True)
        df[col] = df[col].str.replace('%','').astype(float)/100.0

# -- Convertir a numéricas
cols_numericas = [
    'goles_equipo_local','goles_equipo_visitante',
    'puntos_acumulados_local','puntos_acumulados_visitante',
    'wins_last10_local','draws_last10_local','losses_last10_local',
    'wins_last10_visitante','draws_last10_visitante','losses_last10_visitante',
    'goles_favor_last10_local','goles_contra_last10_local',
    'goles_favor_last10_visitante','goles_contra_last10_visitante',
    'head2head_local_wins','head2head_local_draws','head2head_local_losses',
    'head2head_visitante_wins','head2head_visitante_draws','head2head_visitante_losses',
    'puntos_balance','goles_balance','posicion_tabla_local','posicion_tabla_visitante',
    'dias_descanso_local','dias_descanso_visitante','elo_rating_local','elo_rating_visitante',
    'valor_mercado_local','valor_mercado_visitante','jugadores_lesionados_local','jugadores_lesionados_visitante',
    'titulares_sancionados_local','titulares_sancionados_visitante','diff_posicion_tabla','diff_elo_rating','diff_dias_descanso'
]
for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Crear variable objetivo sin empate
df['resultado'] = np.where(
    df['goles_equipo_local'] > df['goles_equipo_visitante'], 'Local Gana',
    np.where(df['goles_equipo_local'] < df['goles_equipo_visitante'], 'Visitante Gana', 'Empate')
)
# Eliminar empates
df = df[df['resultado'] != 'Empate'].reset_index(drop=True)

# Encoding
le = LabelEncoder()
df['resultado_encoded'] = le.fit_transform(df['resultado'])
classes_path = os.path.join(modelo_dir, 'classes.npy')
np.save(classes_path, le.classes_)  # Guardar las clases

# Ordenar por fecha (si existe la columna)
if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.sort_values('fecha').reset_index(drop=True)

# ============================================
# 4. Definir Features
# ============================================
features = [
    'puntos_acumulados_local','puntos_acumulados_visitante',
    'wins_last10_local','draws_last10_local','losses_last10_local',
    'wins_last10_visitante','draws_last10_visitante','losses_last10_visitante',
    'goles_favor_last10_local','goles_contra_last10_local',
    'goles_favor_last10_visitante','goles_contra_last10_visitante',
    'head2head_local_wins','head2head_local_draws','head2head_local_losses',
    'head2head_visitante_wins','head2head_visitante_draws','head2head_visitante_losses',
    'puntos_balance','goles_balance','posicion_tabla_local','posicion_tabla_visitante',
    'dias_descanso_local','dias_descanso_visitante','elo_rating_local','elo_rating_visitante',
    'valor_mercado_local','valor_mercado_visitante','jugadores_lesionados_local','jugadores_lesionados_visitante',
    'titulares_sancionados_local','titulares_sancionados_visitante','diff_posicion_tabla','diff_elo_rating','diff_dias_descanso'
]
features_presentes = [f for f in features if f in df.columns]
X = df[features_presentes]
y = df['resultado_encoded']

print("\nFeatures usadas para entrenar:")
print(features_presentes)
print(f"Tamaño total de X: {X.shape}, y: {y.shape}")

# ============================================
# 5. FunctionTransformer con función normal
#    para evitar error de pickling
# ============================================
def df_to_numpy_func(data):
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return data

df_to_numpy = FunctionTransformer(df_to_numpy_func, validate=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), features_presentes),
    ],
    remainder='passthrough'
)

pipeline_sin_xgb = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('to_numpy', df_to_numpy),
    ('smote', SMOTE(random_state=SEED, k_neighbors=5))
])

# ============================================
# 6. Objective para Optuna (TimeSeriesSplit)
#    con control para saltar folds monoclase
#    y rango de búsqueda reducido
# ============================================
def objective(trial):
    params = {
        # Rango más pequeño (para ser más rápido)
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),

        'use_label_encoder': False,
        'eval_metric': 'logloss',  # binario
        'random_state': SEED,
        'n_jobs': -1,
        # Sin GPU => 'hist' o 'auto'
        'tree_method': 'hist',
        'objective': 'binary:logistic'
    }

    # Menos splits => más rápido
    tscv = TimeSeriesSplit(n_splits=3)
    f1_scores = []

    for train_idx, valid_idx in tscv.split(X, y):
        X_tr, y_tr_ = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]

        if len(np.unique(y_tr_)) < 2 or len(np.unique(y_val)) < 2:
            continue

        # Fit con SMOTE
        X_tr_res, y_tr_res = pipeline_sin_xgb.fit_resample(X_tr, y_tr_)

        # Transform valid sin SMOTE
        X_val_pre = pipeline_sin_xgb[:-1].transform(X_val)

        xgb_clf = xgb.XGBClassifier(**params)

        # early stopping reducido a 20
        xgb_clf.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val_pre, y_val.to_numpy())],
            early_stopping_rounds=20,
            verbose=False,
            callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")]
        )

        y_pred = xgb_clf.predict(X_val_pre)
        f1_val = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1_val)

    if len(f1_scores) == 0:
        return 0.0

    return np.mean(f1_scores)

print("\nIniciando la optimización de hiperparámetros con Optuna (puede demorar).")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)
# n_trials menor => más rápido
study.optimize(objective, n_trials=50, timeout=1800, show_progress_bar=True)

print("\nMejores parámetros hallados por Optuna:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")
print(f"Mejor F1 en CV (media de folds válidos): {study.best_value:.4f}")

best_params = study.best_params

# ============================================
# 7. Entrenamiento Final con TODOS los datos
# ============================================
print("\nEntrenando FINAL con TODOS los datos...")
X_all_res, y_all_res = pipeline_sin_xgb.fit_resample(X, y)

xgb_final = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1,
    tree_method='hist',       # si tienes GPU, cámbialo a 'gpu_hist'
    objective='binary:logistic'
)

xgb_final.fit(X_all_res, y_all_res)
print("Entrenamiento completado con TODOS los datos.")

# ============================================
# 8. Guardar "pipeline + xgb_final"
# ============================================
from sklearn.pipeline import Pipeline

pipeline_final = Pipeline(steps=[
    ('preprocessor', pipeline_sin_xgb.named_steps['preprocessor']),
    ('scaler', pipeline_sin_xgb.named_steps['scaler']),
    ('to_numpy', pipeline_sin_xgb.named_steps['to_numpy']),
    ('smote', pipeline_sin_xgb.named_steps['smote']),
    ('classifier', xgb_final)
])

ruta_modelo = os.path.join(modelo_dir, f"{model_name}.pkl")
joblib.dump(pipeline_final, ruta_modelo)

print(f"\nModelo final guardado en: {ruta_modelo}")
print(f"Clases (LabelEncoder): {classes_path}")
print("¡Proceso finalizado!")
