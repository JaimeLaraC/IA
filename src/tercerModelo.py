# ============================================
# 1. Importación de Librerías
# ============================================
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.impute import SimpleImputer

import xgboost as xgb
import optuna

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 2. Selección del Archivo CSV para Entrenar
# ============================================
data_dir = 'data'  # carpeta donde están tus CSV
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

if not csv_files:
    print(f"No se encontraron archivos CSV en la carpeta '{data_dir}'. Asegúrate de que la carpeta contenga al menos un archivo CSV.")
    exit(1)

print("Archivos CSV disponibles en la carpeta 'data':")
for idx, file in enumerate(csv_files, start=1):
    print(f"{idx}. {file}")

while True:
    try:
        seleccion = int(input(f"Seleccione el número del archivo CSV que desea usar para entrenar (1-{len(csv_files)}): "))
        if 1 <= seleccion <= len(csv_files):
            archivo_seleccionado = csv_files[seleccion - 1]
            break
        else:
            print(f"Por favor, ingrese un número entre 1 y {len(csv_files)}.")
    except ValueError:
        print("Entrada inválida. Por favor, ingrese un número entero.")

ruta_csv = os.path.join(data_dir, archivo_seleccionado)
print(f"Ha seleccionado el archivo: {archivo_seleccionado}")

# ============================================
# 3. Solicitar Nombre del Modelo al Usuario y Crear Directorios
# ============================================
model_name = input("Ingrese el nombre con el que desea guardar el modelo (sin extensión): ").strip()
if not model_name:
    model_name = 'modelo_xgboost_binario_sin_empate'
    print(f"No se ingresó nombre. Se usará el nombre por defecto: {model_name}")
else:
    print(f"El modelo se guardará en la carpeta: models/{model_name}/")

modelo_dir = os.path.join('models', model_name)
os.makedirs(modelo_dir, exist_ok=True)

# ============================================
# 4. Carga y Limpieza de Datos
# ============================================
df = pd.read_csv(ruta_csv)
print("Columnas del DataFrame:", df.columns)

# -- Si hubiera columnas con '%', convertir a float (opcional, según tus datos)
columnas_con_porcentaje = [
    'Ball Possession_local',
    'Passes %_local',
    'Ball Possession_visitante',
    'Passes %_visitante'
]
for col in columnas_con_porcentaje:
    if col in df.columns:
        df[col] = df[col].replace(r'^\s*$', '0%', regex=True)
        df[col] = df[col].str.replace('%', '').astype(float) / 100.0
    else:
        pass  # o print(f"Advertencia: La columna '{col}' no está en el DataFrame.")

# ============================================
# 5. Definición de la Variable Objetivo (sin empates)
# ============================================
# Requerimos las columnas 'goles_equipo_local' y 'goles_equipo_visitante'.
df['resultado'] = np.where(
    df['goles_equipo_local'] > df['goles_equipo_visitante'], 'Local Gana',
    np.where(df['goles_equipo_local'] < df['goles_equipo_visitante'], 'Visitante Gana', 'Empate')
)

# Eliminamos filas con empates
df = df[df['resultado'] != 'Empate'].reset_index(drop=True)

# Codificamos la variable objetivo: 0 -> Local Gana, 1 -> Visitante Gana
label_encoder = LabelEncoder()
df['resultado_encoded'] = label_encoder.fit_transform(df['resultado'])
np.save(os.path.join(modelo_dir, 'classes.npy'), label_encoder.classes_)

# ============================================
# 6. Ordenar por fecha (si la columna 'fecha' existe)
# ============================================
if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.sort_values('fecha').reset_index(drop=True)
else:
    print("Advertencia: No se encontró la columna 'fecha' para orden temporal.")
    # Si no hay fecha, no se puede usar TimeSeriesSplit de forma temporal, 
    # pero lo dejaremos igual como ejemplo.

# ============================================
# 7. Definir las Features
# ============================================
# Usamos todas las que mencionaste en tu JSON final:
feature_cols = [
    # Equipo local
    "puntos_acumulados_local",
    "wins_last10_local",
    "draws_last10_local",
    "losses_last10_local",
    "goles_favor_last10_local",
    "goles_contra_last10_local",
    "current_win_streak_local",
    "current_loss_streak_local",
    "puntos_ultimos5_local",
    "goles_favor_ultimos5_local",
    "goles_contra_ultimos5_local",
    "prom_goles_favor_ultimos5_local",
    "prom_goles_contra_ultimos5_local",
    "clean_sheets_last10_local",
    "btts_last10_local",
    "over25_last10_local",
    "clean_sheets_last5_local",
    "btts_last5_local",
    "over25_last5_local",
    "home_win_rate_local",
    "away_win_rate_local",

    # Equipo visitante
    "puntos_acumulados_visitante",
    "wins_last10_visitante",
    "draws_last10_visitante",
    "losses_last10_visitante",
    "goles_favor_last10_visitante",
    "goles_contra_last10_visitante",
    "current_win_streak_visitante",
    "current_loss_streak_visitante",
    "puntos_ultimos5_visitante",
    "goles_favor_ultimos5_visitante",
    "goles_contra_ultimos5_visitante",
    "prom_goles_favor_ultimos5_visitante",
    "prom_goles_contra_ultimos5_visitante",
    "clean_sheets_last10_visitante",
    "btts_last10_visitante",
    "over25_last10_visitante",
    "clean_sheets_last5_visitante",
    "btts_last5_visitante",
    "over25_last5_visitante",
    "home_win_rate_visitante",
    "away_win_rate_visitante",

    # Head2head
    "head2head_local_wins",
    "head2head_local_draws",
    "head2head_local_losses",
    "head2head_visitante_wins",
    "head2head_visitante_draws",
    "head2head_visitante_losses",

    # Balances
    "puntos_balance",
    "goles_balance",

    # Posiciones (simuladas)
    "posicion_tabla_local",
    "posicion_tabla_visitante",
    "diff_posicion_tabla",

    # Días de descanso
    "dias_descanso_local",
    "dias_descanso_visitante",
    "diff_dias_descanso",

    # Elo (simulado)
    "elo_rating_local",
    "elo_rating_visitante",
    "diff_elo_rating",

    # Otros
    "valor_mercado_local",
    "valor_mercado_visitante",
    "jugadores_lesionados_local",
    "jugadores_lesionados_visitante",
    "titulares_sancionados_local",
    "titulares_sancionados_visitante"
]

# Verificamos que existan en el DataFrame, omitimos si faltan
feature_cols_presentes = [f for f in feature_cols if f in df.columns]
faltantes = set(feature_cols) - set(feature_cols_presentes)
if faltantes:
    print(f"Advertencia: Estas columnas no están en el DataFrame y se omitirán: {faltantes}")

X = df[feature_cols_presentes]
y = df['resultado_encoded']

print("\nFeatures usadas para entrenar:")
print(feature_cols_presentes)

# Convertimos a numérico (por seguridad)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# ============================================
# 8. Construir el Pipeline (Imputación, Escalado, SMOTE, XGB)
# ============================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), feature_cols_presentes),
    ],
    remainder='passthrough'
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('classifier', xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='gpu_hist',   # Habilita GPU (ajusta si no tienes GPU)
        predictor='gpu_predictor' # Habilita GPU para predicción
    ))
])

# ============================================
# 9. Entrenar sin dividir en test
# ============================================
X_train, y_train = X, y
print("\nTamaño de X_train y y_train:", X_train.shape, y_train.shape)

# ============================================
# 10. Optimización de Hiperparámetros con Optuna
#      (usando TimeSeriesSplit para CV)
# ============================================
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10)
    }
    pipeline.set_params(**{f"classifier__{k}": v for k, v in params.items()})
    pipeline.set_params(
        classifier__tree_method='gpu_hist',
        classifier__predictor='gpu_predictor'
    )

    # Validación cruzada temporal
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro')
    return scores.mean()

print("\nIniciando la optimización de hiperparámetros con Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, timeout=7200)  # Ajusta a tu gusto

print("Mejores parámetros encontrados:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
print(f"Mejor F1-macro: {study.best_value}")

# Los aplicamos al pipeline
best_params = {f"classifier__{k}": v for k, v in study.best_params.items()}
best_params['classifier__tree_method'] = 'gpu_hist'
best_params['classifier__predictor'] = 'gpu_predictor'

# ============================================
# 11. Entrenamiento final con TODOS los datos
# ============================================
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)
print("\nEntrenamiento final completado con todas las filas (sin test set).")

# ============================================
# 12. Guardar el Modelo y las Clases
# ============================================
ruta_modelo = os.path.join(modelo_dir, f"{model_name}.pkl")
joblib.dump(pipeline, ruta_modelo)
print(f"Modelo guardado exitosamente en '{ruta_modelo}'.")
print("¡Proceso finalizado!")
