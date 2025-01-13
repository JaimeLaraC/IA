# ============================================
# 1. Importación de Librerías Mejoradas
# ============================================
import os  # NUEVO: Importar la librería os para manejar directorios
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import xgboost as xgb

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

import optuna

# ============================================
# 1.1 Solicitar Nombre del Modelo al Usuario y Crear Directorios
# ============================================
model_name = input("Ingrese el nombre con el que desea guardar el modelo (sin extensión): ").strip()
if not model_name:
    model_name = 'modelo_xgboost_binario_sin_empate'  # Valor por defecto si el usuario no ingresa nada
    print(f"No se ingresó nombre. Se usará el nombre por defecto: {model_name}")
else:
    print(f"El modelo se guardará en la carpeta: models/{model_name}/")

# Definir la ruta completa donde se guardará el modelo
modelo_dir = os.path.join('models', model_name)  # NUEVO: Crear la ruta para el modelo

# Crear la carpeta MODELO si no existe
os.makedirs(modelo_dir, exist_ok=True)  # NUEVO: Crear la carpeta MODELO y la subcarpeta model_name

# ============================================
# 2. Carga y Limpieza de Datos Mejorada
# ============================================
df = pd.read_csv('data/resultados_premier_league.csv')
print("Columnas del DataFrame:", df.columns)

# Convertir columnas con '%' a float en [0,1]
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
        print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame.")

# Convertir a numéricas el resto de las columnas numéricas
columnas_numericas_puras = [
    'goles_equipo_local','goles_equipo_visitante',
    'puntos_acumulados_local','puntos_acumulados_visitante',
    'wins_last10_local','draws_last10_local','losses_last10_local',
    'wins_last10_visitante','draws_last10_visitante','losses_last10_visitante',
    'goles_favor_last10_local','goles_contra_last10_local',
    'goles_favor_last10_visitante','goles_contra_last10_visitante',
    'Shots on Goal_local','Shots off Goal_local','Total Shots_local','Blocked Shots_local',
    'Shots insidebox_local','Shots outsidebox_local','Fouls_local','Corner Kicks_local',
    'Offsides_local','Yellow Cards_local','Red Cards_local','Goalkeeper Saves_local',
    'Total passes_local','Passes accurate_local','Passes %_local',
    'Shots on Goal_visitante','Shots off Goal_visitante','Total Shots_visitante','Blocked Shots_visitante',
    'Shots insidebox_visitante','Shots outsidebox_visitante','Fouls_visitante','Corner Kicks_visitante',
    'Offsides_visitante','Yellow Cards_visitante','Red Cards_visitante','Goalkeeper Saves_visitante',
    'Total passes_visitante','Passes accurate_visitante','Passes %_visitante'
]
for col in columnas_numericas_puras:
    if col in df.columns:
        df[col] = df[col].replace(r'^\s*$', 0, regex=True).astype(float)
    else:
        print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame.")

# Verificar tipos tras la limpieza
print("\nTipos de datos tras limpieza:")
print(df.dtypes)

# ============================================
# 3. Definición de la Variable Objetivo SIN Empate
# ============================================
# Crear la columna 'resultado' como antes
df['resultado'] = np.where(
    df['goles_equipo_local'] > df['goles_equipo_visitante'], 'Local Gana',
    np.where(df['goles_equipo_local'] < df['goles_equipo_visitante'], 'Visitante Gana', 'Empate')
)

# Eliminar filas donde el resultado sea 'Empate'
df = df[df['resultado'] != 'Empate'].reset_index(drop=True)

# Volvemos a codificar la variable objetivo con solo 2 clases:
label_encoder = LabelEncoder()
df['resultado_encoded'] = label_encoder.fit_transform(df['resultado'])  
# Ahora label_encoder.classes_ debería ser: ['Local Gana', 'Visitante Gana']

np.save(os.path.join(modelo_dir, 'classes.npy'), label_encoder.classes_)  # NUEVO: Guardar las clases en la carpeta del modelo

# ============================================
# 4. Ordenar por fecha y crear 'temporada'
# ============================================
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').reset_index(drop=True)

if 'temporada' not in df.columns:
    df['temporada'] = df['fecha'].dt.year

# ============================================
# 5. Ingeniería de Características Avanzada
# ============================================
# Definir las características a utilizar
features = [
    'puntos_acumulados_local',
    'puntos_acumulados_visitante',
    'wins_last10_local', 'draws_last10_local', 'losses_last10_local',
    'wins_last10_visitante', 'draws_last10_visitante', 'losses_last10_visitante',
    'goles_favor_last10_local', 'goles_contra_last10_local',
    'goles_favor_last10_visitante', 'goles_contra_last10_visitante',
    'head2head_local_wins', 'head2head_local_draws', 'head2head_local_losses',
    'head2head_visitante_wins', 'head2head_visitante_draws', 'head2head_visitante_losses',
    'puntos_balance', 'goles_balance',
    'Shots on Goal_local', 'Shots off Goal_local', 'Total Shots_local', 'Blocked Shots_local',
    'Shots insidebox_local', 'Shots outsidebox_local', 'Fouls_local', 'Corner Kicks_local',
    'Offsides_local', 'Ball Possession_local', 'Yellow Cards_local', 'Red Cards_local',
    'Goalkeeper Saves_local', 'Total passes_local', 'Passes accurate_local', 'Passes %_local',
    'Shots on Goal_visitante', 'Shots off Goal_visitante', 'Total Shots_visitante', 'Blocked Shots_visitante',
    'Shots insidebox_visitante', 'Shots outsidebox_visitante', 'Fouls_visitante', 'Corner Kicks_visitante',
    'Offsides_visitante', 'Ball Possession_visitante', 'Yellow Cards_visitante', 'Red Cards_visitante',
    'Goalkeeper Saves_visitante', 'Total passes_visitante', 'Passes accurate_visitante', 'Passes %_visitante'
]

X = df[features]
y = df['resultado_encoded']

# ============================================
# 6. Preprocesamiento y Escalado con Pipeline
# ============================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), features),
    ],
    remainder='passthrough'
)

# Pipeline con SMOTE, FeatureSelection y XGBClassifier
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('feature_selection', SelectFromModel(xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))),
    ('classifier', xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])

# ============================================
# 7. Eliminación de la División del Conjunto de Datos (Train/Test)
# ============================================
# MODIFICADO: Se elimina la división en train_df y test_df
# Se utilizará todo el conjunto de datos para entrenar el modelo
X_train = X
y_train = y

print("\nTamaño de X_train y y_train:", X_train.shape, y_train.shape)

# Verificar valores faltantes
print("\nValores faltantes en X_train antes de la imputación:")
print(X_train.isnull().sum())

# Columnas con todos los valores faltantes en X_train (probablemente ninguna si el preprocesamiento es correcto)
missing_cols = X_train.columns[X_train.isnull().all()].tolist()
if missing_cols:
    features = [f for f in features if f not in missing_cols]
    X_train = X_train[features]
    print("\nCaracterísticas actualizadas después de eliminar las columnas faltantes:")
    print(features)

# Actualizar el preprocessor con la lista de features “vivas” si se eliminaron columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), features),
    ],
    remainder='passthrough'
)
pipeline.set_params(preprocessor=preprocessor)

# ============================================
# 8. Optimización de Hiperparámetros con Optuna
# ============================================
def objective(trial):
    param = {
        'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'classifier__max_depth': trial.suggest_int('max_depth', 3, 15),
        'classifier__learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'classifier__subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'classifier__colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'classifier__gamma': trial.suggest_uniform('gamma', 0, 0.5),
        'classifier__reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
        'classifier__reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
    }
    pipeline.set_params(**param)

    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='f1_macro'  # seguimos con f1_macro para binario
    ).mean()

    return score

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
def imprimir_parcial(study, trial):
    print(f"Trial #{trial.number} F1-macro: {trial.value}")

study.optimize(objective, n_trials=1000, timeout=7200, callbacks=[imprimir_parcial])

print("Mejores parámetros:", study.best_params)
print("Mejor f1_macro:", study.best_value)

best_params = {f"classifier__{k}": v for k, v in study.best_params.items()}
pipeline.set_params(**best_params)

# ============================================
# 9. Entrenamiento del Modelo con los Mejores Parámetros
# ============================================
pipeline.fit(X_train, y_train)

# ============================================
# 10. Eliminación de la Evaluación en Test
# ============================================
# MODIFICADO: Se elimina la evaluación en el conjunto de prueba

# ============================================
# 11. Interpretabilidad con SHAP
# ============================================
import shap

# Crear un sub-pipeline que excluya SMOTE y Feature Selection
preprocess_and_scale = Pipeline(steps=[
    ('preprocessor', pipeline.named_steps['preprocessor']),
    ('scaler', pipeline.named_steps['scaler'])
])

# Transformar todo el conjunto de datos
X_train_preprocessed = preprocess_and_scale.transform(X_train)

# Aplicar selección de características
feature_selection = pipeline.named_steps['feature_selection']
X_train_selected = feature_selection.transform(X_train_preprocessed)

# Verificar las formas
print(f"Forma de X_train_preprocessed: {X_train_preprocessed.shape}")
print(f"Forma de X_train_selected: {X_train_selected.shape}")
print(f"Características esperadas por el clasificador: {pipeline.named_steps['classifier'].get_booster().num_features()}")

# Crear el explainer de SHAP usando el clasificador entrenado
classifier = pipeline.named_steps['classifier']
explainer = shap.Explainer(classifier)

# Calcular los valores SHAP
shap_values = explainer(X_train_selected)

# Visualizar el resumen de SHAP
shap.summary_plot(shap_values, X_train_selected, feature_names=features)

# ============================================
# 12. Guardado del Modelo en la Estructura de Carpetas
# ============================================
joblib.dump(pipeline, os.path.join(modelo_dir, f'{model_name}.pkl'))  # MODIFICACIÓN: Uso del nombre y ruta proporcionados
print(f"\nEntrenamiento completado. Modelo guardado en '{os.path.join(modelo_dir, f'{model_name}.pkl')}'.")
    
# ============================================
# 13. Funciones de Predicción Optimizada Mejoradas
# ============================================
def generar_caracteristicas_optimizado(nombre_equipo_local, nombre_equipo_visitante, df_equipos, features):
    """
    Genera un conjunto de características para un partido específico
    entre un equipo local y uno visitante.
    """
    local_data = df_equipos[df_equipos['nombre_equipo'] == nombre_equipo_local]
    visitante_data = df_equipos[df_equipos['nombre_equipo'] == nombre_equipo_visitante]

    if local_data.empty:
        raise ValueError(f"El equipo local '{nombre_equipo_local}' no se encontró en 'df_equipos'.")
    if visitante_data.empty:
        raise ValueError(f"El equipo visitante '{nombre_equipo_visitante}' no se encontró en 'df_equipos'.")

    caracteristicas = {}

    for feature in features:
        if feature.endswith('_local'):
            valor = local_data.iloc[0][feature]
            caracteristicas[feature] = valor if not pd.isnull(valor) else 0
        elif feature.endswith('_visitante'):
            valor = visitante_data.iloc[0][feature]
            caracteristicas[feature] = valor if not pd.isnull(valor) else 0
        else:
            # En este ejemplo, todas las características son específicas de local o visitante
            caracteristicas[feature] = 0

    df_caracteristicas = pd.DataFrame([caracteristicas])
    return df_caracteristicas

def predecir_resultado_optimizado(nombre_equipo_local, nombre_equipo_visitante, df_equipos, features, modelo):
    """
    Predice el resultado (Local Gana o Visitante Gana).
    """
    clases = np.load(os.path.join(modelo_dir, 'classes.npy'), allow_pickle=True)  # NUEVO: Cargar las clases desde la carpeta del modelo

    X_sample = generar_caracteristicas_optimizado(
        nombre_equipo_local,
        nombre_equipo_visitante,
        df_equipos,
        features
    )
    pred_encoded = modelo.predict(X_sample)
    return clases[pred_encoded][0]

# ============================================
# 14. Ejemplo de Uso (sin empate)
# ============================================
if __name__ == "__main__":  # CORRECCIÓN: Uso correcto de __name__ == "__main__"
    # Suponiendo que 'df_equipos' se construye del mismo modo,
    # solo que ya no tendríamos Empate en el dataset
    equipos_uniques = pd.concat([df['nombre_equipo_local'], df['nombre_equipo_visitante']]).unique()
    caracteristicas_equipos = []

    for eq in equipos_uniques:
        df_local = df[df['nombre_equipo_local'] == eq].tail(1)
        df_visit = df[df['nombre_equipo_visitante'] == eq].tail(1)
        
        if not df_local.empty and not df_visit.empty:
            caracteristicas_equipos.append({
                'nombre_equipo': eq,
                'puntos_acumulados_local': df_local['puntos_acumulados_local'].values[0],
                'puntos_acumulados_visitante': df_visit['puntos_acumulados_visitante'].values[0],
                'wins_last10_local': df_local['wins_last10_local'].values[0],
                'draws_last10_local': df_local['draws_last10_local'].values[0],
                'losses_last10_local': df_local['losses_last10_local'].values[0],
                'goles_favor_last10_local': df_local['goles_favor_last10_local'].values[0],
                'goles_contra_last10_local': df_local['goles_contra_last10_local'].values[0],
                'head2head_local_wins': df_local['head2head_local_wins'].values[0],
                'head2head_local_draws': df_local['head2head_local_draws'].values[0],
                'head2head_local_losses': df_local['head2head_local_losses'].values[0],
                'wins_last10_visitante': df_visit['wins_last10_visitante'].values[0],
                'draws_last10_visitante': df_visit['draws_last10_visitante'].values[0],
                'losses_last10_visitante': df_visit['losses_last10_visitante'].values[0],
                'goles_favor_last10_visitante': df_visit['goles_favor_last10_visitante'].values[0],
                'goles_contra_last10_visitante': df_visit['goles_contra_last10_visitante'].values[0],
                'head2head_visitante_wins': df_visit['head2head_visitante_wins'].values[0],
                'head2head_visitante_draws': df_visit['head2head_visitante_draws'].values[0],
                'head2head_visitante_losses': df_visit['head2head_visitante_losses'].values[0],
                'puntos_balance': df_local['puntos_balance'].values[0],
                'goles_balance': df_local['goles_balance'].values[0],
                'Shots on Goal_local': df_local['Shots on Goal_local'].values[0],
                'Shots off Goal_local': df_local['Shots off Goal_local'].values[0],
                'Total Shots_local': df_local['Total Shots_local'].values[0],
                'Blocked Shots_local': df_local['Blocked Shots_local'].values[0],
                'Shots insidebox_local': df_local['Shots insidebox_local'].values[0],
                'Shots outsidebox_local': df_local['Shots outsidebox_local'].values[0],
                'Fouls_local': df_local['Fouls_local'].values[0],
                'Corner Kicks_local': df_local['Corner Kicks_local'].values[0],
                'Offsides_local': df_local['Offsides_local'].values[0],
                'Ball Possession_local': df_local['Ball Possession_local'].values[0],
                'Yellow Cards_local': df_local['Yellow Cards_local'].values[0],
                'Red Cards_local': df_local['Red Cards_local'].values[0],
                'Goalkeeper Saves_local': df_local['Goalkeeper Saves_local'].values[0],
                'Total passes_local': df_local['Total passes_local'].values[0],
                'Passes accurate_local': df_local['Passes accurate_local'].values[0],
                'Passes %_local': df_local['Passes %_local'].values[0],
                'Shots on Goal_visitante': df_visit['Shots on Goal_visitante'].values[0],
                'Shots off Goal_visitante': df_visit['Shots off Goal_visitante'].values[0],
                'Total Shots_visitante': df_visit['Total Shots_visitante'].values[0],
                'Blocked Shots_visitante': df_visit['Blocked Shots_visitante'].values[0],
                'Shots insidebox_visitante': df_visit['Shots insidebox_visitante'].values[0],
                'Shots outsidebox_visitante': df_visit['Shots outsidebox_visitante'].values[0],
                'Fouls_visitante': df_visit['Fouls_visitante'].values[0],
                'Corner Kicks_visitante': df_visit['Corner Kicks_visitante'].values[0],
                'Offsides_visitante': df_visit['Offsides_visitante'].values[0],
                'Ball Possession_visitante': df_visit['Ball Possession_visitante'].values[0],
                'Yellow Cards_visitante': df_visit['Yellow Cards_visitante'].values[0],
                'Red Cards_visitante': df_visit['Red Cards_visitante'].values[0],
                'Goalkeeper Saves_visitante': df_visit['Goalkeeper Saves_visitante'].values[0],
                'Total passes_visitante': df_visit['Total passes_visitante'].values[0],
                'Passes accurate_visitante': df_visit['Passes accurate_visitante'].values[0],
                'Passes %_visitante': df_visit['Passes %_visitante'].values[0]
            })

    df_equipos = pd.DataFrame(caracteristicas_equipos)
    
    # Verificar columnas
    missing_columns_equipos = set(features) - set(df_equipos.columns)
    if missing_columns_equipos:
        print(f"Faltan columnas en 'df_equipos': {missing_columns_equipos}")
        os.exit(1)
    else:
        print("Todas las columnas necesarias están presentes en 'df_equipos'.")

    # ============================================
    # Nueva Sección: Imprimir Nombres de Equipos
    # ============================================
    
    # Obtener la lista de equipos únicos
    lista_equipos = df_equipos['nombre_equipo'].unique()
    
    # Ordenar la lista alfabéticamente para mejor legibilidad
    lista_equipos_sorted = sorted(lista_equipos)
    
    # Imprimir los nombres de los equipos
    print("\n=== Equipos Disponibles para Predicción ===")
    for equipo in lista_equipos_sorted:
        print(f"- {equipo}")
    print("===========================================\n")
    
    # ============================================
    # Bucle para Solicitar Predicciones
    # ============================================
    
    while True:
        # Solicitar al usuario los nombres de los equipos
        equipo_local = input("Ingrese el nombre del equipo local (o 'salir' para terminar): ").strip()
        if equipo_local.lower() == 'salir':
            print("Saliendo del programa.")
            break
        
        equipo_visitante = input("Ingrese el nombre del equipo visitante (o 'salir' para terminar): ").strip()
        if equipo_visitante.lower() == 'salir':
            print("Saliendo del programa.")
            break
        
        # Validar que los equipos existan
        if equipo_local not in df_equipos['nombre_equipo'].values:
            print(f"El equipo local '{equipo_local}' no se encontró en los datos. Por favor, inténtelo de nuevo.\n")
            continue
        if equipo_visitante not in df_equipos['nombre_equipo'].values:
            print(f"El equipo visitante '{equipo_visitante}' no se encontró en los datos. Por favor, inténtelo de nuevo.\n")
            continue
        
        # Realizar la predicción
        try:
            resultado = predecir_resultado_optimizado(
                equipo_local,
                equipo_visitante,
                df_equipos,
                features,
                pipeline
            )
            print(f"\nPredicción: {equipo_local} vs {equipo_visitante} => {resultado}\n")
        except Exception as e:
            print(f"Error al predecir el resultado: {e}\n")
