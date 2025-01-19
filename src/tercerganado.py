#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tercerganado.py (versión refactorizada sin inputs por consola)

import os
import sys
import pandas as pd
import numpy as np
import joblib

def predecir_jornada(model_name, csv_file, jornada_num):
    """
    Carga el modelo `model_name` desde la carpeta 'models/',
    lee el CSV `csv_file` desde la carpeta 'data/',
    filtra los registros de la jornada `jornada_num`
    y retorna un DataFrame con las columnas:
      [id_partido, fecha, nombre_equipo_local, nombre_equipo_visitante, Predicción, Confianza]

    También imprime en pantalla el TOP 4 partidos con mayor confianza.
    """
    # =============================
    # 1. Rutas y validaciones
    # =============================
    modelos_dir = "models"
    model_path = os.path.join(modelos_dir, model_name, f"{model_name}.pkl")
    classes_path = os.path.join(modelos_dir, model_name, "classes.npy")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"No se encontró el archivo de clases: {classes_path}")

    data_dir = "data"
    jornada_csv = os.path.join(data_dir, csv_file)
    if not os.path.exists(jornada_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {jornada_csv}")

    # =============================
    # 2. Cargar el modelo y las clases
    # =============================
    try:
        pipeline = joblib.load(model_path)
        print(f"Modelo cargado exitosamente desde: {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo '{model_path}': {e}")
        sys.exit(1)

    try:
        clases = np.load(classes_path, allow_pickle=True)
        print(f"Clases cargadas exitosamente desde: {classes_path}: {clases}\n")
    except Exception as e:
        print(f"Error al cargar las clases '{classes_path}': {e}")
        sys.exit(1)

    # =============================
    # 3. Cargar y preparar el DataFrame
    # =============================
    df = pd.read_csv(jornada_csv)
    print(f"Datos cargados desde '{jornada_csv}'. Filas: {len(df)}")

    # Normalizar nombres de columnas (opcional)
    df.columns = df.columns.str.strip().str.lower()

    # Convertir 'fecha' a numérico para obtener la jornada
    if 'fecha' not in df.columns:
        raise KeyError("No existe la columna 'fecha' en el CSV para filtrar jornada.")
    df['jornada_numero'] = pd.to_numeric(df['fecha'], errors='coerce').astype('Int64')
    if df['jornada_numero'].isna().any():
        raise ValueError("Existen valores no numéricos en 'fecha' que impiden asignar la 'jornada_numero'.")

    # Filtrar la jornada
    df_jornada = df[df['jornada_numero'] == jornada_num].copy()
    if df_jornada.empty:
        raise ValueError(f"No se encontraron registros para la jornada {jornada_num} en '{csv_file}'.")

    # =============================
    # 4. Revisar columnas esenciales
    # =============================
    info_partido_cols = [
        'id_partido', 'fecha', 'nombre_equipo_local', 'nombre_equipo_visitante'
    ]
    for col in info_partido_cols:
        if col not in df_jornada.columns:
            raise KeyError(f"No se encontró la columna '{col}' en el CSV. Abortando.")

    # Definir features esperadas por el modelo
    features = [
        # Local
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

        # Visitante
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

        # Posiciones
        "posicion_tabla_local",
        "posicion_tabla_visitante",
        "diff_posicion_tabla",

        # Días de descanso
        "dias_descanso_local",
        "dias_descanso_visitante",
        "diff_dias_descanso",

        # ELO
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

    missing_cols = set(features) - set(df_jornada.columns)
    if missing_cols:
        raise KeyError(f"Faltan columnas en el CSV para poder predecir: {missing_cols}")

    df_features = df_jornada[features].copy()

    # Convertir todo a numérico y rellenar NaNs con 0, por si acaso
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)

    # =============================
    # 5. Predecir
    # =============================
    try:
        probas = pipeline.predict_proba(df_features)
        pred_encoded = np.argmax(probas, axis=1)  # Índice de la clase mayor prob
        pred_labels = np.array(clases)[pred_encoded]  # Etiquetas de clase
    except Exception as e:
        print(f"Error al predecir: {e}")
        sys.exit(1)

    # Calcular confianza
    confianzas = np.max(probas, axis=1)

    # =============================
    # 6. Construir DataFrame de salida
    # =============================
    df_resultados = df_jornada[info_partido_cols].copy()
    df_resultados['Predicción'] = pred_labels
    df_resultados['Confianza'] = confianzas

    # Ordenar por confianza desc
    df_resultados = df_resultados.sort_values('Confianza', ascending=False).reset_index(drop=True)

    # Imprimir TOP 4
    df_top4 = df_resultados.head(4)
    print("\n=== TOP 4 PARTIDOS CON MAYOR PROBABILIDAD DE ACERTAR ===\n")
    for idx, row in df_top4.iterrows():
        print(f"Partido ID {row['id_partido']} | Jornada: {row['fecha']} | "
              f"{row['nombre_equipo_local']} vs {row['nombre_equipo_visitante']} => "
              f"{row['Predicción']} (Confianza: {row['Confianza']:.2f})")
    print("\n===================================\n")

    # Retornar todas las predicciones
    return df_resultados


def main():
    """
    Ejemplo de ejecución manual (sin interacción):
    Para ejecutarlo directamente, editar las variables:
      - model_name
      - csv_file
      - jornada_num
    antes de correr `python tercerganado.py`.
    """
    model_name = "nombre_del_modelo"  # nombre de la carpeta en models/
    csv_file = "temporada_308_2024.csv"  # ejemplo
    jornada_num = 10  # jornada a predecir

    try:
        df_resultados = predecir_jornada(model_name, csv_file, jornada_num)
        # Podrías guardar df_resultados a un CSV si quisieras, por ej.:
        # df_resultados.to_csv("outputs/predicciones_jornada_10.csv", index=False)
        print(f"\nPredicciones completas:\n{df_resultados.head()}\n")
    except Exception as e:
        print(f"\nOcurrió un error en 'main': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
