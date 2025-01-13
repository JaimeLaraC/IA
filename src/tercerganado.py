#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import joblib

def main():
    # =============================
    # 1. Listar modelos disponibles y elegir uno
    # =============================
    modelos_dir = "models"
    print("=== PREDICCIÓN DE RESULTADOS PARA TODAS LAS JORNADAS ===\n")
    
    # Listamos las carpetas dentro de "models" (cada carpeta representa un modelo)
    model_folders = [d for d in os.listdir(modelos_dir) 
                     if os.path.isdir(os.path.join(modelos_dir, d))]
    if not model_folders:
        print(f"No se encontraron modelos en la carpeta '{modelos_dir}'. Abortando.")
        sys.exit(1)

    print("Modelos disponibles:")
    for i, folder in enumerate(model_folders, start=1):
        print(f"{i}. {folder}")

    # Pedimos elegir una carpeta/modelo
    try:
        model_idx = int(input("\nElija el número del modelo que desea usar: ").strip())
        if model_idx < 1 or model_idx > len(model_folders):
            raise ValueError
    except ValueError:
        print("No se eligió un número de modelo válido. Abortando.")
        sys.exit(1)

    model_name = model_folders[model_idx - 1]
    model_path = os.path.join(modelos_dir, model_name, f"{model_name}.pkl")
    classes_path = os.path.join(modelos_dir, model_name, "classes.npy")

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print("Error: modelo o clases no encontrados.")
        sys.exit(1)

    # Cargamos el modelo y las clases
    try:
        pipeline = joblib.load(model_path)
        print(f"\nModelo cargado exitosamente desde: {model_path}")
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
    # 2. Listar CSVs disponibles en la carpeta data y elegir uno
    # =============================
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"La carpeta '{data_dir}' no existe. Abortando.")
        sys.exit(1)

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No se encontraron archivos CSV en la carpeta '{data_dir}'. Abortando.")
        sys.exit(1)

    print("Archivos CSV disponibles:")
    for i, f in enumerate(csv_files, start=1):
        print(f"{i}. {f}")

    try:
        csv_idx = int(input("\nElija el número del archivo CSV de la jornada: ").strip())
        if csv_idx < 1 or csv_idx > len(csv_files):
            raise ValueError
    except ValueError:
        print("No se eligió un número de archivo CSV válido. Abortando.")
        sys.exit(1)

    jornada_csv = os.path.join(data_dir, csv_files[csv_idx - 1])
    if not os.path.exists(jornada_csv):
        print(f"Error: El archivo '{jornada_csv}' no existe.")
        sys.exit(1)

    # Cargamos los datos
    try:
        df = pd.read_csv(jornada_csv)
        print(f"\nDatos cargados exitosamente desde '{jornada_csv}'. Filas: {len(df)}")
        print("Columnas en el CSV:", df.columns.tolist())  # Añadido para diagnóstico
    except Exception as e:
        print(f"Error al leer '{jornada_csv}': {e}")
        sys.exit(1)

    # Normalizar nombres de columnas: eliminar espacios y convertir a minúsculas (opcional)
    df.columns = df.columns.str.strip().str.lower()
    print("Columnas normalizadas:", df.columns.tolist())

    # =============================
    # 3. Verificar y Utilizar 'fecha' como número de jornada
    # =============================
    if 'fecha' not in df.columns:
        print("No existe la columna 'fecha' en el CSV para filtrar jornada. Abortando.")
        sys.exit(1)

    # Convertimos la columna 'fecha' a tipo numérico (int), manejando posibles errores
    try:
        df['jornada_numero'] = pd.to_numeric(df['fecha'], errors='coerce').astype('Int64')
    except Exception as e:
        print(f"Error al convertir 'fecha' a numérico para 'jornada_numero': {e}")
        sys.exit(1)

    if df['jornada_numero'].isna().any():
        print("Existen valores no numéricos en 'fecha' que no pudieron convertirse a 'jornada_numero'. Abortando.")
        sys.exit(1)

    # =============================
    # 4. Listar las jornadas disponibles
    # =============================
    jornadas_unicas = df['jornada_numero'].drop_duplicates().sort_values().reset_index(drop=True)
    jornadas_unicas = jornadas_unicas.dropna().astype(int)  # Aseguramos que sean enteros
    num_jornadas = len(jornadas_unicas)

    if num_jornadas == 0:
        print("No hay jornadas disponibles para predecir. Abortando.")
        sys.exit(1)

    print("\nJornadas disponibles:")
    for idx, jornada in enumerate(jornadas_unicas, start=1):
        print(f"{idx}. Jornada {jornada}")

    try:
        jornada_selection = int(input(f"\nIngrese el número de la jornada que desea predecir (1-{num_jornadas}): ").strip())
        if jornada_selection < 1 or jornada_selection > num_jornadas:
            raise ValueError
    except ValueError:
        print("La jornada ingresada no es un número válido o está fuera de rango. Abortando.")
        sys.exit(1)

    jornada_num = jornadas_unicas.iloc[jornada_selection - 1]
    df_jornada = df[df['jornada_numero'] == jornada_num].copy()
    if df_jornada.empty:
        print(f"No se encontraron datos para la jornada {jornada_num}. Abortando.")
        sys.exit(1)

    print(f"\nDatos filtrados para la jornada {jornada_num}. Filas: {len(df_jornada)}")

    # =============================
    # 5. Guardar info del partido para imprimir después
    # =============================
    info_partido_cols = [
        'id_partido', 'fecha', 'nombre_equipo_local', 'nombre_equipo_visitante'
    ]
    # Verificamos que esas columnas existan
    for col in info_partido_cols:
        if col not in df_jornada.columns:
            print(f"Error: No se encontró la columna '{col}' en el CSV. Abortando.")
            sys.exit(1)

    df_jornada_info = df_jornada[info_partido_cols].copy()

    # =============================
    # 6. Definir la lista de features que el modelo espera
    #    (las mismas que usaste en el training)
    # =============================
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

    # Verificamos que existan en el DataFrame
    missing_cols = set(features) - set(df_jornada.columns)
    if missing_cols:
        print(f"Error: Faltan columnas en el CSV para poder predecir: {missing_cols}")
        sys.exit(1)

    # Filtramos
    df_jornada_filtrado = df_jornada[features].copy()

    # =============================
    # 7. (Opcional) Rellenar goles_equipo_local/visitante con -1
    #    SOLO si tu modelo incluye esas columnas y las consideraste. 
    #    Si NO las usas, omite este paso.  
    # =============================
    # Si se usan goles en el training, ponlos a -1 para "no conocer resultado".
    # Ejemplo (comentar/ajustar si no los necesitas):
    # df_jornada_filtrado['goles_equipo_local'] = -1
    # df_jornada_filtrado['goles_equipo_visitante'] = -1

    # Convertir todo a numérico para evitar problemas
    for col in df_jornada_filtrado.columns:
        df_jornada_filtrado[col] = pd.to_numeric(df_jornada_filtrado[col], errors='coerce').fillna(0)

    # =============================
    # 8. Predecir con PROBABILIDADES
    # =============================
    try:
        probas = pipeline.predict_proba(df_jornada_filtrado)
        pred_encoded = np.argmax(probas, axis=1)   # Índice de la clase con mayor prob
        pred_labels = clases[pred_encoded]         # Nombre de la clase
    except Exception as e:
        print(f"Error al predecir los partidos: {e}")
        sys.exit(1)

    # =============================
    # 9. Calcular la confianza
    # =============================
    confianzas = np.max(probas, axis=1)

    # Adjuntamos predicciones en df_jornada_info
    df_jornada_info['Predicción'] = pred_labels
    df_jornada_info['Confianza'] = confianzas

    # =============================
    # 10. Ordenar por confianza desc (opcional) y mostrar TOP 4
    # =============================
    df_top4 = df_jornada_info.sort_values('Confianza', ascending=False).head(4)

    # =============================
    # 11. Imprimir resultados
    # =============================
    print("\n=== TOP 4 PARTIDOS CON MAYOR PROBABILIDAD DE ACERTAR ===\n")
    for idx, row in df_top4.iterrows():
        pid = row['id_partido']
        fecha = row['fecha']
        local = row['nombre_equipo_local']
        visitante = row['nombre_equipo_visitante']
        prediccion = row['Predicción']
        confianza = row['Confianza']
        print(f"Partido ID {pid} | Jornada: {fecha} | {local} vs {visitante} => {prediccion} (Confianza: {confianza:.2f})")

    print("\n==============================\n")

if __name__ == "__main__":
    main()
