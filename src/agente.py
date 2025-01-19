#!/usr/bin/env python3
# agente.py
# -*- coding: utf-8 -*-

import requests
import logging
import pandas as pd
import sys
import os
from datetime import datetime
import time

# Aquí importas la función 'procesar_liga' o 'main' (con alias) de info.py,
# asumiendo que la renombraste o usas un alias para evitar colisión:
from info import procesar_liga

from tercerganado import predecir_jornada

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ===============================
# CONFIGURACIÓN DE LA API
# ===============================
API_KEY = "0fd247b049e29f77d89dce2eea2d08f1"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# LISTA ESTÁTICA DE LIGAS QUE INTERESAN (POR SU ID)
LIGAS_SELECCIONADAS_IDS = [
    39,    # Premier League (ejemplo)
    140,   # La Liga (ejemplo)
    135,
    78,
    141,
    88,
    79,
    63
]

# Nombre del modelo (o carpeta en "models")
MODEL_NAME = "Modelo 1.5"

# Ruta final del CSV de predicciones
OUTPUT_PREDICCIONES = os.path.join("outputs", "top_partidos_predicciones.csv")


def obtener_partidos_del_dia():
    """
    Descarga la lista de partidos programados para HOY desde la API,
    retornando un DataFrame con:
      fixture_id, fecha, liga_id, nombre_liga, equipo_local, equipo_visitante, round_info
    """
    logging.info("Obteniendo lista de partidos del día actual...")

    hoy = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/fixtures?date={hoy}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        logging.error(f"Error al obtener partidos: {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json().get("response", [])
    if not data:
        logging.warning("No se encontraron partidos para el día de hoy.")
        return pd.DataFrame()

    registros = []
    for item in data:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})

        round_str = league.get("round", "")

        registros.append({
            "fixture_id": fixture.get("id"),
            "fecha": fixture.get("date"),
            "liga_id": league.get("id"),
            "nombre_liga": league.get("name"),      # Podrías incluso omitirlo, si ya no te interesa
            "equipo_local": teams.get("home", {}).get("name"),
            "equipo_visitante": teams.get("away", {}).get("name"),
            "round_info": round_str
        })

    return pd.DataFrame(registros)


def extraer_numero_jornada(round_str):
    """
    Intenta extraer el número de jornada de un string tipo:
      "Regular Season - 10", "Jornada - 5", etc.
    Retorna None si no se puede convertir a int.
    """
    if not round_str or "-" not in round_str:
        return None
    partes = round_str.split("-")
    if not partes:
        return None

    posible_num = partes[-1].strip()
    try:
        return int(posible_num)
    except ValueError:
        return None


def main():
    # 1. Obtener los partidos del día
    df_partidos_hoy = obtener_partidos_del_dia()
    if df_partidos_hoy.empty:
        logging.info("No hay partidos programados para hoy. Finalizando agente.")
        return

    # 2. Filtrar por las ligas seleccionadas usando la columna 'liga_id'
    df_partidos_filtrados = df_partidos_hoy[df_partidos_hoy['liga_id'].isin(LIGAS_SELECCIONADAS_IDS)].copy()
    if df_partidos_filtrados.empty:
        logging.warning(f"No hay partidos para las ligas con IDs {LIGAS_SELECCIONADAS_IDS}. Finalizando.")
        return

    logging.info(
        f"Se encontraron {len(df_partidos_filtrados)} partidos para las ligas con IDs "
        f"{LIGAS_SELECCIONADAS_IDS}."
    )

    # 3. Extraer los IDs de liga únicos que se juegan hoy
    ligas_ids = df_partidos_filtrados['liga_id'].unique().tolist()

    # 4. Para cada ID de liga, descargamos la temporada 2024 y predecimos la jornada
    df_predicciones_combined = pd.DataFrame()

    for lid in ligas_ids:
        logging.info(f"\n=== Procesando Liga ID: {lid} ===")
        
        csv_file = f"liga"
        # 4a. Descargar/actualizar datos de temporada 2024 (info.py)
        procesar_liga(league_id=lid)  # Ojo, 'procesar_liga' es la alias de info.main

        csv_file = f"liga.csv"
        # 4b. Tomar los partidos filtrados de HOY para esa liga
        df_partidos_liga = df_partidos_filtrados[df_partidos_filtrados['liga_id'] == lid].copy()
        if df_partidos_liga.empty:
            logging.warning(f"No hay partidos de hoy para la liga {lid}. Saltando.")
            continue

        # Extraer la jornada del primer partido
        round_str = df_partidos_liga.iloc[0]['round_info']
        jornada_num = extraer_numero_jornada(round_str)
        if not jornada_num:
            logging.warning(f"No se pudo extraer jornada de '{round_str}' para la liga {lid}. Asignando 1.")
            jornada_num = 1

        # 4c. Construir el nombre del CSV de la temporada (asumimos "liga.csv")
        

        # 4d. Llamar a predecir_jornada
        try:
            df_pred = predecir_jornada(
                model_name=MODEL_NAME,
                csv_file=csv_file,
                jornada_num=jornada_num
            )
        except Exception as e:
            logging.error(f"Error al predecir la jornada {jornada_num} de la liga {lid}: {e}")
            continue

        if df_pred.empty:
            logging.warning(f"No se generaron predicciones para la liga {lid}, jornada {jornada_num}.")
            continue

        # 4e. Concatenar los resultados de esta liga
        df_predicciones_combined = pd.concat([df_predicciones_combined, df_pred], ignore_index=True)
        
        logging.info("Esperando 40 segundos antes de procesar la siguiente liga...")
        time.sleep(40)

    # 5. Si no hubo predicciones en ninguna liga, finalizamos
    if df_predicciones_combined.empty:
        logging.warning("No se generaron predicciones para ninguna liga seleccionada. Finalizando.")
        return

    df_top = df_predicciones_combined.sort_values("Confianza", ascending=False).head(4)

    # 7. Guardar en outputs/top_partidos_predicciones.csv
    os.makedirs("outputs", exist_ok=True)
    df_top.to_csv(OUTPUT_PREDICCIONES, index=False, encoding='utf-8')
    logging.info(f"\nTOP 4 predicciones guardadas en '{OUTPUT_PREDICCIONES}'\n")

    # 8. Mostrar en consola el TOP 4
    logging.info("=== TOP 4 PARTIDOS CON MAYOR PROBABILIDAD DE ACERTAR (todas las ligas) ===\n")
    for idx, row in df_top.iterrows():
        logging.info(
            f"Partido ID {row['id_partido']} | "
            f"Jornada: {row['fecha']} | "
            f"{row['nombre_equipo_local']} vs {row['nombre_equipo_visitante']} => "
            f"{row['Predicción']} (Confianza: {row['Confianza']:.2f})"
        )
    logging.info("===============================")


if __name__ == "__main__":
    main()
