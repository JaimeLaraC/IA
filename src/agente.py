#!/usr/bin/env python3
# agente.py
# -*- coding: utf-8 -*-

import requests
import logging
import pandas as pd
import sys
import os
import pytz
from datetime import datetime, timedelta
import time

# Aquí importas la función 'procesar_liga' o 'main' (con alias) de info.py
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
    78,
    140,
    141,
    61,
    39,
    135,
    88,
    79,
    62,
    63,
    40,
    41,
    42
]

# Nombre del modelo (o carpeta en "models")
MODEL_NAME = "Modelo 1.5"

# Ruta final del CSV de predicciones
OUTPUT_PREDICCIONES = os.path.join("outputs", "top_partidos_predicciones.csv")


def obtener_partidos_dos_dias_utc():
    """
    Descarga la lista de partidos de HOY (UTC) y MAÑANA (UTC) desde la API,
    y los combina en un único DataFrame. Esto se hace para cubrir rangos
    donde en tu zona horaria local aún es 'hoy', pero en UTC ya cambió de día.
    """
    logging.info("Obteniendo lista de partidos del día UTC actual y del día UTC siguiente...")

    # Fecha UTC actual y la siguiente
    hoy_utc = datetime.utcnow().date()
    manana_utc = hoy_utc - timedelta(days=4)

    # Convertir a cadena "YYYY-MM-DD"
    hoy_utc_str = hoy_utc.strftime("%Y-%m-%d")
    manana_utc_str = manana_utc.strftime("%Y-%m-%d")

    # Llamadas a la API
    df_hoy = _descargar_partidos_por_fecha_utc(hoy_utc_str)
    df_manana = _descargar_partidos_por_fecha_utc(manana_utc_str)

    # Combinar
    df = pd.concat([df_hoy, df_manana], ignore_index=True)

    return df


def _descargar_partidos_por_fecha_utc(fecha_utc_str):
    """
    Función auxiliar que descarga los partidos según la fecha en formato UTC (YYYY-MM-DD).
    Devuelve un DataFrame con las columnas:
    [fixture_id, fecha (string), liga_id, nombre_liga, equipo_local, equipo_visitante, round_info].
    """
    url = f"{BASE_URL}/fixtures?date={fecha_utc_str}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        logging.error(f"Error al obtener partidos: {response.status_code} - {response.text}")
        return pd.DataFrame()

    data = response.json().get("response", [])
    if not data:
        logging.warning(f"No se encontraron partidos para la fecha (UTC) {fecha_utc_str}.")
        return pd.DataFrame()

    registros = []
    for item in data:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})

        round_str = league.get("round", "")

        registros.append({
            "fixture_id": fixture.get("id"),
            "fecha": fixture.get("date"),  # string con fecha/hora en UTC
            "liga_id": league.get("id"),
            "nombre_liga": league.get("name"),
            "equipo_local": teams.get("home", {}).get("name"),
            "equipo_visitante": teams.get("away", {}).get("name"),
            "round_info": round_str
        })

    return pd.DataFrame(registros)


def obtener_partidos_del_dia_local():
    """
    Obtiene los partidos que, en la hora local, corresponden a 'mañana'.
    
    - Descarga partidos del día UTC actual y del siguiente (para cubrir
      desfases horarios).
    - Convierte la columna 'fecha' (UTC) a tu zona horaria local.
    - Filtra para quedarse únicamente con los que caen en la fecha local de 'mañana'.
    """
    tz_local = pytz.timezone("America/Mexico_City")
    # Obtenemos la fecha local de hoy
    hoy_local = datetime.now(tz_local).date()
    # Calculamos 'mañana' en hora local
    manana_local = hoy_local - timedelta(days=4)

    # 1. Descargar partidos de hoy y mañana en UTC
    df = obtener_partidos_dos_dias_utc()
    if df.empty:
        return df

    # 2. Convertir la columna 'fecha' a datetime UTC
    df["fecha_utc"] = pd.to_datetime(df["fecha"], utc=True)

    # 3. Convertir a hora local
    df["fecha_local"] = df["fecha_utc"].dt.tz_convert(tz_local)

    # 4. Filtrar por la parte date local == manana_local
    df = df[df["fecha_local"].dt.date == manana_local]

    # 5. Actualizar la columna 'fecha' con la fecha local (opcional, por estética)
    df["fecha"] = df["fecha_local"]
    df.drop(columns=["fecha_utc", "fecha_local"], inplace=True)

    return df


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
    # 1. Obtener los partidos "de mañana" según la hora local
    df_partidos_manana = obtener_partidos_del_dia_local()
    if df_partidos_manana.empty:
        logging.info("No hay partidos programados (en hora local) para mañana. Finalizando agente.")
        return

    # 2. Filtrar por las ligas seleccionadas
    df_partidos_filtrados = df_partidos_manana[df_partidos_manana['liga_id'].isin(LIGAS_SELECCIONADAS_IDS)].copy()
    if df_partidos_filtrados.empty:
        logging.warning(f"No hay partidos (en hora local) para ligas con IDs {LIGAS_SELECCIONADAS_IDS}. Finalizando.")
        return

    logging.info(
        f"Se encontraron {len(df_partidos_filtrados)} partidos para las ligas con IDs "
        f"{LIGAS_SELECCIONADAS_IDS} que se juegan MAÑANA (en hora local)."
    )

    # 3. Extraer los IDs de liga únicos que se juegan mañana
    ligas_ids = df_partidos_filtrados['liga_id'].unique().tolist()

    # 4. Para cada ID de liga, procesar datos y predecir
    df_predicciones_combined = pd.DataFrame()

    for lid in ligas_ids:
        logging.info(f"\n=== Procesando Liga ID: {lid} ===")
        procesar_liga(league_id=lid)
        # -------------------------------
        # 1. Obtener SOLO los partidos de MAÑANA de esa liga
        # -------------------------------
        df_partidos_liga = df_partidos_filtrados[df_partidos_filtrados['liga_id'] == lid].copy()
        if df_partidos_liga.empty:
            continue

        round_str = df_partidos_liga.iloc[0]['round_info']
        jornada_num = extraer_numero_jornada(round_str)
        if not jornada_num:
            logging.warning(f"No se pudo extraer jornada de '{round_str}' para la liga {lid}. Asignando 1.")
            jornada_num = 1

        # -------------------------------
        # 2. Filtrar liga.csv por esos fixture_id de MAÑANA
        # -------------------------------
        csv_file_path = os.path.join("data", "liga.csv")  # Ajusta si tu CSV se llama distinto

        # Lee el CSV completo de la temporada (todas las jornadas)
        df_liga_completa = pd.read_csv(csv_file_path)

        # Qué fixture_ids están en los partidos de MAÑANA para esta liga
        manana_fixture_ids = df_partidos_liga["fixture_id"].unique()

        # Filtra la DataFrame de la liga completa para quedarte sólo con esos partidos
        df_liga_filtrada = df_liga_completa[df_liga_completa["id_partido"].isin(manana_fixture_ids)]

        if df_liga_filtrada.empty:
            logging.warning(f"Tras filtrar, no hay partidos para mañana en la liga {lid}. Saltando.")
            continue

        # Guardamos en un CSV temporal (ejemplo: sobreescribiendo "liga.csv" o como gustes)
        liga_filtrada_csv = os.path.join("data", f"liga.csv")
        df_liga_filtrada.to_csv(liga_filtrada_csv, index=False)

        # -------------------------------
        # 3. Llamar a la predicción usando el CSV filtrado
        # -------------------------------
        try:
            df_pred = predecir_jornada(
                model_name=MODEL_NAME,
                csv_file=f"liga.csv",  # Usamos el CSV filtrado
                jornada_num=jornada_num
            )
        except Exception as e:
            logging.error(f"Error al predecir la jornada {jornada_num} de la liga {lid}: {e}")
            continue

        if df_pred.empty:
            logging.warning(f"No se generaron predicciones para la liga {lid}, jornada {jornada_num}.")
            continue

        # -------------------------------
        # 4. Concatenar las predicciones
        # -------------------------------
        df_predicciones_combined = pd.concat([df_predicciones_combined, df_pred], ignore_index=True)

        logging.info("Esperando 40 segundos antes de procesar la siguiente liga...")
        time.sleep(40)


    if df_predicciones_combined.empty:
        logging.warning("No se generaron predicciones para ninguna liga seleccionada. Finalizando.")
        return

    # Obtener TOP 6 por "Confianza" (o el número que desees)
    df_top = df_predicciones_combined.sort_values("Confianza", ascending=False).head(6)

    # Guardar
    os.makedirs("outputs", exist_ok=True)
    df_top.to_csv(OUTPUT_PREDICCIONES, index=False, encoding='utf-8')
    logging.info(f"\nTOP 6 predicciones guardadas en '{OUTPUT_PREDICCIONES}'\n")

    # Mostrar en consola
    logging.info("=== TOP 6 PARTIDOS CON MAYOR PROBABILIDAD DE ACIERTO (todas las ligas) ===\n")
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
