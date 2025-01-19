#!/usr/bin/env python3
# ligaID.py
# -*- coding: utf-8 -*-

import requests
import sys
import logging
import pandas as pd
import os

# Configura aquí tu clave de API y la URL base
API_KEY = "0fd247b049e29f77d89dce2eea2d08f1"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def obtener_ligas_disponibles():
    """
    Llama a la API y obtiene un listado de ligas (nombre e ID).
    Devuelve un DataFrame con columnas: [league_id, league_name]
    """
    logging.info("Obteniendo todas las ligas disponibles desde la API...")
    url = f"{BASE_URL}/leagues"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        logging.error(f"Error al listar ligas: {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json().get("response", [])
    if not data:
        logging.warning("No se encontraron ligas en la respuesta de la API.")
        sys.exit(1)

    registros = []
    for item in data:
        league_data = item.get("league", {})
        league_id = league_data.get("id")
        league_name = league_data.get("name")

        registros.append({
            "league_id": league_id,
            "league_name": league_name
        })

    df_ligas = pd.DataFrame(registros).drop_duplicates("league_id").reset_index(drop=True)
    return df_ligas

def get_league_id_by_name(league_name):
    """
    Dado el nombre de una liga (cadena),
    busca en la lista de ligas disponibles y
    retorna el 'league_id' correspondiente.

    Si no la encuentra, retorna None.
    """
    df_ligas = obtener_ligas_disponibles()

    # Filtrar por nombre exacto.
    # Ojo: el nombre debe coincidir tal cual lo devuelva la API.
    df_filtrado = df_ligas[df_ligas['league_name'].str.lower() == league_name.lower()]

    if df_filtrado.empty:
        return None

    # Tomamos el primer resultado, en caso de que hubiera duplicados
    return int(df_filtrado.iloc[0]['league_id'])

def main():
    """
    Uso:
      python ligaID.py "Nombre de la Liga"

    Ejemplo:
      python ligaID.py "Premier League"
    """

    if len(sys.argv) < 2:
        print(f"Uso: python {os.path.basename(__file__)} \"Nombre de la Liga\"")
        sys.exit(1)

    league_name = " ".join(sys.argv[1:])  # Tomamos todo lo que venga como nombre
    logging.info(f"Buscando la ID para la liga con nombre: '{league_name}'")

    league_id = get_league_id_by_name(league_name)
    if league_id is None:
        print(f"No se encontró ninguna liga con el nombre '{league_name}'.")
        sys.exit(1)

    print(f"La liga '{league_name}' tiene ID: {league_id}")

if __name__ == "__main__":
    main()
