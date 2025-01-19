#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import os
import time
import pandas as pd
from datetime import datetime

# ================================================
# CONFIGURACIÓN DE LA API Y PARÁMETROS GENERALES
# ================================================
API_KEY = "0fd247b049e29f77d89dce2eea2d08f1"  # Tu clave de API
BASE_URL = "https://v3.football.api-sports.io"
SEASONS = ["2024"]  # Temporadas a consultar (siempre la 2024)
REQUEST_LIMIT = 300
RESET_INTERVAL = 60  # en segundos

# Carpetas de salida
OUTPUT_JSON_DIR = "outputs"
OUTPUT_CSV_DIR = "data"

# Nombre base (sin extensión) de los archivos
BASE_FILENAME = "liga"

# Asegurarnos de que las carpetas existen
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# Rutas de salida para JSON y CSV
JSON_PATH = os.path.join(OUTPUT_JSON_DIR, f"{BASE_FILENAME}.json")
CSV_PATH = os.path.join(OUTPUT_CSV_DIR, f"{BASE_FILENAME}.csv")

# Cabeceras para la API
headers = {"x-apisports-key": API_KEY}

# Contador de solicitudes
REQUEST_COUNT = 0


# ================================================
# FUNCIONES AUXILIARES
# ================================================
def check_request_limit():
    """
    Pausa si se alcanzó el límite de solicitudes a la API.
    """
    global REQUEST_COUNT
    if REQUEST_COUNT >= REQUEST_LIMIT:
        print(f"Límite de {REQUEST_LIMIT} solicitudes alcanzado. Pausando {RESET_INTERVAL} seg...")
        time.sleep(RESET_INTERVAL)
        REQUEST_COUNT = 0

def parse_fixture_date(date_str):
    """
    Convierte la cadena de fecha (con posible '+00:00') a datetime.
    Ajusta según el formato que devuelva la API.
    """
    if "+" in date_str:
        date_str = date_str.split("+")[0]
    return datetime.fromisoformat(date_str)

def load_existing_results(filepath):
    """
    Carga datos existentes si el archivo ya existe, para irlos actualizando.
    Retorna un dict. Si no existe, retorna dict vacío.
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    return {}

def get_winner_id(h2h_match):
    """
    Dado un 'fixture' de head-to-head de la API,
    determina el ID del equipo ganador, o None si hay empate.
    """
    teams = h2h_match.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    if home.get("winner") is True:
        return home.get("id")
    elif away.get("winner") is True:
        return away.get("id")
    return None

# ================================================
# FUNCIÓN PRINCIPAL DE CÁLCULO DE FEATURES
# ================================================
def calculate_team_features(team_id, fixture_datetime, previous_fixtures, head_to_head):
    """
    Calcula estadísticas para un 'team_id' dado, considerando
    partidos previos anteriores a 'fixture_datetime'.

    Retorna un dict con múltiples características (puntos, rachas, etc.).
    """
    # Filtramos sólo los partidos previos de este equipo
    matches_before = []
    for match in previous_fixtures:
        match_date = parse_fixture_date(match["date"])
        if match_date < fixture_datetime:
            if (match["home_team_id"] == team_id) or (match["away_team_id"] == team_id):
                matches_before.append(match)

    total_points = 0
    last10_results = []
    goals_scored_list = []
    goals_conceded_list = []
    last5_matches = []

    # Rachas
    current_win_streak = 0
    current_loss_streak = 0

    # Contadores casa/visitante
    home_matches_played = 0
    away_matches_played = 0
    home_wins = 0
    away_wins = 0

    # Para clean sheets y estadísticas de goles en últimos 10
    clean_sheets_last10 = 0
    btts_last10 = 0
    over25_last10 = 0

    # Recorrer partidos previos (del más reciente al más antiguo)
    for match in reversed(matches_before):
        goals_home = match.get("goals_home", 0) or 0
        goals_away = match.get("goals_away", 0) or 0

        if team_id == match["home_team_id"]:
            gf = goals_home
            gc = goals_away
            home_matches_played += 1
            if gf > gc:
                home_wins += 1
        else:
            gf = goals_away
            gc = goals_home
            away_matches_played += 1
            if gf > gc:
                away_wins += 1

        # Calcular puntos
        if gf > gc:
            total_points += 3
            result_str = "win"
        elif gf == gc:
            total_points += 1
            result_str = "draw"
        else:
            result_str = "lose"

        last10_results.append(result_str)
        goals_scored_list.append(gf)
        goals_conceded_list.append(gc)

        # Rachas (solo para el partido más reciente o consecutivos)
        if len(last10_results) == 1:
            if result_str == "win":
                current_win_streak = 1
                current_loss_streak = 0
            elif result_str == "lose":
                current_loss_streak = 1
                current_win_streak = 0
            else:
                current_win_streak = 0
                current_loss_streak = 0
        else:
            if result_str == "win":
                if current_win_streak > 0:
                    current_win_streak += 1
                else:
                    current_win_streak = 1
                current_loss_streak = 0
            elif result_str == "lose":
                if current_loss_streak > 0:
                    current_loss_streak += 1
                else:
                    current_loss_streak = 1
                current_win_streak = 0
            else:  # draw
                current_win_streak = 0
                current_loss_streak = 0

        # Últimos 5
        last5_matches.append((gf, gc, result_str))
        if len(last5_matches) > 5:
            last5_matches.pop(0)

        # Estadísticas sobre los últimos 10
        if gc == 0:
            clean_sheets_last10 += 1
        if (gf > 0) and (gc > 0):
            btts_last10 += 1
        if (gf + gc) >= 3:
            over25_last10 += 1

        # Romper si ya tenemos 10 partidos para las últimas stats
        if len(last10_results) >= 10:
            break

    # Resumen últimos 10
    wins_last10 = last10_results.count("win")
    draws_last10 = last10_results.count("draw")
    losses_last10 = last10_results.count("lose")
    goles_favor_last10 = sum(goals_scored_list)
    goles_contra_last10 = sum(goals_conceded_list)

    # Métricas últimos 5
    points_last5 = 0
    gf_last5 = 0
    gc_last5 = 0
    clean_sheets_last5 = 0
    btts_last5 = 0
    over25_last5 = 0

    for (gf_m, gc_m, result) in last5_matches:
        gf_last5 += gf_m
        gc_last5 += gc_m
        if result == "win":
            points_last5 += 3
        elif result == "draw":
            points_last5 += 1

        if gc_m == 0:
            clean_sheets_last5 += 1
        if gf_m > 0 and gc_m > 0:
            btts_last5 += 1
        if gf_m + gc_m >= 3:
            over25_last5 += 1

    # Promedios últimos 5
    count_matches_5 = len(last5_matches) or 1
    avg_goals_scored_last5 = gf_last5 / count_matches_5
    avg_goals_conceded_last5 = gc_last5 / count_matches_5

    # Head2Head
    h2h_wins = 0
    h2h_draws = 0
    h2h_losses = 0
    for h2h in head_to_head:
        winner = get_winner_id(h2h)
        if winner == team_id:
            h2h_wins += 1
        elif winner is None:
            h2h_draws += 1
        else:
            h2h_losses += 1

    # Días de descanso
    last_fixture_date = None
    for pf in reversed(previous_fixtures):
        d_pf = parse_fixture_date(pf["date"])
        if d_pf < fixture_datetime and (pf["home_team_id"] == team_id or pf["away_team_id"] == team_id):
            last_fixture_date = d_pf
            break

    if last_fixture_date is not None:
        days_rest = (fixture_datetime - last_fixture_date).days
    else:
        days_rest = 0

    # Variables "simuladas"
    position_table = team_id % 20 + 1
    elo_rating = 1500 + (team_id % 100)
    market_value = 50_000_000 + (team_id % 10) * 1_000_000
    injured_players = team_id % 3
    suspended_starters = team_id % 2

    # Rendimiento local / visitante
    home_win_rate = round((home_wins / home_matches_played), 2) if home_matches_played > 0 else 0
    away_win_rate = round((away_wins / away_matches_played), 2) if away_matches_played > 0 else 0

    return {
        "puntos_acumulados": total_points,
        "wins_last10": wins_last10,
        "draws_last10": draws_last10,
        "losses_last10": losses_last10,
        "goles_favor_last10": goles_favor_last10,
        "goles_contra_last10": goles_contra_last10,
        "head2head_wins": h2h_wins,
        "head2head_draws": h2h_draws,
        "head2head_losses": h2h_losses,
        "puntos_balance": total_points,
        "goles_balance": goles_favor_last10 - goles_contra_last10,
        "posicion_tabla": position_table,
        "dias_descanso": days_rest,
        "elo_rating": elo_rating,
        "valor_mercado": market_value,
        "jugadores_lesionados": injured_players,
        "titulares_sancionados": suspended_starters,

        # Rachas
        "current_win_streak": current_win_streak,
        "current_loss_streak": current_loss_streak,

        # Últimos 5
        "puntos_ultimos5": points_last5,
        "goles_favor_ultimos5": gf_last5,
        "goles_contra_ultimos5": gc_last5,
        "prom_goles_favor_ultimos5": round(avg_goals_scored_last5, 2),
        "prom_goles_contra_ultimos5": round(avg_goals_conceded_last5, 2),

        # Estadísticas últimos 10
        "clean_sheets_last10": clean_sheets_last10,
        "btts_last10": btts_last10,
        "over25_last10": over25_last10,

        # Estadísticas últimos 5
        "clean_sheets_last5": clean_sheets_last5,
        "btts_last5": btts_last5,
        "over25_last5": over25_last5,

        # Rendimiento local/visitante
        "home_win_rate": home_win_rate,
        "away_win_rate": away_win_rate
    }


# ================================================
# 1. DESCARGAR PARTIDOS DE LA API Y GUARDAR EN JSON
# ================================================
def fetch_and_save_json(league_id):
    """
    Descarga y procesa la información de la liga (league_id) para
    las temporadas en SEASONS, y guarda el resultado en JSON.
    """
    global REQUEST_COUNT

    # Cargar JSON existente (si lo hubiera) para no sobrescribirlo completamente
    all_results = load_existing_results(JSON_PATH)

    try:
        for SEASON in SEASONS:
            hoy_str = datetime.now().strftime("%Y-%m-%d")
            fecha_inicio = "2024-01-01"  # Desde enero de 2024
            fecha_fin = hoy_str         # Hasta hoy

            url = (
                f"{BASE_URL}/fixtures?"
                f"league={league_id}&"
                f"season={SEASON}&"
                f"from={fecha_inicio}&"
                f"to={fecha_fin}"
            )
            response = requests.get(url, headers=headers)
            REQUEST_COUNT += 1
            check_request_limit()

            if response.status_code == 200:
                data = response.json()
                fixtures = data.get("response", [])

                if not fixtures:
                    print(f"No se encontraron partidos para la temporada {SEASON} (Liga {league_id}).")
                    continue

                # Ordenar fixtures por fecha
                fixtures.sort(key=lambda x: parse_fixture_date(x["fixture"]["date"]))

                # Asegurarnos de tener la clave SEASON en all_results
                all_results[SEASON] = []

                previous_fixtures = []

                for match in fixtures:
                    fixture_info = match.get("fixture", {})
                    teams_info = match.get("teams", {})
                    goals_info = match.get("goals", {})

                    fixture_date_str = fixture_info.get("date", "")
                    fixture_datetime = parse_fixture_date(fixture_date_str)  # Devuelve un datetime

                    if fixture_datetime.date() > datetime.now().date():
                        # Este partido es del futuro (mañana o después), lo omitimos
                        continue

                    fixture_date_str = fixture_info.get("date", "")
                    fixture_datetime = parse_fixture_date(fixture_date_str)

                    # Manejo de goles (None -> 0)
                    goals_home = goals_info.get("home", 0) if goals_info.get("home") is not None else 0
                    goals_away = goals_info.get("away", 0) if goals_info.get("away") is not None else 0

                    home_team_id = teams_info.get("home", {}).get("id")
                    away_team_id = teams_info.get("away", {}).get("id")

                    # Número de jornada (si aplica)
                    round_info = match.get("league", {}).get("round", "Jornada desconocida")
                    if " - " in round_info:
                        jornada_numero = round_info.split(" - ")[-1]
                    else:
                        jornada_numero = round_info  # fallback

                    # Head2Head
                    if home_team_id and away_team_id:
                        h2h_url = f"{BASE_URL}/fixtures/headtohead?h2h={home_team_id}-{away_team_id}"
                        h2h_response = requests.get(h2h_url, headers=headers)
                        REQUEST_COUNT += 1
                        check_request_limit()

                        if h2h_response.status_code == 200:
                            head_to_head_data = h2h_response.json().get("response", [])
                        else:
                            head_to_head_data = []
                    else:
                        head_to_head_data = []

                    # Calcular características del equipo local
                    if home_team_id:
                        home_features = calculate_team_features(
                            home_team_id, fixture_datetime, previous_fixtures, head_to_head_data
                        )
                    else:
                        home_features = {}

                    # Calcular características del equipo visitante
                    if away_team_id:
                        away_features = calculate_team_features(
                            away_team_id, fixture_datetime, previous_fixtures, head_to_head_data
                        )
                    else:
                        away_features = {}

                    # Agregar partido a 'previous_fixtures' solo si YA se jugó
                    if fixture_datetime < datetime.now():
                        previous_fixtures.append({
                            "date": fixture_date_str,
                            "home_team_id": home_team_id,
                            "away_team_id": away_team_id,
                            "goals_home": goals_home,
                            "goals_away": goals_away
                        })

                    # Guardar la info en la estructura final
                    all_results[SEASON].append({
                        "id_partido": fixture_info.get("id"),
                        "fecha": jornada_numero,
                        "fecha_partido": fixture_date_str,
                        "nombre_equipo_local": teams_info.get("home", {}).get("name"),
                        "nombre_equipo_visitante": teams_info.get("away", {}).get("name"),
                        "goles_equipo_local": goals_home,
                        "goles_equipo_visitante": goals_away,

                        # Features equipo local
                        "puntos_acumulados_local": home_features.get("puntos_acumulados", 0),
                        "wins_last10_local": home_features.get("wins_last10", 0),
                        "draws_last10_local": home_features.get("draws_last10", 0),
                        "losses_last10_local": home_features.get("losses_last10", 0),
                        "goles_favor_last10_local": home_features.get("goles_favor_last10", 0),
                        "goles_contra_last10_local": home_features.get("goles_contra_last10", 0),
                        "current_win_streak_local": home_features.get("current_win_streak", 0),
                        "current_loss_streak_local": home_features.get("current_loss_streak", 0),
                        "puntos_ultimos5_local": home_features.get("puntos_ultimos5", 0),
                        "goles_favor_ultimos5_local": home_features.get("goles_favor_ultimos5", 0),
                        "goles_contra_ultimos5_local": home_features.get("goles_contra_ultimos5", 0),
                        "prom_goles_favor_ultimos5_local": home_features.get("prom_goles_favor_ultimos5", 0),
                        "prom_goles_contra_ultimos5_local": home_features.get("prom_goles_contra_ultimos5", 0),
                        "clean_sheets_last10_local": home_features.get("clean_sheets_last10", 0),
                        "btts_last10_local": home_features.get("btts_last10", 0),
                        "over25_last10_local": home_features.get("over25_last10", 0),
                        "clean_sheets_last5_local": home_features.get("clean_sheets_last5", 0),
                        "btts_last5_local": home_features.get("btts_last5", 0),
                        "over25_last5_local": home_features.get("over25_last5", 0),
                        "home_win_rate_local": home_features.get("home_win_rate", 0),
                        "away_win_rate_local": home_features.get("away_win_rate", 0),

                        # Features equipo visitante
                        "puntos_acumulados_visitante": away_features.get("puntos_acumulados", 0),
                        "wins_last10_visitante": away_features.get("wins_last10", 0),
                        "draws_last10_visitante": away_features.get("draws_last10", 0),
                        "losses_last10_visitante": away_features.get("losses_last10", 0),
                        "goles_favor_last10_visitante": away_features.get("goles_favor_last10", 0),
                        "goles_contra_last10_visitante": away_features.get("goles_contra_last10", 0),
                        "current_win_streak_visitante": away_features.get("current_win_streak", 0),
                        "current_loss_streak_visitante": away_features.get("current_loss_streak", 0),
                        "puntos_ultimos5_visitante": away_features.get("puntos_ultimos5", 0),
                        "goles_favor_ultimos5_visitante": away_features.get("goles_favor_ultimos5", 0),
                        "goles_contra_ultimos5_visitante": away_features.get("goles_contra_ultimos5", 0),
                        "prom_goles_favor_ultimos5_visitante": away_features.get("prom_goles_favor_ultimos5", 0),
                        "prom_goles_contra_ultimos5_visitante": away_features.get("prom_goles_contra_ultimos5", 0),
                        "clean_sheets_last10_visitante": away_features.get("clean_sheets_last10", 0),
                        "btts_last10_visitante": away_features.get("btts_last10", 0),
                        "over25_last10_visitante": away_features.get("over25_last10", 0),
                        "clean_sheets_last5_visitante": away_features.get("clean_sheets_last5", 0),
                        "btts_last5_visitante": away_features.get("btts_last5", 0),
                        "over25_last5_visitante": away_features.get("over25_last5", 0),
                        "home_win_rate_visitante": away_features.get("home_win_rate", 0),
                        "away_win_rate_visitante": away_features.get("away_win_rate", 0),

                        # Head2head
                        "head2head_local_wins": home_features.get("head2head_wins", 0),
                        "head2head_local_draws": home_features.get("head2head_draws", 0),
                        "head2head_local_losses": home_features.get("head2head_losses", 0),
                        "head2head_visitante_wins": away_features.get("head2head_wins", 0),
                        "head2head_visitante_draws": away_features.get("head2head_draws", 0),
                        "head2head_visitante_losses": away_features.get("head2head_losses", 0),

                        # Balances
                        "puntos_balance": home_features.get("puntos_balance", 0) - away_features.get("puntos_balance", 0),
                        "goles_balance": home_features.get("goles_balance", 0) - away_features.get("goles_balance", 0),

                        # Posición en tabla simulada
                        "posicion_tabla_local": home_features.get("posicion_tabla", 0),
                        "posicion_tabla_visitante": away_features.get("posicion_tabla", 0),
                        "diff_posicion_tabla": home_features.get("posicion_tabla", 0) - away_features.get("posicion_tabla", 0),

                        # Días de descanso
                        "dias_descanso_local": home_features.get("dias_descanso", 0),
                        "dias_descanso_visitante": away_features.get("dias_descanso", 0),
                        "diff_dias_descanso": home_features.get("dias_descanso", 0) - away_features.get("dias_descanso", 0),

                        # ELO (simulado)
                        "elo_rating_local": home_features.get("elo_rating", 0),
                        "elo_rating_visitante": away_features.get("elo_rating", 0),
                        "diff_elo_rating": home_features.get("elo_rating", 0) - away_features.get("elo_rating", 0),

                        # Otros simulados
                        "valor_mercado_local": home_features.get("valor_mercado", 0),
                        "valor_mercado_visitante": away_features.get("valor_mercado", 0),
                        "jugadores_lesionados_local": home_features.get("jugadores_lesionados", 0),
                        "jugadores_lesionados_visitante": away_features.get("jugadores_lesionados", 0),
                        "titulares_sancionados_local": home_features.get("titulares_sancionados", 0),
                        "titulares_sancionados_visitante": away_features.get("titulares_sancionados", 0),
                    })

                print(f"Resultados de la temporada {SEASON} (Liga {league_id}) obtenidos exitosamente.")
            else:
                print(f"Error al obtener la temporada {SEASON} (Liga {league_id}): {response.status_code} - {response.text}")

    except KeyboardInterrupt:
        print("Ejecución interrumpida por el usuario. Guardando datos parciales...")

    finally:
        # Guardar todos los datos en el archivo JSON
        with open(JSON_PATH, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=4, ensure_ascii=False)

        print(f"Datos guardados en '{JSON_PATH}'")


# ================================================
# 2. CONVERTIR EL JSON A CSV
# ================================================
def convert_json_to_csv():
    """
    Lee el archivo JSON que se generó, filtra (solo temporada 2024)
    y guarda un CSV con los campos disponibles en 'data/prueba.csv'.
    """

    def formatear_fecha(fecha_str):
        """
        Convierte 'YYYY-MM-DDTHH:MM:SS' en 'DD/MM/YYYY'.
        Si no es válido, lo deja tal cual.
        """
        try:
            fecha_solo = fecha_str.split("T")[0]  # Tomar solo 'YYYY-MM-DD'
            return datetime.strptime(fecha_solo, "%Y-%m-%d").strftime("%d/%m/%Y")
        except (ValueError, TypeError):
            return fecha_str

    # Verificar que existe el JSON
    if not os.path.exists(JSON_PATH):
        print(f"No existe el archivo JSON en {JSON_PATH}.")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("El JSON no tiene la estructura esperada (dict con claves de temporada).")
        return

    # Siempre trabajamos únicamente con la temporada 2024
    registros = []
    for temporada, partidos in data.items():
        if temporada != "2024":
            print(f"> Omitiendo datos de la temporada {temporada} (solo se procesará 2024).")
            continue

        for partido in partidos:
            partido['temporada'] = temporada
            # Ajustamos la fecha para el CSV
            partido['fecha_partido'] = formatear_fecha(partido.get("fecha_partido", ""))
            registros.append(partido)

    if not registros:
        print("No se encontraron datos para la temporada 2024 en el JSON.")
        return

    df = pd.DataFrame(registros)

    # Reordenar columnas (opcional)
    columnas = list(df.columns)
    if 'temporada' in columnas:
        # Mover 'temporada' al inicio
        columnas.insert(0, columnas.pop(columnas.index('temporada')))
    df = df.reindex(columns=columnas)

    # Guardar en CSV
    df.to_csv(CSV_PATH, index=False, encoding='utf-8')
    print(f"\nCSV generado correctamente en: {CSV_PATH}")


# ================================================
# 3. MAIN: Ejecutar todo el flujo con league_id
# ================================================
def procesar_liga(league_id):
    """
    Llamar a esta función desde otro script para procesar la liga 'league_id'.
    1) Descarga los datos de la temporada (fetch_and_save_json).
    2) Genera el CSV (convert_json_to_csv).
    """
    # 1. Llamar a la API y guardar en JSON
    fetch_and_save_json(league_id=league_id)

    # 2. Convertir ese JSON a CSV (solo temporada 2024)
    convert_json_to_csv()
