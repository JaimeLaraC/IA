#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# -----------------------------------------------------
#               CONFIGURACIÓN INICIAL
# -----------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Se recomienda manejar el token vía variable de entorno.
TOKEN = os.getenv("TELEGRAM_TOKEN", "7402710972:AAF6o-l5mHLdrCmrtbtyL0-ZerYzxKuBQ6g")
if not TOKEN:
    logger.error("El token de Telegram (TELEGRAM_TOKEN) no está configurado.")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ajusta estas rutas a tu estructura real
MODELS_DIR = os.path.join(BASE_DIR, "../models")
CSV_DIR = os.path.join(BASE_DIR, "../data")

# Rango de jornadas (1..38). Ajústalo si tu liga tiene otra cantidad.
JORNADAS = list(range(1, 39))

# Para no bloquear el bot al hacer la predicción
executor = ThreadPoolExecutor(max_workers=5)

# -----------------------------------------------------
#   FUNCIONES AUXILIARES PARA OBTENER MODELOS Y CSV
# -----------------------------------------------------

def obtener_modelos():
    """
    Retorna una lista con los nombres base (sin extensión) de archivos .pkl
    en la carpeta MODELS_DIR, ordenados alfabéticamente.
    """
    modelos = []
    try:
        for root, _, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith(".pkl"):
                    modelos.append(os.path.splitext(file)[0])
        modelos.sort()
    except FileNotFoundError:
        logger.error(f"No se encontró el directorio de modelos: {MODELS_DIR}")
    return modelos

def obtener_csv():
    """
    Retorna un diccionario {nombre_simplificado: ruta_absoluta} para
    todos los archivos .csv en CSV_DIR, ordenado por clave.
    """
    csv_dict = {}
    try:
        archivos = os.listdir(CSV_DIR)
        for archivo in archivos:
            if archivo.endswith(".csv"):
                # Ej: si el archivo es "2024LaLiga.csv", extrae "LaLiga"
                match = re.match(r"2024(.+)\.csv", archivo)
                nombre = match.group(1) if match else archivo.replace(".csv", "")
                csv_dict[nombre] = os.path.join(CSV_DIR, archivo)
        csv_dict = dict(sorted(csv_dict.items()))
    except FileNotFoundError:
        logger.error(f"No se encontró el directorio de CSVs: {CSV_DIR}")
    return csv_dict

# -----------------------------------------------------
#   FUNCIÓN DE NORMALIZACIÓN DE PREDICCIÓN
# -----------------------------------------------------

def normalizar_prediccion(pred):
    """
    Convierte etiquetas del modelo como 'Local Gana', 'Visitante Gana', etc.
    en 'Local', 'Visitante' o 'Empate'.
    """
    pred_lower = pred.lower()
    if "local" in pred_lower:
        return "Local"
    elif "visitante" in pred_lower:
        return "Visitante"
    return "Empate"

# -----------------------------------------------------
#   FUNCIÓN AUXILIAR: CARGAR CSV + FILTRAR JORNADA
# -----------------------------------------------------

def cargar_y_filtrar_csv(csv_path: str, jornada: int) -> pd.DataFrame or str:
    """
    Lee el CSV, normaliza columnas, filtra la jornada solicitada
    y retorna el DataFrame filtrado. Si hay error, retorna string de error.
    """
    # Verificar existencia de archivo
    if not os.path.exists(csv_path):
        return f"❌ **Error**: No se encontró el archivo CSV: {csv_path}"

    # Leer CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"❌ **Error** al leer CSV '{csv_path}': {e}"

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if "fecha" not in df.columns:
        return "❌ **Error**: No existe la columna 'fecha' en el CSV."

    # Convertir 'fecha' a 'jornada_numero'
    df["jornada_numero"] = pd.to_numeric(df["fecha"], errors="coerce").astype("Int64")
    df_jornada = df[df["jornada_numero"] == jornada].copy()
    if df_jornada.empty:
        return f"❌ **Error**: No se encontraron datos para la jornada {jornada}."
    return df_jornada

# -----------------------------------------------------
#   PREDICCIÓN DE TOP 4 POR JORNADA
# -----------------------------------------------------

def predecir_top4(modelo: str, csv_path: str, jornada: int) -> str:
    """
    1. Carga el modelo y sus clases.
    2. Filtra la jornada completa del CSV.
    3. Predice TODOS los partidos, construye un Top 4 (ordenado por confianza).
    4. Devuelve un string con esos 4 partidos y su predicción.
    """
    # Paths de modelo y clases
    model_folder = os.path.join(MODELS_DIR, modelo)
    model_path = os.path.join(model_folder, f"{modelo}.pkl")
    classes_path = os.path.join(model_folder, "classes.npy")

    # Verificar archivos
    if not os.path.exists(model_path):
        return f"❌ **Error**: No se encontró el archivo del modelo: {model_path}"
    if not os.path.exists(classes_path):
        return f"❌ **Error**: No se encontró el archivo de clases: {classes_path}"

    # Cargar CSV y filtrar
    df_jornada = cargar_y_filtrar_csv(csv_path, jornada)
    if isinstance(df_jornada, str):  # si hay error, df_jornada será un string
        return df_jornada

    # Cargar modelo y clases
    try:
        pipeline = joblib.load(model_path)
        clases = np.load(classes_path, allow_pickle=True)
    except Exception as e:
        return f"❌ **Error** al cargar modelo/clases: {e}"

    # Lista de features (ajusta con tus columnas reales)
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

    # Agregar columnas faltantes con 0
    missing = set(features) - set(df_jornada.columns)
    if missing:
        for col in missing:
            df_jornada[col] = 0
        logger.warning(f"Se agregaron columnas faltantes con 0: {', '.join(missing)}")

    # Crear sub-DataFrame de features y convertir a numérico
    df_features = df_jornada[features].copy()
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce").fillna(0)

    # Predicciones
    try:
        probas = pipeline.predict_proba(df_features)
    except Exception as e:
        return f"❌ **Error** al predecir: {e}"

    pred_indices = np.argmax(probas, axis=1)
    pred_labels = clases[pred_indices]
    confianzas = np.max(probas, axis=1)

    # Construir DataFrame con info de partidos
    info_partido_cols = [
        "id_partido",
        "nombre_equipo_local",
        "nombre_equipo_visitante",
        "goles_equipo_local",
        "goles_equipo_visitante"
    ]
    for col in info_partido_cols:
        if col not in df_jornada.columns:
            return f"❌ **Error**: No se encontró la columna '{col}' en el CSV."

    df_jornada_info = df_jornada[info_partido_cols].copy()
    df_jornada_info["Predicción"] = pred_labels
    df_jornada_info["Confianza"] = confianzas

    # Ordenar por confianza y tomar top 4
    df_top4 = df_jornada_info.sort_values("Confianza", ascending=False).head(4)

    # Construcción de texto de salida
    resultado = [f"**TOP 4 de la Jornada {jornada} - Modelo `{modelo}`**\n"]
    for _, row in df_top4.iterrows():
        pid = row["id_partido"]
        local = row["nombre_equipo_local"]
        visitante = row["nombre_equipo_visitante"]
        pred = row["Predicción"]
        conf = row["Confianza"]
        txt = (
            f"Partido ID {pid}\n"
            f"_{local}_ vs _{visitante}_\n"
            f"   - Predicción: **{pred}** (conf: {conf:.2f})\n"
        )
        resultado.append(txt)

    return "\n".join(resultado)

# -----------------------------------------------------
#   COMPARACIÓN DE TOP 4 (IGNORANDO EMPATES)
# -----------------------------------------------------

def comparar_top4(modelo: str, csv_path: str, jornada: int) -> str:
    """
    1. Carga el modelo y sus clases.
    2. Filtra la jornada completa del CSV.
    3. Predice TODOS los partidos, toma el Top 4 por confianza.
    4. Normaliza la predicción (Local/Visitante/Empate).
    5. EXCLUYE EMPATES en la estadística (ni acierto ni fallo).
    6. Devuelve un string con el reporte de aciertos y fallos del Top 4.
    """
    model_folder = os.path.join(MODELS_DIR, modelo)
    model_path = os.path.join(model_folder, f"{modelo}.pkl")
    classes_path = os.path.join(model_folder, "classes.npy")

    # Verificar existencia de archivos
    if not os.path.exists(model_path):
        return f"❌ **Error**: No se encontró el archivo del modelo: {model_path}"
    if not os.path.exists(classes_path):
        return f"❌ **Error**: No se encontró el archivo de clases: {classes_path}"

    # Cargar y filtrar CSV
    df_jornada = cargar_y_filtrar_csv(csv_path, jornada)
    if isinstance(df_jornada, str):
        return df_jornada  # si hay error, df_jornada es un string con el mensaje

    # Cargar modelo y clases
    try:
        pipeline = joblib.load(model_path)
        clases = np.load(classes_path, allow_pickle=True)
    except Exception as e:
        return f"❌ **Error** al cargar modelo/clases: {e}"

    # Lista de features (ajusta con tus columnas reales)
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

    # Agregar columnas faltantes con 0
    missing = set(features) - set(df_jornada.columns)
    if missing:
        for col in missing:
            df_jornada[col] = 0
        logger.warning(f"Se agregaron columnas faltantes con 0: {', '.join(missing)}")

    # Sub-DataFrame y conversión a numérico
    df_features = df_jornada[features].copy()
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce").fillna(0)

    # Predicción
    try:
        probas = pipeline.predict_proba(df_features)
    except Exception as e:
        return f"❌ **Error** al predecir: {e}"

    pred_indices = np.argmax(probas, axis=1)
    pred_labels = clases[pred_indices]
    confianzas = np.max(probas, axis=1)

    # Construir DataFrame con info de partidos
    info_partido_cols = [
        "id_partido",
        "nombre_equipo_local",
        "nombre_equipo_visitante",
        "goles_equipo_local",
        "goles_equipo_visitante"
    ]
    for col in info_partido_cols:
        if col not in df_jornada.columns:
            return f"❌ **Error**: No se encontró la columna '{col}' en el CSV."

    df_jornada_info = df_jornada[info_partido_cols].copy()
    df_jornada_info["Predicción"] = pred_labels
    df_jornada_info["Confianza"] = confianzas

    # Ordenar por confianza y tomar top 4
    df_top4 = df_jornada_info.sort_values("Confianza", ascending=False).head(4)

    # --- Lógica de comparación ---
    aciertos = 0
    fallos = 0
    partidos_ignorados = 0
    detalles = []

    for _, row in df_top4.iterrows():
        pid = row["id_partido"]
        local = row["nombre_equipo_local"]
        visitante = row["nombre_equipo_visitante"]
        goles_local = row["goles_equipo_local"]
        goles_visitante = row["goles_equipo_visitante"]
        pred_raw = row["Predicción"]
        conf = row["Confianza"]

        # Normalizar (Local, Visitante, Empate)
        pred_norm = normalizar_prediccion(pred_raw)

        # Resultado real
        if goles_local > goles_visitante:
            real = "Local"
        elif goles_local < goles_visitante:
            real = "Visitante"
        else:
            real = "Empate"

        # Construcción de texto
        linea = (
            f"Partido ID {pid}\n"
            f"_{local}_ vs _{visitante}_\n"
            f"   - Predicción: **{pred_norm}** (conf: {conf:.2f})\n"
            f"   - Resultado real: **{real}**\n"
        )

        # Ignoramos empates en la estadística
        if real == "Empate":
            linea += "   - [IGNORADO] (el resultado real fue Empate)\n"
            partidos_ignorados += 1
        elif pred_norm == "Empate":
            linea += "   - [IGNORADO] (la predicción es Empate)\n"
            partidos_ignorados += 1
        else:
            # Comparar
            if pred_norm == real:
                linea += "   - **ACIERTO**\n"
                aciertos += 1
            else:
                linea += "   - **FALLO**\n"
                fallos += 1

        detalles.append(linea)

    # Resumen
    total_tomados = len(df_top4) - partidos_ignorados
    texto_resumen = (
        f"**TOP 4 de la Jornada {jornada} - Modelo `{modelo}`**\n\n"
        + "\n".join(detalles)
        + "\n"
        + f"Aciertos (sin empates): {aciertos}\n"
        + f"Fallos (sin empates): {fallos}\n"
        + f"Partidos tomados en cuenta (sin Empate): {total_tomados}\n"
    )

    return texto_resumen

# -----------------------------------------------------
#   COMPROBAR TODAS LAS JORNADAS (opcional)
# -----------------------------------------------------

def comprobar_toda_la_liga(modelo: str, csv_path: str, jornadas=38) -> str:
    """
    1. Para cada jornada (1..jornadas), llama a 'comparar_top4'.
    2. Acumula resultados en un reporte global.
    """
    detalles_por_jornada = []
    for j in range(1, jornadas + 1):
        reporte_jornada = comparar_top4(modelo, csv_path, j)
        if "❌ **Error**" in reporte_jornada:
            detalles_por_jornada.append(f"Jornada {j} => Error:\n{reporte_jornada}")
            continue
        detalles_por_jornada.append(f"**Jornada {j}**\n{reporte_jornada}\n")

    resultado_final = [
        f"**Comprobación de TODA la liga (1..{jornadas})**\n",
        *detalles_por_jornada
    ]
    return "\n".join(resultado_final)

# -----------------------------------------------------
#         UTILIDADES PARA ENVIAR MENSAJES
# -----------------------------------------------------

async def send_and_track_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup=None,
    parse_mode=None
):
    """
    Envía un mensaje y registra su ID en el contexto del usuario
    para poder eliminarlo posteriormente.
    """
    message = await update.effective_message.reply_text(
        text=text,
        reply_markup=reply_markup,
        parse_mode=parse_mode
    )
    context.user_data.setdefault("messages", []).append(message.message_id)
    return message

async def limpiar_mensajes_anteriores(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Elimina todos los mensajes registrados en context.user_data['messages'].
    """
    if "messages" in context.user_data:
        for msg_id in context.user_data["messages"]:
            try:
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=msg_id
                )
            except Exception as e:
                logger.warning(f"No se pudo eliminar el mensaje {msg_id}: {e}")
        context.user_data["messages"].clear()

# -----------------------------------------------------
#           HANDLERS DE COMANDOS
# -----------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler del comando /start:
      1) Limpia mensajes anteriores
      2) Muestra lista de modelos disponibles
    """
    await limpiar_mensajes_anteriores(update, context)

    modelos = obtener_modelos()
    context.user_data["csv_files"] = obtener_csv()

    if not modelos:
        await send_and_track_message(update, context, "❌ **Error:** No se encontraron modelos disponibles.")
        return

    keyboard = [
        [InlineKeyboardButton(modelo, callback_data=f"modelo_{modelo}")]
        for modelo in modelos
    ]
    keyboard.append([InlineKeyboardButton("❌ Cancelar", callback_data="cancelar")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await send_and_track_message(
        update,
        context,
        "📂 **Selecciona un modelo:**",
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler del comando /help: muestra una breve descripción.
    """
    help_text = (
        "ℹ️ **Comandos disponibles:**\n\n"
        "/start - Inicia la selección de modelo, CSV y jornada\n"
        "/help - Muestra este mensaje de ayuda\n\n"
        "Flujo típico:\n"
        "1. /start para iniciar\n"
        "2. Seleccionar modelo\n"
        "3. Seleccionar CSV (liga)\n"
        "4. Seleccionar jornada o comprobar TODA la liga\n"
        "5. Ver sólo los 4 partidos con mayor probabilidad\n"
        "6. Comparar con resultados reales (Top 4), ignorando empates\n"
        "7. Volver atrás, predecir de nuevo o finalizar\n"
        "8. O comprobar TODA la liga de 1..38 y ver aciertos globales."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# -----------------------------------------------------
#           HANDLERS DE CALLBACKS
# -----------------------------------------------------

async def seleccionar_modelo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Callback para cuando el usuario selecciona un modelo.
    Se guarda el modelo y se muestra la lista de CSV disponibles.
    """
    query = update.callback_query
    await query.answer()

    modelo = query.data.split("_", 1)[1]
    context.user_data["modelo"] = modelo

    csv_files = context.user_data.get("csv_files", {})
    if not csv_files:
        await query.edit_message_text("❌ **Error:** No se encontraron archivos CSV disponibles.")
        return

    keyboard = [
        [InlineKeyboardButton(nombre, callback_data=f"csv_{nombre}")]
        for nombre in csv_files.keys()
    ]
    keyboard.append([InlineKeyboardButton("❌ Cancelar", callback_data="cancelar")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=f"✅ **Modelo seleccionado:** {modelo}\n📄 **Selecciona un archivo CSV:**",
        reply_markup=reply_markup
    )

async def seleccionar_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Tras elegir el CSV (liga), ofrecemos elegir una jornada específica
    o "Comprobar toda la liga".
    """
    query = update.callback_query
    await query.answer()

    csv_nombre = query.data.split("_", 1)[1]
    csv_files = context.user_data.get("csv_files", {})

    if csv_nombre not in csv_files:
        await query.edit_message_text("❌ **Error:** Archivo CSV no encontrado.")
        return

    context.user_data["csv"] = csv_files[csv_nombre]

    # Armamos las opciones: Jornadas + opción de comprobar toda la liga
    keyboard = [
        [InlineKeyboardButton(f"🔢 Jornada {j}", callback_data=f"jornada_{j}")]
        for j in JORNADAS
    ]
    keyboard.append([InlineKeyboardButton("📋 Comprobar TODA la liga", callback_data="comprobar_toda_liga")])
    keyboard.append([InlineKeyboardButton("❌ Cancelar", callback_data="cancelar")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=(
            f"✅ **Archivo CSV seleccionado:** {csv_nombre}\n"
            f"📅 **Selecciona una jornada o comprueba toda la liga:**"
        ),
        reply_markup=reply_markup
    )

async def seleccionar_jornada(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Cuando el usuario elige la jornada, predecimos los 4 partidos
    con mayor probabilidad y mostramos el resultado con opciones.
    """
    query = update.callback_query
    await query.answer()

    jornada = int(query.data.split("_", 1)[1])
    context.user_data["jornada"] = jornada

    modelo = context.user_data.get("modelo")
    csv_path = context.user_data.get("csv")

    if not modelo or not csv_path:
        await query.edit_message_text(
            "❌ **Error:** Faltan datos para ejecutar la predicción. Por favor, usa /start."
        )
        return

    await query.edit_message_text("⏳ **Procesando tu predicción (Top 4)...**")

    loop = asyncio.get_running_loop()
    def run_top4():
        return predecir_top4(modelo, csv_path, jornada)

    # Ejecutar predecir_top4 en un thread para no bloquear
    prediccion_text = await loop.run_in_executor(executor, run_top4)
    context.user_data["prediccion_text"] = prediccion_text

    keyboard = [
        [InlineKeyboardButton("✅ Comparar con resultado real (Top4)", callback_data="comprobar_resultado")],
        [InlineKeyboardButton("🔄 Predecir de nuevo", callback_data="predecir_de_nuevo")],
        [InlineKeyboardButton("❌ Finalizar", callback_data="finalizar")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await send_and_track_message(
        update,
        context,
        text=prediccion_text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def comprobar_resultado_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Cuando el usuario pulsa "Comparar con resultado real (Top4)",
    comparamos sólo el Top 4 de partidos con la lógica de aciertos y empates ignorados.
    """
    query = update.callback_query
    await query.answer()

    modelo = context.user_data.get("modelo")
    csv_path = context.user_data.get("csv")
    jornada = context.user_data.get("jornada")

    if not modelo or not csv_path or not jornada:
        await query.edit_message_text("❌ **Error**: Faltan datos para comparar.")
        return

    await query.edit_message_text("⏳ **Comparando resultados Top 4 (ignorando empates)...**")

    loop = asyncio.get_running_loop()
    def run_compare():
        return comparar_top4(modelo, csv_path, jornada)

    comparison_text = await loop.run_in_executor(executor, run_compare)
    context.user_data["comparison_text"] = comparison_text

    keyboard = [
        [InlineKeyboardButton("⬅️ Volver atrás", callback_data="volver_prediccion")],
        [InlineKeyboardButton("🔄 Predecir de nuevo", callback_data="predecir_de_nuevo")],
        [InlineKeyboardButton("❌ Finalizar", callback_data="finalizar")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=comparison_text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def volver_prediccion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Cuando el usuario pulsa "Volver atrás" tras comparar,
    volvemos a mostrar la predicción original con su teclado.
    """
    query = update.callback_query
    await query.answer()

    prediccion_text = context.user_data.get("prediccion_text")
    if not prediccion_text:
        await query.edit_message_text("❌ **No hay predicción previa almacenada.**")
        return

    keyboard = [
        [InlineKeyboardButton("✅ Comparar con resultado real (Top4)", callback_data="comprobar_resultado")],
        [InlineKeyboardButton("🔄 Predecir de nuevo", callback_data="predecir_de_nuevo")],
        [InlineKeyboardButton("❌ Finalizar", callback_data="finalizar")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=prediccion_text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def comprobar_toda_liga_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Cuando el usuario pulsa "Comprobar TODA la liga":
    Itera jornadas 1..38 (o personalizadas) y muestra reporte.
    """
    query = update.callback_query
    await query.answer()

    modelo = context.user_data.get("modelo")
    csv_path = context.user_data.get("csv")
    max_jornadas = 38  # Ajustar según corresponda

    if not modelo or not csv_path:
        await query.edit_message_text("❌ **Error**: Faltan datos para comparar toda la liga.")
        return

    await query.edit_message_text("⏳ **Comprobando TODA la liga...**")

    loop = asyncio.get_running_loop()
    def run_toda():
        return comprobar_toda_la_liga(modelo, csv_path, max_jornadas)

    reporte_toda_liga = await loop.run_in_executor(executor, run_toda)

    keyboard = [
        [InlineKeyboardButton("🔄 Predecir de nuevo", callback_data="predecir_de_nuevo")],
        [InlineKeyboardButton("❌ Finalizar", callback_data="finalizar")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=reporte_toda_liga,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

# -----------------------------------------------------
#           HANDLERS DE CALLBACKS (CONTINUACIÓN)
# -----------------------------------------------------

async def predecir_de_nuevo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Limpia mensajes y reinicia el proceso con /start.
    """
    query = update.callback_query
    await query.answer()
    await limpiar_mensajes_anteriores(update, context)
    await start(update, context)

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Mensaje de despedida y se limpia el contexto.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("✅ **Gracias por usar el bot. ¡Hasta luego!**")
    context.user_data.clear()

async def manejar_cancelacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    El usuario cancela el proceso.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("❌ **Proceso cancelado.**")
    context.user_data.clear()

# -----------------------------------------------------
#       MANEJADOR GLOBAL DE ERRORES
# -----------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """
    Captura excepciones globales y notifica al usuario si es posible.
    """
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ **Ocurrió un error inesperado. Por favor, intenta más tarde.**"
            )
        except Exception as e:
            logger.error(f"No se pudo enviar mensaje de error al usuario: {e}")

# -----------------------------------------------------
#              FUNCIÓN PRINCIPAL
# -----------------------------------------------------

def main():
    """
    Punto de entrada del script: configura el bot y arranca el polling.
    """
    application = ApplicationBuilder().token(TOKEN).build()

    # Handlers de comandos
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Handlers de callbacks (inline buttons)
    application.add_handler(CallbackQueryHandler(seleccionar_modelo, pattern="^modelo_"))
    application.add_handler(CallbackQueryHandler(seleccionar_csv, pattern="^csv_"))
    application.add_handler(CallbackQueryHandler(seleccionar_jornada, pattern="^jornada_"))
    application.add_handler(CallbackQueryHandler(manejar_cancelacion, pattern="^cancelar$"))
    application.add_handler(CallbackQueryHandler(predecir_de_nuevo, pattern="^predecir_de_nuevo$"))
    application.add_handler(CallbackQueryHandler(finalizar, pattern="^finalizar$"))

    # Para comparar resultados top4 (ignorando empates)
    application.add_handler(CallbackQueryHandler(comprobar_resultado_callback, pattern="^comprobar_resultado$"))
    application.add_handler(CallbackQueryHandler(volver_prediccion, pattern="^volver_prediccion$"))

    # Handler para comprobar toda la liga
    application.add_handler(CallbackQueryHandler(comprobar_toda_liga_callback, pattern="^comprobar_toda_liga$"))

    # Manejador global de errores
    application.add_error_handler(error_handler)

    logger.info("📡 Iniciando el bot de Telegram...")
    application.run_polling()


if __name__ == "__main__":
    main()
