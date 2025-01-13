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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# TOKEN DEL BOT (tómalo de variable de entorno o reemplázalo directamente)
TOKEN = "7402710972:AAF6o-l5mHLdrCmrtbtyL0-ZerYzxKuBQ6g"
# Si prefieres "quemar" el token (NO recomendado para producción):
# TOKEN = "TU_TOKEN_AQUI"

if not TOKEN:
    logger.error("El token de Telegram (TELEGRAM_TOKEN) no está configurado en las variables de entorno.")
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
    Retorna una lista con los nombres de los archivos .pkl
    en la carpeta MODELS_DIR, ordenados alfabéticamente.
    """
    modelos = []
    try:
        for root, _, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith(".pkl"):
                    modelos.append(os.path.splitext(file)[0])
        modelos.sort()
        return modelos
    except FileNotFoundError:
        logger.error(f"No se encontró el directorio de modelos: {MODELS_DIR}")
        return []

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
        return dict(sorted(csv_dict.items()))
    except FileNotFoundError:
        logger.error(f"No se encontró el directorio de CSVs: {CSV_DIR}")
        return {}

# -----------------------------------------------------
#    FUNCIÓN DE NORMALIZACIÓN DE PREDICCIÓN
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
    else:
        # Asumimos que lo demás es 'Empate'
        return "Empate"

# -----------------------------------------------------
#   PREDICCIÓN DE TOP 4 POR JORNADA
# -----------------------------------------------------

def predecir_top4(modelo: str, csv_path: str, jornada: int) -> str:
    """
    1. Carga el modelo y clases.
    2. Filtra la jornada completa del CSV.
    3. Predice TODOS los partidos, pero construye un Top 4 (ordenado por confianza).
    4. Devuelve un string con esos 4 partidos y su predicción (NO compara con goles).
    """
    model_folder = os.path.join(MODELS_DIR, modelo)
    model_path = os.path.join(model_folder, f"{modelo}.pkl")
    classes_path = os.path.join(model_folder, "classes.npy")

    # Verificar archivos
    if not os.path.exists(model_path):
        return f"❌ **Error**: No se encontró el archivo del modelo: {model_path}"
    if not os.path.exists(classes_path):
        return f"❌ **Error**: No se encontró el archivo de clases: {classes_path}"
    if not os.path.exists(csv_path):
        return f"❌ **Error**: No se encontró el CSV: {csv_path}"

    try:
        pipeline = joblib.load(model_path)
        clases = np.load(classes_path, allow_pickle=True)
    except Exception as e:
        return f"❌ **Error** al cargar modelo/clases: {e}"

    # Cargar CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"❌ **Error** al leer CSV '{csv_path}': {e}"

    df.columns = df.columns.str.strip().str.lower()
    if 'fecha' not in df.columns:
        return "❌ **Error**: No existe la columna 'fecha' en el CSV."

    df['jornada_numero'] = pd.to_numeric(df['fecha'], errors='coerce').astype('Int64')
    df_jornada = df[df['jornada_numero'] == jornada].copy()
    if df_jornada.empty:
        return f"❌ **Error**: No se encontraron datos para la jornada {jornada}."

    # Lista de features (ajústala según tu modelo)
    features = [
        'puntos_acumulados_local', 'puntos_acumulados_visitante',
        'wins_last10_local', 'draws_last10_local', 'losses_last10_local',
        'wins_last10_visitante', 'draws_last10_visitante', 'losses_last10_visitante',
        'goles_favor_last10_local', 'goles_contra_last10_local',
        'goles_favor_last10_visitante', 'goles_contra_last10_visitante',
        'goles_equipo_local', 'goles_equipo_visitante',
        'head2head_local_wins', 'head2head_local_draws', 'head2head_local_losses',
        'head2head_visitante_wins', 'head2head_visitante_draws', 'head2head_visitante_losses',
        'puntos_balance', 'goles_balance',
        'posicion_tabla_local', 'posicion_tabla_visitante',
        'dias_descanso_local', 'dias_descanso_visitante',
        'elo_rating_local', 'elo_rating_visitante',
        'valor_mercado_local', 'valor_mercado_visitante',
        'jugadores_lesionados_local', 'jugadores_lesionados_visitante',
        'titulares_sancionados_local', 'titulares_sancionados_visitante',
        'diff_posicion_tabla', 'diff_elo_rating', 'diff_dias_descanso'
    ]
    missing = set(features) - set(df_jornada.columns)
    if missing:
        return f"❌ **Error**: Faltan columnas en el CSV: {missing}"

    df_features = df_jornada[features].copy()

    # Predicción
    try:
        probas = pipeline.predict_proba(df_features)
    except Exception as e:
        return f"❌ **Error** al predecir: {e}"

    pred_indices = np.argmax(probas, axis=1)
    pred_labels = clases[pred_indices]
    confianzas = np.max(probas, axis=1)

    # Construimos un DataFrame para filtrar top4
    df_jornada_info = df_jornada[[
        'id_partido', 'fecha',
        'nombre_equipo_local', 'nombre_equipo_visitante'
    ]].copy()

    df_jornada_info['Predicción'] = pred_labels
    df_jornada_info['Confianza'] = confianzas

    # Ordenamos por confianza y tomamos top 4
    df_top4 = df_jornada_info.sort_values('Confianza', ascending=False).head(4)

    # Construimos texto
    resultado = [f"**TOP 4 de la Jornada {jornada} - Modelo `{modelo}`**\n"]
    for _, row in df_top4.iterrows():
        pid = row['id_partido']
        local = row['nombre_equipo_local']
        visitante = row['nombre_equipo_visitante']
        pred = row['Predicción']
        conf = row['Confianza']
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
    1. Carga el modelo y clases.
    2. Filtra la jornada completa del CSV.
    3. Predice TODOS los partidos, toma el Top 4 por confianza.
    4. Normaliza la predicción (si dice "Local Gana" -> "Local").
    5. EXCLUYE EMPATES de la estadística: no los cuenta ni como acierto ni como fallo.
    6. Devuelve un string con el reporte de aciertos.
    """
    model_folder = os.path.join(MODELS_DIR, modelo)
    model_path = os.path.join(model_folder, f"{modelo}.pkl")
    classes_path = os.path.join(model_folder, "classes.npy")

    if not os.path.exists(model_path):
        return f"❌ **Error**: No se encontró el archivo del modelo: {model_path}"
    if not os.path.exists(classes_path):
        return f"❌ **Error**: No se encontró el archivo de clases: {classes_path}"
    if not os.path.exists(csv_path):
        return f"❌ **Error**: No se encontró el CSV: {csv_path}"

    try:
        pipeline = joblib.load(model_path)
        clases = np.load(classes_path, allow_pickle=True)
    except Exception as e:
        return f"❌ **Error** al cargar modelo/clases: {e}"

    # Leer CSV y filtrar jornada
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"❌ **Error** al leer CSV '{csv_path}': {e}"

    df.columns = df.columns.str.strip().str.lower()
    if 'fecha' not in df.columns:
        return "❌ **Error**: No existe la columna 'fecha' en el CSV."

    df['jornada_numero'] = pd.to_numeric(df['fecha'], errors='coerce').astype('Int64')
    df_jornada = df[df['jornada_numero'] == jornada].copy()
    if df_jornada.empty:
        return f"❌ **Error**: No se encontraron datos para la jornada {jornada}."

    # Mismas features
    features = [
        'puntos_acumulados_local', 'puntos_acumulados_visitante',
        'wins_last10_local', 'draws_last10_local', 'losses_last10_local',
        'wins_last10_visitante', 'draws_last10_visitante', 'losses_last10_visitante',
        'goles_favor_last10_local', 'goles_contra_last10_local',
        'goles_favor_last10_visitante', 'goles_contra_last10_visitante',
        'goles_equipo_local', 'goles_equipo_visitante',
        'head2head_local_wins', 'head2head_local_draws', 'head2head_local_losses',
        'head2head_visitante_wins', 'head2head_visitante_draws', 'head2head_visitante_losses',
        'puntos_balance', 'goles_balance',
        'posicion_tabla_local', 'posicion_tabla_visitante',
        'dias_descanso_local', 'dias_descanso_visitante',
        'elo_rating_local', 'elo_rating_visitante',
        'valor_mercado_local', 'valor_mercado_visitante',
        'jugadores_lesionados_local', 'jugadores_lesionados_visitante',
        'titulares_sancionados_local', 'titulares_sancionados_visitante',
        'diff_posicion_tabla', 'diff_elo_rating', 'diff_dias_descanso'
    ]
    missing = set(features) - set(df_jornada.columns)
    if missing:
        return f"❌ **Error**: Faltan columnas en el CSV: {missing}"

    df_features = df_jornada[features].copy()

    # Predicción
    try:
        probas = pipeline.predict_proba(df_features)
        pred_indices = np.argmax(probas, axis=1)
        pred_labels = clases[pred_indices]
        confianzas = np.max(probas, axis=1)
    except Exception as e:
        return f"❌ **Error** al predecir: {e}"

    df_jornada_info = df_jornada[[
        'id_partido',
        'nombre_equipo_local',
        'nombre_equipo_visitante',
        'goles_equipo_local',
        'goles_equipo_visitante'
    ]].copy()

    df_jornada_info['PredicciónOriginal'] = pred_labels
    df_jornada_info['Confianza'] = confianzas

    # Ordenar por confianza y tomar top4
    df_jornada_info = df_jornada_info.sort_values('Confianza', ascending=False).head(4)

    # Normalizar la predicción
    df_jornada_info['PrediccionNorm'] = df_jornada_info['PredicciónOriginal'].apply(normalizar_prediccion)

    # Obtener resultado real
    def obtener_resultado_real(row):
        gl = row['goles_equipo_local']
        gv = row['goles_equipo_visitante']
        if gl > gv:
            return "Local"
        elif gl < gv:
            return "Visitante"
        else:
            return "Empate"

    df_jornada_info['ResultadoReal'] = df_jornada_info.apply(obtener_resultado_real, axis=1)

    # Comparar normalizada vs real
    df_jornada_info['Acierto'] = df_jornada_info['PrediccionNorm'] == df_jornada_info['ResultadoReal']

    # -- IGNORAR EMPATES EN ESTADÍSTICAS --
    df_empates = df_jornada_info[df_jornada_info['ResultadoReal'] == 'Empate'].copy()
    df_no_empates = df_jornada_info[df_jornada_info['ResultadoReal'] != 'Empate'].copy()

    total_sin_empates = len(df_no_empates)
    aciertos_sin_empates = df_no_empates['Acierto'].sum()
    numero_empates = len(df_empates)

    if total_sin_empates > 0:
        efectividad = 100.0 * aciertos_sin_empates / total_sin_empates
    else:
        efectividad = 0.0

    # Construir reporte
    reporte = []
    reporte.append(f"**Comparación (Top 4) - Jornada {jornada} - Modelo `{modelo}`**")
    reporte.append(f"- Total Top 4: {len(df_jornada_info)}")
    reporte.append(f"- Partidos ignorados (Empate): {numero_empates}")
    reporte.append(f"- Partidos tomados en cuenta (sin Empate): {total_sin_empates}")
    reporte.append(f"- Aciertos (sin empates): {aciertos_sin_empates}")
    reporte.append(f"- Efectividad (sin empates): {efectividad:.2f}%\n")

    # Mostrar detalles de los no empate
    for _, row in df_no_empates.iterrows():
        pid = row['id_partido']
        local = row['nombre_equipo_local']
        visitante = row['nombre_equipo_visitante']
        pred_orig = row['PredicciónOriginal']
        pred_norm = row['PrediccionNorm']
        real = row['ResultadoReal']
        conf = row['Confianza']
        icon = "✔️" if row['Acierto'] else "❌"

        txt = (
            f"Partido {pid}: {local} vs {visitante}\n"
            f"   - Predicción Original: `{pred_orig}` (conf: {conf:.2f})\n"
            f"   - Normalizada: `{pred_norm}`\n"
            f"   - Real: `{real}`\n"
            f"   - Acierto: {icon}\n"
        )
        reporte.append(txt)

    # (Opcional) mostrar los empates ignorados
    if not df_empates.empty:
        reporte.append("\n**Partidos ignorados (Empate):**\n")
        for _, row in df_empates.iterrows():
            pid = row['id_partido']
            local = row['nombre_equipo_local']
            visitante = row['nombre_equipo_visitante']
            pred_orig = row['PredicciónOriginal']
            pred_norm = row['PrediccionNorm']
            real = row['ResultadoReal']
            conf = row['Confianza']
            txt = (
                f"Partido {pid}: {local} vs {visitante}\n"
                f"   - Predicción Original: `{pred_orig}` (conf: {conf:.2f})\n"
                f"   - Normalizada: `{pred_norm}`\n"
                f"   - Real: `{real}` (Ignorado por Empate)\n"
            )
            reporte.append(txt)

    return "\n".join(reporte)

# -----------------------------------------------------
#   COMPROBAR TODAS LAS JORNADAS
# -----------------------------------------------------

def comprobar_toda_la_liga(modelo: str, csv_path: str, jornadas=38) -> str:
    """
    1. Para cada jornada (1..jornadas), llama a 'comparar_top4'.
    2. Acumula cuántos partidos (sin empates) y cuántos aciertos hubo.
    3. Retorna un reporte global y, opcionalmente, un resumen de cada jornada.
    """
    total_partidos_sin_empate_global = 0
    total_aciertos_global = 0
    detalles_por_jornada = []

    for j in range(1, jornadas + 1):
        reporte_jornada = comparar_top4(modelo, csv_path, j)
        if "❌ **Error**" in reporte_jornada:
            # Marcamos error en esa jornada y seguimos
            detalles_por_jornada.append(f"Jornada {j} => Error:\n{reporte_jornada}")
            continue
        
        # Parseamos con regex los totales
        match_total = re.search(r"Partidos tomados en cuenta \(sin Empate\): (\d+)", reporte_jornada)
        match_aciertos = re.search(r"Aciertos \(sin empates\): (\d+)", reporte_jornada)

        if match_total and match_aciertos:
            total_j = int(match_total.group(1))
            aciertos_j = int(match_aciertos.group(1))

            total_partidos_sin_empate_global += total_j
            total_aciertos_global += aciertos_j

            detalles_por_jornada.append(
                f"Jornada {j}: {aciertos_j} / {total_j}"
            )
        else:
            detalles_por_jornada.append(
                f"Jornada {j}: No se pudo extraer aciertos:\n{reporte_jornada}"
            )

    # Efectividad global
    if total_partidos_sin_empate_global > 0:
        efectividad_global = 100.0 * total_aciertos_global / total_partidos_sin_empate_global
    else:
        efectividad_global = 0.0

    resultado_final = []
    resultado_final.append(f"**Comprobación de TODA la liga (1..{jornadas})**")
    resultado_final.append(f"- Total de partidos (sin empates): {total_partidos_sin_empate_global}")
    resultado_final.append(f"- Aciertos globales: {total_aciertos_global}")
    resultado_final.append(f"- Efectividad global: {efectividad_global:.2f}%\n")

    resultado_final.append("**Resumen por Jornada:**")
    for d in detalles_por_jornada:
        resultado_final.append(f"- {d}")

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
    para poder eliminarlo después si se desea.
    """
    message = await update.effective_message.reply_text(
        text=text,
        reply_markup=reply_markup,
        parse_mode=parse_mode
    )
    context.user_data.setdefault('messages', []).append(message.message_id)
    return message

async def limpiar_mensajes_anteriores(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Elimina todos los mensajes registrados en context.user_data['messages'].
    """
    if 'messages' in context.user_data:
        for msg_id in context.user_data['messages']:
            try:
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=msg_id
                )
            except Exception as e:
                logger.warning(f"No se pudo eliminar el mensaje {msg_id}: {e}")
        context.user_data['messages'].clear()

# -----------------------------------------------------
#           HANDLERS DE COMANDOS
# -----------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler del comando /start:
      1) Limpia mensajes anteriores
      2) Muestra lista de modelos
    """
    await limpiar_mensajes_anteriores(update, context)

    modelos = obtener_modelos()
    context.user_data['csv_files'] = obtener_csv()

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
    query = update.callback_query
    await query.answer()

    modelo = query.data.split("_", 1)[1]
    context.user_data['modelo'] = modelo

    csv_files = context.user_data.get('csv_files', {})
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
    Tras elegir el CSV (liga), preguntamos por la jornada específica
    o la nueva opción: "Comprobar TODA la liga".
    """
    query = update.callback_query
    await query.answer()

    csv_nombre = query.data.split("_", 1)[1]
    csv_files = context.user_data.get('csv_files', {})

    if csv_nombre not in csv_files:
        await query.edit_message_text("❌ **Error:** Archivo CSV no encontrado.")
        return

    context.user_data['csv'] = csv_files[csv_nombre]

    # Armamos las opciones: jornadas + una nueva para "toda la liga"
    keyboard = [
        [InlineKeyboardButton(f"🔢 Jornada {j}", callback_data=f"jornada_{j}")]
        for j in JORNADAS
    ]
    # Nuevo botón para comprobar toda la liga
    keyboard.append([InlineKeyboardButton("📋 Comprobar TODA la liga", callback_data="comprobar_toda_liga")])
    keyboard.append([InlineKeyboardButton("❌ Cancelar", callback_data="cancelar")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=f"✅ **Archivo CSV seleccionado:** {csv_nombre}\n"
             f"📅 **Selecciona una jornada o comprueba toda la liga:**",
        reply_markup=reply_markup
    )

async def seleccionar_jornada(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Cuando el usuario elige la jornada, predecimos los 4 partidos con mayor probabilidad
    y mostramos el resultado con opciones para comparar, predecir de nuevo o finalizar.
    """
    query = update.callback_query
    await query.answer()

    jornada = int(query.data.split("_", 1)[1])
    context.user_data['jornada'] = jornada

    modelo = context.user_data.get('modelo')
    csv_path = context.user_data.get('csv')

    if not modelo or not csv_path:
        await query.edit_message_text(
            "❌ **Error:** Faltan datos para ejecutar la predicción. Por favor, usa /start."
        )
        return

    await query.edit_message_text("⏳ **Procesando tu predicción (Top 4)...**")

    loop = asyncio.get_running_loop()
    def run_top4():
        return predecir_top4(modelo, csv_path, jornada)

    prediccion_text = await loop.run_in_executor(executor, run_top4)

    # Guardamos el texto de la predicción en user_data para "Volver atrás" en caso de comparación
    context.user_data['prediccion_text'] = prediccion_text

    keyboard = [
        [InlineKeyboardButton("✅ Comparar con resultado real (Top4)", callback_data="comprobar_resultado")],
        [InlineKeyboardButton("🔄 Predecir de nuevo", callback_data="predecir_de_nuevo")],
        [InlineKeyboardButton("❌ Finalizar", callback_data="finalizar")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Enviamos el Top4 al usuario
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
    se compara sólo el Top 4 de partidos y se muestra el reporte de aciertos.
    Empates se ignoran (ni acierto ni fallo).
    Incluimos un botón "Volver atrás" para regresar al texto de la predicción.
    """
    query = update.callback_query
    await query.answer()

    modelo = context.user_data.get('modelo')
    csv_path = context.user_data.get('csv')
    jornada = context.user_data.get('jornada')

    if not modelo or not csv_path or not jornada:
        await query.edit_message_text("❌ **Error**: Faltan datos para comparar.")
        return

    await query.edit_message_text("⏳ **Comparando resultados Top 4 (ignorando empates)...**")

    loop = asyncio.get_running_loop()
    def run_compare():
        return comparar_top4(modelo, csv_path, jornada)

    comparison_text = await loop.run_in_executor(executor, run_compare)
    context.user_data['comparison_text'] = comparison_text  # por si se usa en otro lado

    # Construimos teclado con botón "Volver atrás"
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
    Cuando el usuario pulsa "Volver atrás" después de comparar,
    volvemos a mostrar la predicción original Top 4 y su teclado.
    """
    query = update.callback_query
    await query.answer()

    prediccion_text = context.user_data.get('prediccion_text')
    if not prediccion_text:
        await query.edit_message_text("❌ **No hay predicción previa almacenada.**")
        return

    # Reconstruimos el teclado post-predicción
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

# NUEVO CALLBACK: COMPROBAR TODA LA LIGA
async def comprobar_toda_liga_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Cuando el usuario pulsa "Comprobar TODA la liga":
      1) Itera jornadas 1..38 (o las que definas).
      2) Llama comparar_top4 para cada jornada.
      3) Acumula aciertos y totales.
      4) Devuelve un reporte global.
    """
    query = update.callback_query
    await query.answer()

    modelo = context.user_data.get('modelo')
    csv_path = context.user_data.get('csv')
    max_jornadas = 38  # O detecta automáticamente cuántas jornadas hay.

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

async def manejar_cancelacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    El usuario cancela el proceso.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("❌ **Proceso cancelado.**")
    context.user_data.clear()

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
    Mensaje de despedida y limpia el contexto.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("✅ **Gracias por usar el bot. ¡Hasta luego!**")
    context.user_data.clear()

# -----------------------------------------------------
#       MANEJADOR GLOBAL DE ERRORES
# -----------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
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

    # NUEVO handler para comprobar toda la liga
    application.add_handler(CallbackQueryHandler(comprobar_toda_liga_callback, pattern="^comprobar_toda_liga$"))

    # Manejador global de errores
    application.add_error_handler(error_handler)

    logger.info("📡 Iniciando el bot de Telegram...")
    application.run_polling()


if __name__ == "__main__":
    main()
