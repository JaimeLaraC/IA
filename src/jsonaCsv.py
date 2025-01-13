import json
import pandas as pd
import os
from datetime import datetime

def seleccionar_archivo_json(carpeta_json):
    """
    Lista los archivos JSON en la carpeta especificada y permite al usuario seleccionar uno.
    Si solo hay un archivo, lo selecciona automáticamente.
    """
    json_files = [f for f in os.listdir(carpeta_json) if f.endswith('.json')]

    if not json_files:
        print(f"No se encontraron archivos JSON en la carpeta '{carpeta_json}'.")
        exit(1)

    print("Archivos JSON disponibles en la carpeta 'outputs':")
    for idx, file in enumerate(json_files, start=1):
        print(f"{idx}. {file}")

    if len(json_files) == 1:
        archivo_seleccionado = json_files[0]
        print(f"\nSe ha seleccionado automáticamente el único archivo disponible: {archivo_seleccionado}")
    else:
        while True:
            try:
                seleccion = int(input(f"Seleccione el número del archivo JSON que desea convertir a CSV (1-{len(json_files)}): "))
                if 1 <= seleccion <= len(json_files):
                    archivo_seleccionado = json_files[seleccion - 1]
                    break
                else:
                    print(f"Por favor, ingrese un número entre 1 y {len(json_files)}.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número entero.")

    return archivo_seleccionado

def filtrar_temporadas(df, omitir_2024):
    """
    Filtra el DataFrame para omitir o incluir únicamente la temporada 2024.
    """
    if omitir_2024:
        df_filtrado = df[df['temporada'] != '2024']
        print("Se omitirá la temporada 2024 durante la conversión.")
    else:
        df_filtrado = df[df['temporada'] == '2024']
        print("Solo se incluirá la temporada 2024 durante la conversión.")
    return df_filtrado

def formatear_fecha(fecha_str):
    """
    Formatea una cadena de fecha del formato YYYY-MM-DD a DD/MM/YYYY.
    Si la fecha no es válida, retorna "Fecha inválida".
    """
    try:
        return datetime.strptime(fecha_str, "%Y-%m-%d").strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return "Fecha inválida"

def main():
    # ============================================
    # 1. Definición de las Rutas de las Carpetas
    # ============================================
    carpeta_json = 'outputs'  # Carpeta que contiene los archivos JSON
    carpeta_csv = 'data'      # Carpeta donde se guardarán los archivos CSV

    # Asegurarse de que la carpeta de salida exista
    os.makedirs(carpeta_csv, exist_ok=True)

    # ============================================
    # 2. Listar y Seleccionar el Archivo JSON
    # ============================================
    archivo_seleccionado = seleccionar_archivo_json(carpeta_json)
    ruta_json = os.path.join(carpeta_json, archivo_seleccionado)
    nombre_base = os.path.splitext(archivo_seleccionado)[0]
    nombre_csv = f"{nombre_base}.csv"
    ruta_csv = os.path.join(carpeta_csv, nombre_csv)

    print(f"\nHa seleccionado el archivo: {archivo_seleccionado}")
    print(f"El archivo CSV se guardará como: {nombre_csv} en la carpeta '{carpeta_csv}'.")

    # ============================================
    # 3. Solicitar al Usuario si Desea Omitir la Temporada 2024
    # ============================================
    while True:
        respuesta = input("\n¿Desea omitir la temporada 2024? (s/n): ").strip().lower()
        if respuesta in ['s', 'n']:
            omitir_2024 = respuesta == 's'
            break
        else:
            print("Entrada inválida. Por favor, ingrese 's' para sí o 'n' para no.")

    if omitir_2024:
        print("Se omitirá la temporada 2024 durante la conversión.")
    else:
        print("Solo se incluirá la temporada 2024 durante la conversión.")

    # ============================================
    # 4. Procesar el Archivo JSON y Convertir a CSV
    # ============================================
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error al leer {ruta_json}: {e}")
        exit(1)
    except FileNotFoundError:
        print(f"El archivo {ruta_json} no existe.")
        exit(1)

    # Validar la estructura del JSON
    if not isinstance(data, dict):
        print("El JSON no tiene la estructura esperada (debe ser un diccionario de temporadas).")
        exit(1)

    registros = []
    for temporada, partidos in data.items():
        temporada_str = str(temporada)
        if omitir_2024:
            if temporada_str == "2024":
                print(f"Omitiendo la temporada {temporada_str}.")
                continue
        else:
            if temporada_str != "2024":
                print(f"Omitiendo la temporada {temporada_str}.")
                continue

        if not isinstance(partidos, list):
            print(f"La temporada {temporada_str} no contiene una lista de partidos.")
            continue

        for partido in partidos:
            # Añadir la temporada al partido
            partido['temporada'] = temporada_str

            # Formatear la fecha del partido si existe
            fecha_partido_str = partido.get("fecha_partido", "")
            partido['fecha_partido'] = formatear_fecha(fecha_partido_str)

            # Añadir al registro
            registros.append(partido)

    if not registros:
        print("No hay partidos para procesar después de filtrar las temporadas.")
        exit(0)

    # Crear DataFrame
    df = pd.DataFrame(registros)

    # Identificar todas las columnas automáticamente
    todas_columnas = df.columns.tolist()

    # Reordenar las columnas para que 'temporada' esté al inicio (opcional)
    if 'temporada' in todas_columnas:
        todas_columnas.insert(0, todas_columnas.pop(todas_columnas.index('temporada')))
    df = df.reindex(columns=todas_columnas)

    # Opcional: Ordenar las columnas alfabéticamente
    # todas_columnas_sorted = sorted(todas_columnas)
    # df = df.reindex(columns=todas_columnas_sorted)

    # Guardar a CSV
    try:
        df.to_csv(ruta_csv, index=False, encoding='utf-8')
        print(f"\nEl archivo CSV '{nombre_csv}' se ha creado con éxito en la carpeta '{carpeta_csv}'.")
    except IOError as e:
        print(f"Error al escribir el archivo CSV: {e}")
        exit(1)

if __name__ == "__main__":
    main()
