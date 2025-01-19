import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

def list_json_files(outputs_path):
    """
    Lista todos los archivos JSON dentro de 'outputs_path'.

    Parameters:
    - outputs_path (Path): Ruta a la carpeta 'outputs'.

    Returns:
    - List[Path]: Lista de rutas completas a los archivos JSON.
    """
    json_files = list(outputs_path.glob("*.json"))
    return json_files

def display_files(json_files):
    """
    Muestra una lista numerada de archivos JSON.

    Parameters:
    - json_files (List[Path]): Lista de rutas de archivos JSON.
    """
    print("\nArchivos JSON disponibles para concatenar:")
    for idx, file in enumerate(json_files, start=1):
        print(f"{idx}. {file.name}")

def get_user_selection(num_files):
    """
    Solicita al usuario que seleccione archivos ingresando números separados por comas.

    Parameters:
    - num_files (int): Número total de archivos disponibles.

    Returns:
    - List[int]: Lista de índices seleccionados por el usuario.
    """
    while True:
        selection = input("\nIngresa los números de los archivos que deseas concatenar (separados por comas): ")
        try:
            # Separar la entrada por comas y convertir a enteros
            selected_indices = [int(num.strip()) for num in selection.split(',')]
            if not selected_indices:
                raise ValueError("No se seleccionó ningún archivo.")
            if any(num < 1 or num > num_files for num in selected_indices):
                raise ValueError("Uno o más números están fuera del rango válido.")
            return selected_indices
        except ValueError as ve:
            print(f"Entrada inválida: {ve}. Por favor, intenta de nuevo.")

def extract_season(fecha_partido):
    """
    Extrae el año de la fecha del partido para determinar la temporada.

    Parameters:
    - fecha_partido (str): Fecha del partido en formato ISO (e.g., "2023-08-11T19:00:00+00:00").

    Returns:
    - str: Año como cadena si es válido, de lo contrario una cadena vacía.
    """
    try:
        fecha = datetime.fromisoformat(fecha_partido)
        return str(fecha.year)
    except ValueError:
        return ""

def combine_selected_files(outputs_path, selected_files, output_file):
    """
    Combina los archivos JSON seleccionados organizados por temporadas y guarda el resultado.

    Parameters:
    - outputs_path (Path): Ruta a la carpeta 'outputs'.
    - selected_files (List[Path]): Lista de rutas de archivos JSON seleccionados.
    - output_file (Path): Ruta del archivo JSON de salida.
    """
    combined_data = {}
    for file_path in selected_files:
        try:
            with file_path.open('r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Procesamiento según la estructura del JSON
                if isinstance(data, dict):
                    for temporada, partidos in data.items():
                        if not isinstance(partidos, list):
                            logging.warning(f"La temporada '{temporada}' en '{file_path.name}' no contiene una lista válida. Se omitirá.")
                            continue
                        combined_data.setdefault(temporada, []).extend(partidos)
                
                elif isinstance(data, list):
                    for partido in data:
                        fecha_partido = partido.get("fecha_partido", "")
                        if not fecha_partido:
                            logging.warning(f"Partido sin 'fecha_partido' en '{file_path.name}': {partido}. Se omitirá.")
                            continue
                        temporada = extract_season(fecha_partido)
                        if not temporada:
                            logging.warning(f"Partido con 'fecha_partido' inválida: '{fecha_partido}' en '{file_path.name}'. Se omitirá.")
                            continue
                        combined_data.setdefault(temporada, []).append(partido)
                
                else:
                    logging.warning(f"El archivo '{file_path.name}' no contiene un formato válido. Se omitirá.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decodificando JSON en '{file_path.name}': {e}. Se omitirá.")
        except IOError as e:
            logging.error(f"Error leyendo '{file_path.name}': {e}. Se omitirá.")
    
    # Guardar los datos combinados en el archivo de salida
    try:
        with output_file.open('w', encoding='utf-8') as file:
            json.dump(combined_data, file, ensure_ascii=False, indent=4)
        logging.info(f"\nArchivos combinados guardados en: '{output_file}'")
    except IOError as e:
        logging.error(f"Error escribiendo en '{output_file}': {e}")

def main():
    # Configuración de logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Combina archivos JSON de resultados por temporada.")
    parser.add_argument(
        "--folder_path",
        default="/home/jaimelara/Escritorio/Repositorios/IA",
        help="Ruta a la carpeta principal que contiene 'outputs'. Por defecto: /home/jaime-lara/Escritorio/Repositorios/IA"
    )
    parser.add_argument(
        "--output_filename",
        default="resultados_Modelo_General_3.0.json",
        help="Nombre del archivo JSON de salida. Por defecto: resultados_Modelo_General_3.0.json"
    )
    args = parser.parse_args()
    
    # Ruta de la carpeta 'outputs'
    outputs_path = Path(args.folder_path) / "outputs"
    
    # Verificar si la carpeta 'outputs' existe
    if not outputs_path.exists() or not outputs_path.is_dir():
        logging.error(f"La carpeta 'outputs' no existe en '{args.folder_path}'.")
        return
    
    # Listar los archivos JSON disponibles
    json_files = list_json_files(outputs_path)
    
    if not json_files:
        logging.warning("No se encontraron archivos JSON en la carpeta 'outputs'.")
        return
    
    # Mostrar los archivos y obtener la selección del usuario
    display_files(json_files)
    selected_indices = get_user_selection(len(json_files))
    selected_files = [json_files[i - 1] for i in selected_indices]
    
    logging.info("\nHas seleccionado los siguientes archivos:")
    for file in selected_files:
        logging.info(f"- {file.name}")
    
    # Ruta del archivo de salida
    output_file = outputs_path / args.output_filename
    
    # Combinar los archivos seleccionados
    combine_selected_files(outputs_path, selected_files, output_file)

if __name__ == "__main__":
    main()
