import json

def agregar_diff_posicion_tabla(input_path: str, output_path: str):
    """
    Lee un archivo JSON con partidos, agrega (o actualiza) la 
    clave 'diff_posicion_tabla' en cada partido, y lo guarda en otro JSON.
    """
    # 1. Cargar el JSON original
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Recorrer todas las temporadas y partidos
    #    (Asumiendo que data es un dict con la estructura { "YYYY": [ {..partido..}, ... ], ... })
    for year, matches in data.items():
        for match in matches:
            # Obtener valores de posición local/visitante
            pos_local = match.get("posicion_tabla_local", 0)
            pos_visitante = match.get("posicion_tabla_visitante", 0)

            # Calcular la diferencia
            diff_pos = pos_local - pos_visitante

            # Asignar/actualizar el valor en el diccionario
            match["diff_posicion_tabla"] = diff_pos

    # 3. Guardar el JSON con la nueva clave diff_posicion_tabla
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Ajusta la ruta de los archivos según tu necesidad
    input_file = "resultados_serieA.json"
    output_file = "resultados_serieA.json"

    agregar_diff_posicion_tabla(input_file, output_file)

    print(f"Se ha agregado 'diff_posicion_tabla' en '{output_file}'.")
