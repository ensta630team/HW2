def check_num(number):
    """
    Formatea un número para su visualización, manejando casos especiales.

    Argumentos:
        number: El número a formatear (puede ser None, float, int, etc.).

    Retorna:
        str: El número formateado como una cadena de texto.
    """
    # Si el número es None, retorna 'N/A' para indicar que no hay un valor.
    if number is None:
        return 'N/A'
    # Si es un número flotante, lo redondea a 2 decimales para una mejor presentación.
    elif isinstance(number, float):
        number = round(number, 2)
        return str(number)
    # Para cualquier otro tipo (como enteros), simplemente lo convierte a cadena.
    else:       
        return str(number)