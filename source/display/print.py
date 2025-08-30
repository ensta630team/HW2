import numpy as np 


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
    
def print_valores_criticos(values):
    estadisticos = np.sort(values)
    critic_value_1  = np.percentile(estadisticos, 99)
    critic_value_5  = np.percentile(estadisticos, 95)
    critic_value_10 = np.percentile(estadisticos, 90)
    print("Valor crítico al 1%:", critic_value_1)
    print("Valor crítico al 5%:", critic_value_5)
    print("Valor crítico al 10%:", critic_value_10)