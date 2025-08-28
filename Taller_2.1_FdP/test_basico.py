# Ejemplo de buenas prácticas en nuestra librería

print("=== EJEMPLO DE BUENAS PRÁCTICAS ===")

class Vector:
    def __init__(self, comps):
        self._data = [float(x) for x in comps]
    def __len__(self): return len(self._data)
    def __repr__(self): return f"Vector({self._data})"

def vector_add(v1: Vector, v2: Vector) -> Vector:
    """
    Suma dos vectores componente por componente.

    Args:
        v1: Primer vector
        v2: Segundo vector

    Returns:
        Un nuevo vector resultado de la suma

    Raises:
        ValueError: Si los vectores tienen dimensiones diferentes
        TypeError: Si los argumentos no son vectores

    Example:
        >>> v1 = Vector([1, 2])
        >>> v2 = Vector([3, 4])
        >>> vector_add(v1, v2)
        Vector([4.0, 6.0])
    """
    # Validación de tipos
    if not isinstance(v1, Vector) or not isinstance(v2, Vector):
        raise TypeError("Ambos argumentos deben ser vectores")

    # Validación de dimensiones
    if len(v1) != len(v2):
        raise ValueError("Los vectores deben tener la misma dimensión")

    # Implementación (usamos la representación interna _data)
    result_components = [a + b for a, b in zip(v1._data, v2._data)]
    return Vector(result_components)

print("Docstring de vector_add:")
print(vector_add.__doc__)

print("\n=== EJEMPLO DE MANEJO DE ERRORES ===")

def safe_divide(a, b):
    """División segura con manejo de errores."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Los argumentos deben ser números")
    if b == 0:
        raise ValueError("No se puede dividir por cero")
    return a / b

# Uso con manejo de excepciones
try:
    resultado = safe_divide(10, 2)
    print(f"10 / 2 = {resultado}")

    resultado = safe_divide(10, 0)  # Esto generará un error
except ValueError as e:
    print(f"Error de valor: {e}")
except TypeError as e:
    print(f"Error de tipo: {e}")