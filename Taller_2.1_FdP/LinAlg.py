"""
Módulo principal de la librería de álgebra lineal
=================================================

Este módulo contiene las implementaciones de las clases Vector y Matrix,
así como las funciones de álgebra lineal asociadas.
"""

import math
from typing import List, Union, Tuple, Optional
from collections.abc import Sequence
from typing import Union


class Vector:
    """
    Clase para representar y manipular vectores.
    
    Un vector es una lista de números que puede representar
    puntos en el espacio, direcciones, o cualquier secuencia ordenada de valores.
    """
    
    def __init__(self, components: List[Union[int, float]]):
        """
        Inicializa un vector con sus componentes.
        Args:
            components: Lista de números que representan las componentes del vector
        """
        if components is None:
            raise ValueError("Components no puede ser None.")
        try:
            self._data: List[float] = [float(x) for x in components]
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Components debe ser un iterable de números (int/float)."
            ) from e
        if not self._data:
            raise ValueError("Un vector debe tener al menos un componente.")
    
    def _check_same_dimension(self, other: 'Vector', op: str = "operación") -> None:
        """Verifica que 'other' sea un Vector de la misma dimensión."""
        if not isinstance(other, Vector):
            raise TypeError(f"El otro operando debe ser Vector para {op}.")
        if len(self) != len(other):
            raise ValueError(
                f"Dimensiones incompatibles para {op}: {len(self)} != {len(other)}."
        )
            
    def _check_scalar(self, value, op: str = "operación") -> None:
        """Verifica que 'value' sea un escalar numérico (no bool)."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Solo se permite {op} por un escalar (int o float).")
    
    def __str__(self) -> str:
        """Representación en string del vector."""
        return f"Vector({self._data})"
    
    def __repr__(self) -> str:
        """Representación detallada del vector."""
        return f"Vector({self._data!r})"
    
    def __len__(self) -> int:
        """Retorna la dimensión del vector."""
        return len(self._data)
    
    def __getitem__(self, index: int) -> Union[int, float]:
        """Permite acceder a los componentes del vector usando índices."""
        return self._data[index]
    
    def __setitem__(self, index: int, value: Union[int, float]):
        """Permite modificar componentes del vector usando índices."""
        self._data[index] = float(value)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Suma de vectores usando el operador +."""
        self._check_same_dimension(other)
        return Vector([a + b for a, b in zip(self._data, other._data)])
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Resta de vectores usando el operador -."""
        self._check_same_dimension(other)
        return Vector([a - b for a, b in zip(self._data, other._data)])
    
    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """Multiplicación por escalar usando el operador *."""
        self._check_scalar(scalar, op = "multiplicación")
        return Vector([a * float(scalar) for a in self._data])
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Multiplicación por escalar (orden invertido)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """División por escalar usando el operador /."""
        self._check_scalar(scalar, op ="división")
        scalar = float(scalar)
        if scalar == 0.0:
            raise ZeroDivisionError("No se puede dividir un vector por cero.")
        return Vector([a / scalar for a in self._data])
    
    def __eq__(self, other: 'Vector') -> bool:
        """Igualdad entre vectores usando el operador ==."""
        if isinstance(other, Vector) and len(self) == len(other):
            return all(a == b for a, b in zip(self._data, other._data))
        return False
    
    def __ne__(self, other: 'Vector') -> bool:
        """Desigualdad entre vectores usando el operador !=."""
        if isinstance(other, Vector) and len(self) == len(other):
            return any(a != b for a, b in zip(self._data, other._data))
        return True
    
    @property
    def magnitude(self) -> float:
        """Calcula y retorna la magnitud (norma) del vector."""
        return math.sqrt(sum(x*x for x in self._data))
    
    @property
    def unit_vector(self) -> 'Vector':
        """Retorna el vector unitario (normalizado)."""
        mag = self.magnitude
        if mag == 0.0:
            raise ValueError("No se puede normalizar el vector cero.")
        return self / mag
    
    def dot(self, other: 'Vector') -> float:
        """
        Calcula el producto punto con otro vector.
        Args:
            other: Otro vector para el producto punto
        Returns:
            El producto punto como un número
        """
        self._check_same_dimension(other, op = "producto punto")
        return float(sum(a * b for a, b in zip(self._data, other._data)))
    
    def cross(self, other: 'Vector') -> 'Vector':
        """
        Calcula el producto cruz con otro vector (solo para vectores 3D).
        Args:
            other: Otro vector para el producto cruz
        Returns:
            Un nuevo vector resultado del producto cruz
        """
        if len(self) != 3 or len(other) != 3:
            raise ValueError("El producto cruz solo está definido para vectores 3D.")
        a1, a2, a3 = self._data
        b1, b2, b3 = other._data
        return Vector([
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ])
    
    def angle_with(self, other: 'Vector') -> float:
        """
        Calcula el ángulo entre este vector y otro.
        Args:
            other: Otro vector
        Returns:
            El ángulo en radianes
        """
        self._check_same_dimension(other, op = "cálculo de ángulo")
        mag1 = self.magnitude
        mag2 = other.magnitude
        if mag1 == 0.0 or mag2 == 0.0:
            raise ValueError("No se puede calcular el ángulo con el vector cero.")
        dot = self.dot(other)
        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)


class Matrix:
    """
    Clase para representar y manipular matrices.
    
    Una matriz es una colección rectangular de números organizados en filas y columnas.
    """
    
    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Inicializa una matriz con sus datos.
        Args:
            data: Lista de listas que representa las filas de la matriz
        """
        if data is None or not data:
            raise ValueError("Data no puede ser None o una lista vacía.")
        if any(not isinstance(row, Sequence) for row in data):
            raise TypeError("Cada fila debe ser una secuencia (lista o tupla) de números.")
        if any(len(row) == 0 for row in data):
            raise ValueError("Cada fila debe tener al menos una columna.")
        ncols = len(data[0])
        if any(len(row) != ncols for row in data):
            raise ValueError("Todas las filas deben tener la misma cantidad de columnas (matriz rectangular).")
        try:
            self._data: List[List[float]] = [[float(x) for x in row] for row in data]
        except (TypeError, ValueError) as e:
            raise TypeError("Data debe ser una secuencia de secuencias de números (int/float).") from e
        self._nrows = len(self._data)
        self._ncols = ncols
    
    def _check_same_shape(self, other: 'Matrix', op: str = "operación") -> None:
        if not isinstance(other, Matrix):
            raise TypeError(f"El otro operando debe ser Matrix para {op}.")
        if self._nrows != other._nrows or self._ncols != other._ncols:
            raise ValueError(
                f"Formas incompatibles para {op}: "
                f"{(self._nrows, self._ncols)} != {(other._nrows, other._ncols)}."
        )
    
    def __str__(self) -> str:
        """Representación en string de la matriz."""
        return f"Matrix({self._data})"
    
    def __repr__(self) -> str:
        """Representación detallada de la matriz."""
        return f"Matrix({self._data!r})"
    
    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[List[Union[int, float]], Union[int, float]]:
        """Permite acceder a filas o elementos específicos de la matriz."""
        if isinstance(key, int):
            return list(self._data[key])
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            return self._data[row][col]
        raise TypeError("Key debe ser un entero (para filas) o una tupla de dos enteros (para elementos específicos).")
    
    def __setitem__(self, key: Union[int, Tuple[int, int]], value: Union[List[Union[int, float]], Union[int, float]]):
        """Permite modificar filas o elementos específicos de la matriz."""
        if isinstance(key, int):
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise ValueError("El valor debe ser una secuencia (lista o tupla) de números.")
            if len(value) != self._ncols:
                raise ValueError(f"La fila asignada debe tener {self._ncols} columnas.")
            try:
                self._data[key] = [float(x) for x in value]
            except (TypeError, ValueError) as e:
                raise TypeError("El valor debe ser una secuencia de números (int/float).") from e
            return 
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            try:
                self._data[row][col] = float(value)
            except (TypeError, ValueError) as e:
                raise TypeError("El valor asignado debe ser un número (int/float).") from e
            return
        raise TypeError("Key debe ser un entero (para filas) o una tupla de dos enteros (para elementos específicos).")
        
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Suma de matrices usando el operador +."""
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other, op="suma")
        return Matrix([[a + b for a, b in zip(ra, rb)] for ra, rb in zip(self._data, other._data)])
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Resta de matrices usando el operador -."""
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other, op="resta")
        return Matrix([[a - b for a, b in zip(ra, rb)] for ra, rb in zip(self._data, other._data)])
    
    def __mul__(self, other: Union['Matrix', 'Vector', int, float]) -> Union['Matrix', 'Vector']:
        """Multiplicación de matrices/vectores/escalares usando el operador *."""
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            alpha = float(other)
            return Matrix([[alpha * a for a in row] for row in self._data])
        
        elif isinstance(other, Vector):
            if self._ncols != len(other):
                raise ValueError("El número de columnas de la matriz debe ser igual a la dimensión del vector para la multiplicación.")
            result = [sum(a * b for a, b in zip(row, other._data)) for row in self._data]
            return Vector(result)
        
        elif isinstance(other, Matrix):
            if self._ncols != other._nrows:
                raise ValueError("El número de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz para la multiplicación.")
            cols_B = list(zip(*other._data))
            prod = [[sum(a*b for a, b in zip(rowA, colB)) for colB in cols_B]
                    for rowA in self._data]
            return Matrix(prod)
        
        return NotImplemented
            
    def __rmul__(self, scalar: Union[int, float]) -> 'Matrix':
        """Multiplicación por escalar (orden invertido)."""
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            return self * float(scalar)
        return NotImplemented
    
    def __eq__(self, other: 'Matrix') -> bool:
        """Igualdad entre matrices usando el operador ==."""
        return isinstance(other, Matrix) and self._data == other._data
    
    def __ne__(self, other: 'Matrix') -> bool:
        """Desigualdad entre matrices usando el operador !=."""
        return (not isinstance(other, Matrix)) or (self._data != other._data)
    
    @property
    def num_rows(self) -> int:
        """Retorna el número de filas de la matriz."""
        return self._nrows
    
    @property
    def num_columns(self) -> int:
        """Retorna el número de columnas de la matriz."""
        return self._ncols
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Retorna las dimensiones de la matriz como (filas, columnas)."""
        return (self._nrows, self._ncols)
    
    @property
    def T(self) -> 'Matrix':
        """Retorna la transpuesta de la matriz."""
        return Matrix([list(row) for row in zip(*self._data)])
    
    @property
    def trace(self) -> Union[int, float]:
        """Calcula y retorna la traza de la matriz (suma de elementos diagonales)."""
        if self._nrows != self._ncols:
            raise ValueError("La traza está definida solo para matrices cuadradas.")
        t = sum(self._data[i][i] for i in range(self._nrows))
        
        return int(t) if float(t).is_integer() else t
    
    @property
    def determinant(self) -> Union[int, float]:
        """Calcula y retorna el determinante de la matriz."""
        if self._nrows != self._ncols:
            raise ValueError("El determinante está definido solo para matrices cuadradas.")
        
        n = self._nrows
        A = [row[:] for row in self._data]
        sign = 1.0
        eps = 1e-12
        
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[p][i]) <= eps:
                return 0.0
            if p != i:
                A[i], A[p] = A[p], A[i]
                sign *= -1.0
            for r in range(i + 1, n):
                factor = A[r][i] / A[i][i]
                for c in range(i, n):
                    A[r][c] -= factor * A[i][c]
        det = sign
        for i in range(n):
            det *= A[i][i]
        rounded = round(det)
        return int(rounded) if abs(det - rounded) <= 1e-9 else det
    
    @property
    def inverse(self) -> 'Matrix':
        """Calcula y retorna la matriz inversa."""
        self.ensure_square("inversa")
        n = self._nrows
        eps = 1e-12
        
        A = [row[:] for row in self._data]
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        for i in range(n):
            pivot_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            pivot_val = A[pivot_row][i]
            
            if abs(pivot_val) <= eps:
                raise ValueError("La matriz es singular; no tiene inversa.")
            
            if pivot_row != i:
                A[i], A[pivot_row] = A[pivot_row], A[i]
                I[i], I[pivot_row] = I[pivot_row], I[i]
                
            inv_p = 1.0 / A[i][i]
            for c in range(n):
                A[i][c] *= inv_p
                I[i][c] *= inv_p
            
            for r in range(n):
                if r == i:
                    continue
                
                factor = A[r][i]
                if abs(factor) <= eps:
                    continue
                for c in range(n):
                    A[r][c] -= factor * A[i][c]
                    I[r][c] -= factor * I[i][c]
                    
        return Matrix(I)
    
    def is_square(self) -> bool:
        """Verifica si la matriz es cuadrada."""
        return self._nrows == self._ncols
    
    def is_symmetric(self) -> bool:
        """Verifica si la matriz es simétrica."""
        if not self.is_square():
            return False
        for i in range(self._nrows):
            for j in range(i + 1, self._ncols):
                if self._data[i][j] != self._data[j][i]:
                    return False
        return True
    
    def is_diagonal(self) -> bool:
        """Verifica si la matriz es diagonal."""
        if self._nrows != self._ncols:
            return False 
        A = self._data
        n = self._nrows
        for i in range(n):
            for j in range(i + 1, n):
                if (not math.isclose(A[i][j], 0.0, rel_tol=1e-9, abs_tol=1e-12) or
                    not math.isclose(A[j][i], 0.0, rel_tol=1e-9, abs_tol=1e-12)):
                    return False
        return True
    
    def get_row(self, index: int) -> 'Vector':
        """
        Obtiene una fila específica como vector.
        
        Args:
            index: Índice de la fila
            
        Returns:
            Vector con los elementos de la fila
        """
        return Vector(self._data[index])
    
    def get_column(self, index: int) -> 'Vector':
        """
        Obtiene una columna específica como vector.
        
        Args:
            index: Índice de la columna
            
        Returns:
            Vector con los elementos de la columna
        """
        return Vector([self._data[i][index] for i in range(self._nrows)])


# =============================================================================
# FUNCIONES DE VECTOR
# =============================================================================

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Calcula el producto punto entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        El producto punto como un número
    """
    if not isinstance(v1, Vector) or not isinstance(v2, Vector):
        raise TypeError("Ambos argumentos deben ser Vector.")
    if len(v1) != len(v2):
        raise ValueError("Dimensiones incompatibles para producto punto.")
    return sum(a * b for a, b in zip(v1._data, v2._data))


def magnitude(v: Vector) -> float:
    """
    Calcula la magnitud (norma) de un vector.
    
    Args:
        v: El vector
        
    Returns:
        La magnitud del vector
    """
    if not isinstance(v, Vector):
        raise TypeError("El argumento debe ser un Vector.")
    return math.sqrt(sum(x*x for x in v._data))


def normalize(v: Vector) -> Vector:
    """
    Normaliza un vector (lo convierte en vector unitario).
    
    Args:
        v: El vector a normalizar
        
    Returns:
        Un nuevo vector normalizado
    """
    if not isinstance(v, Vector):
        raise TypeError("El argumento debe ser Vector.")
    m = magnitude(v)
    if m == 0.0:
        raise ValueError("No se puede normalizar el vector cero.")
    return Vector([x / m for x in v._data])


def cross_product(v1: Vector, v2: Vector) -> Vector:
    """
    Calcula el producto cruz entre dos vectores 3D.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Un nuevo vector resultado del producto cruz
    """
    if not isinstance(v1, Vector) or not isinstance(v2, Vector):
        raise TypeError("Ambos argumentos deben ser Vector.")
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("El producto cruz solo está definido para vectores 3D.")
    a1, a2, a3 = v1._data
    b1, b2, b3 = v2._data
    return Vector([
        a2 * b3 - a3 * b2,
        a3 * b1 - a1 * b3,
        a1 * b2 - a2 * b1
    ])


def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calcula el ángulo entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        El ángulo en radianes
    """
    if not isinstance(v1, Vector) or not isinstance(v2, Vector):
        raise TypeError("Ambos argumentos deben ser Vector.")
    if len(v1) != len(v2):
        raise ValueError("Dimensiones incompatibles para cálculo de ángulo.")
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0.0 or mag2 == 0.0:
        raise ValueError("No se puede calcular el ángulo con el vector cero.")
    cos_theta = dot_product(v1, v2) / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


# =============================================================================
# FUNCIONES DE MATRIZ
# =============================================================================

def scale(matrix: Matrix, scalar: Union[int, float]) -> Matrix:
    """
    Multiplica una matriz por un escalar.
    
    Args:
        matrix: La matriz
        scalar: El escalar
        
    Returns:
        Una nueva matriz escalada
    """
    if not isinstance(matrix, Matrix):
        raise TypeError("El primer argumento debe ser una instancia de Matrix.")
    if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
        raise TypeError("El escalar debe ser un número (int/float).")
    return Matrix * float(scalar)


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Suma dos matrices.
    
    Args:
        m1: Primera matriz
        m2: Segunda matriz
        
    Returns:
        Una nueva matriz resultado de la suma
    """
    return m1 + m2


def subtract(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Resta dos matrices.
    
    Args:
        m1: Primera matriz
        m2: Segunda matriz
        
    Returns:
        Una nueva matriz resultado de la resta
    """
    return m1 - m2


def vector_multiply(matrix: Matrix, vector: Vector) -> Vector:
    """
    Multiplica una matriz por un vector.
    
    Args:
        matrix: La matriz
        vector: El vector
        
    Returns:
        Un nuevo vector resultado de la multiplicación
    """
    return matrix * vector


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Multiplica dos matrices.
    
    Args:
        m1: Primera matriz
        m2: Segunda matriz
        
    Returns:
        Una nueva matriz resultado de la multiplicación
    """
    return m1 * m2


def transpose(matrix: Matrix) -> Matrix:
    """
    Calcula la transpuesta de una matriz.
    
    Args:
        matrix: La matriz
        
    Returns:
        Una nueva matriz transpuesta
    """
    return matrix.T


def determinant(matrix: Matrix) -> Union[int, float]:
    """
    Calcula el determinante de una matriz cuadrada.
    
    Args:
        matrix: La matriz cuadrada
        
    Returns:
        El determinante
    """
    return matrix.determinant


def inverse(matrix: Matrix) -> Matrix:
    """
    Calcula la matriz inversa.
    
    Args:
        matrix: La matriz cuadrada invertible
        
    Returns:
        Una nueva matriz inversa
    """
    return matrix.inverse


def identity_matrix(size: int) -> Matrix:
    """
    Crea una matriz identidad de tamaño especificado.
    
    Args:
        size: El tamaño de la matriz (size x size)
        
    Returns:
        Una nueva matriz identidad
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("El tamaño debe ser un entero positivo.")
    return Matrix([[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)])


def zeros_matrix(rows: int, columns: int) -> Matrix:
    """
    Crea una matriz de ceros con las dimensiones especificadas.
    
    Args:
        rows: Número de filas
        columns: Número de columnas
        
    Returns:
        Una nueva matriz llena de ceros
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or not isinstance(columns, int) or isinstance(columns, bool):
        raise TypeError("las filas y colummas deben ser enteros.")
    if rows <= 0 or columns <= 0:
        raise ValueError("las filas y columnas deben ser enteros positivos.")
    return Matrix([[0.0] * columns for _ in range(rows)])


def ones_matrix(rows: int, columns: int) -> Matrix:
    """
    Crea una matriz de unos con las dimensiones especificadas.
    
    Args:
        rows: Número de filas
        columns: Número de columnas
        
    Returns:
        Una nueva matriz llena de unos
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or not isinstance(columns, int) or isinstance(columns, bool):
        raise TypeError("Las filas y las columnas deben ser enteros.")

    if rows <= 0 or columns <= 0:
        raise ValueError("las filas y las columnas deben ser enteros positivos.")
    
    return Matrix([[1.0] * columns for _ in range(rows)])