Librería de Álgebra Lineal (LAL)
Una librería personalizada para operaciones de álgebra lineal desarrollada como proyecto educativo para el curso de Fundamentos de Programación de la Universidad de Antioquia.

Estructura del proyecto
LinAlg/
├── __init__.py         # Configuración del módulo y exportaciones
├── LinAlg.py           # Implementación principal de clases y funciones
├── ejem_uso.py         # Ejemplos de cómo usar la librería
└── README.md           # Esta documentación
└── test_basico.py      #Prueba la funcionalidad de las clases y las funciones  

Clases principales
Vector
Representa un vector matemático con sus componentes.

Propiedades principales:

magnitude: Magnitud del vector
unit_vector: Vector normalizado

Métodos principales:

dot(other): Producto punto
cross(other): Producto cruz (vectores 3D)
angle_with(other): Ángulo entre vectores
Operadores soportados:

+, -: Suma y resta de vectores
*, /: Multiplicación y división por escalar
==, !=: Comparación de vectores

Matrix
Representa una matriz matemática.

Propiedades principales:

num_rows, num_columns: Dimensiones
shape: Forma como tupla (filas, columnas)
T: Transpuesta
trace: Traza (suma diagonal)
determinant: Determinante
inverse: Matriz inversa
Métodos principales:

is_square(): Verifica si es cuadrada
is_symmetric(): Verifica si es simétrica
get_row(index), get_column(index): Obtiene fila/columna
Operadores soportados:

+, -: Suma y resta de matrices
*: Multiplicación (matriz-matriz, matriz-vector, matriz-escalar)
==, !=: Comparación de matrices
Funciones del módulo
Funciones de Vector
dot_product(v1, v2): Producto punto
magnitude(v): Magnitud
normalize(v): Normalización
cross_product(v1, v2): Producto cruz
angle_between(v1, v2): Ángulo entre vectores
Funciones de Matrix
scale(matrix, scalar): Multiplicación por escalar
add(m1, m2): Suma de matrices
subtract(m1, m2): Resta de matrices
vector_multiply(matrix, vector): Multiplicación matriz-vector
matrix_multiply(m1, m2): Multiplicación matriz-matriz
transpose(matrix): Transpuesta
determinant(matrix): Determinante
inverse(matrix): Matriz inversa
Funciones de creación
identity_matrix(size): Matriz identidad
zeros_matrix(rows, cols): Matriz de ceros
ones_matrix(rows, cols): Matriz de unos