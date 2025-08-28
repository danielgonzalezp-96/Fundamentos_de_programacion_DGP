"""
Librería de Álgebra Lineal (LAL)
================================

Una librería personalizada para operaciones de álgebra lineal con vectores y matrices.

Módulos disponibles:
- Vector: clase para trabajar con vectores
- Matrix: clase para trabajar con matrices
- Funciones de vector: operaciones con vectores
- Funciones de matriz: operaciones con matrices
"""
from .LinAlg import Vector, Matrix
from .LinAlg import (
                     dot_product,
                     magnitude,
                     normalize,
                     angle_between,
                     cross_product,
                     scale,
                     add,
                     subtract,
                     vector_multiply,
                     matrix_multiply,
                     transpose,
                     determinant,
                     inverse,
                     identity_matrix,
                     zeros_matrix,
                     ones_matrix
                   )

_version_ = "1.0.0"
__author__ = "Daniel Alberto González Pabón"
__license__ = "UdeA"
__copyright__ = "Copyright 2025, Daniel Alberto González Pabón"
__all__ = [
    "Vector",
    "Matrix",
    "dot_product",
    "magnitude",
    "normalize",
    "angle_between",
    "cross_product",
    "scale",
    "add",
    "subtract",
    "vector_multiply",
    "matrix_multiply",
    "transpose",
    "determinant",
    "inverse",
    "identity_matrix",
    "zeros_matrix",
    "ones_matrix"
]