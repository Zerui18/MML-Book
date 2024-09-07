from typing import Callable

from chapter2 import *


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Returns the dot product of two vectors. """
    return (v1 * v2).sum()


INNER_PRODUCT: Callable[[np.ndarray, np.ndarray], float] = dot_product
""" The inner product function used by the functions in this module. """


def set_inner_product(func: Callable[[np.ndarray, np.ndarray], float]):
    """ Sets the inner product function used by the functions in this module. """
    global INNER_PRODUCT
    INNER_PRODUCT = func


def get_inner_product() -> Callable[[np.ndarray, np.ndarray], float]:
    """ Returns the inner product function used by the functions in this module. """
    return INNER_PRODUCT


def norm(v: np.ndarray) -> float:
    """ Returns the norm of a vector. """
    return INNER_PRODUCT(v, v) ** 0.5


def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Returns the angle between two vectors. """
    return np.arccos(INNER_PRODUCT(v1, v2) / (norm(v1) * norm(v2)))


def distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Returns the distance between two vectors. """
    return norm(v1 - v2)


def is_orthogonal(v1: np.ndarray, v2: np.ndarray, tol=1e-10) -> bool:
    """ Returns True if the angle between two vectors is less than tol. """
    return np.isclose(angle(v1, v2), np.pi / 2, atol=tol)


def project(v: np.ndarray, B: np.ndarray, orthogonal: bool = False) -> np.ndarray:
    """ Returns the orthogonal projection of v onto the subspace spanned by B. 

        v: (m)
        B: (m, n)
        orthogonal: if True, assumes B is orthogonal and uses a faster method
        Returns: (m)
    """
    if orthogonal:
        # simply summing the projections onto the basis vectors
        # noinspection PyTypeChecker
        return sum([INNER_PRODUCT(v, b) * b / INNER_PRODUCT(b, b) for b in B.T])
    else:
        # general method using matrix inversion
        assert rk(B) == min(B.shape), 'B is not full rank. B.T @ B is not invertible.'
        return B @ (invert(B.T @ B) @ B.T @ v)


def find_orthonormal_basis_gram_schmidt(B: np.ndarray) -> np.ndarray:
    """ Returns an orthonormal basis for the subspace spanned by B, using the Gram-Schmidt process.
        B: (m, n)
        Returns: (m, min(n, rk(B)))
    """
    if not is_linearly_independent(B):
        print(
            f'B is not linearly independent: rk(B) = {rk(B)} != {B.shape[0]}, '
            f'reducing it to a linearly independent set.'
        )
        B = find_linind_vectors(B)  # (m, rk(B))
    Q = np.zeros_like(B, dtype=np.float64)
    for i in range(B.shape[1]):
        Q[:, i] = B[:, i] - project(B[:, i], Q[:, :i])
        Q[:, i] /= norm(Q[:, i])
    return Q


def givens_rotation(rad: float, dims: int, plane: tuple[int, int]) -> np.ndarray:
    """ Returns the transformation matrix for a rotation in the 2d-plane, in a space of `dims` dimensions.

        rad: the angle of rotation
        dims: the dimension of the space the rotation is in
        plane: the 2d-plane to rotate in, direction of rotation is from the first to the second axis
        Returns: (dims, dims)
    """
    R = np.eye(dims)
    i, j = plane
    c, s = np.cos(rad), np.sin(rad)
    R[i, i], R[i, j] = c, -s
    R[j, i], R[j, j] = s, c
    return R


def rotate(v: np.ndarray, rad: float, plane: tuple[int, int]) -> np.ndarray:
    """ Returns the vector v rotated by rad in the 2d-plane.

        v: (n)
        rad: the angle of rotation
        plane: the 2d-plane to rotate in, direction of rotation is from the first to the second dimension
        Returns: (n)
    """
    v = v.copy().astype(np.float64)
    i, j = plane
    c, s = np.cos(rad), np.sin(rad)
    vi, vj = v[i], v[j]
    v[i] = c * vi - s * vj
    v[j] = s * vi + c * vj
    return v


def rotate_slow(v: np.ndarray, rad: float, plane: tuple[int, int]) -> np.ndarray:
    """ Returns the vector v rotated by rad in the 2d-plane.

        v: (n)
        rad: the angle of rotation
        plane: the 2d-plane to rotate in, direction of rotation is from the first to the second dimension
        Returns: (n)

        This function is slower than `rotate_fast` and is only here for demonstration purposes.
    """
    return givens_rotation(rad, len(v), plane) @ v