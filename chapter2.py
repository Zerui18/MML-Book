import numpy as np
from typing import Tuple


def find_ref(A: np.ndarray, reduced=False) -> np.ndarray:
    """ Returns the row echelon form of A, optionally reduced.
        A: (m, n)
        Returns: (m, n)
    """
    rows, columns = A.shape
    A = A.copy().astype(np.float64)
    # idx of current pivot candidate
    i, j = 0, 0
    while i < rows and j < columns:
        # check if column is all 0 from pivot down
        if np.allclose(A[i:, j], 0):
            # move to next column
            j += 1
            continue
        # partial pivoting
        # swap the greatest entry in the column to the pivot position
        new_pivot = np.argmax(np.abs(A[i:, j])) + i
        A[[i, new_pivot]] = A[[new_pivot, i]]
        # make pivot 1
        A[i] /= A[i, j]
        # make all entries below pivot 0
        for k in range(i + 1, rows):
            A[k] -= A[i] * A[k, j]
        if reduced:
            # make all entries above pivot 0
            for k in range(i):
                A[k] -= A[i] * A[k, j]
        # move to next row and column
        i += 1
        j += 1
    return A


def find_pivot_columns(ref: np.ndarray, return_indices=False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """ Returns the pivot columns of the given REF matrix.
        ref: (m, n)
        Returns: (m, r) where r is the rank of A.
    """
    rows, columns = ref.shape
    # find pivot columns
    pivot_columns = []
    pivot_rows = []
    i, j = 0, 0
    while i < rows and j < columns:
        if ref[i, j] == 1:
            pivot_columns.append(j)
            pivot_rows.append(i)
            i += 1
        j += 1
    if return_indices:
        return np.array(pivot_columns), np.array(pivot_rows)
    else:
        return ref[:, pivot_columns]


def rk(A: np.ndarray) -> int:
    """ Returns the rank of A. """
    return find_pivot_columns(find_ref(A)).shape[1]


def kernel_space(A: np.ndarray, is_rref=False) -> np.ndarray:
    """ Returns a basis of the kernel of A.
        A: (m, n)
        Returns: (m, k) where k is the dimension of ker(A).
        If A is not in REF, it is first reduced to REF.
    """
    rows, columns = A.shape
    if not is_rref:
        rref = find_ref(A, reduced=True)
    else:
        rref = A
    pivot_columns, pivot_rows = find_pivot_columns(rref, return_indices=True)
    non_pivot_columns = set(range(columns)) - set(pivot_columns)
    basis = []
    for j in non_pivot_columns:
        b = np.zeros(columns, dtype=np.float64)
        # copy over at rows corresponding to pivot columns
        b[pivot_columns] = rref[pivot_rows, j]
        # set current column to -1
        b[j] = -1
        basis.append(b)
    if len(basis) == 0:
        return np.zeros((rows, 1), dtype=np.float64)
    return np.array(basis).T


def invert(A: np.ndarray) -> np.ndarray:
    """ Returns the inverse of A.
        A: (n, n)
        Returns: (n, n)
    """
    # [A|I] -> [I|A^-1]
    assert A.shape[0] == A.shape[1], 'Matrix is not square.'
    augmented = np.hstack((A, np.eye(A.shape[0])))
    ref = find_ref(augmented, reduced=True)
    if not np.all(ref[:, :A.shape[0]] == np.eye(A.shape[0])):
        raise ValueError('Matrix is not invertible.')
    return ref[:, A.shape[0]:]


def solve_linear_systems(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the general solutions to AX = B in the form (particular solutions, basis of null space).
        A: (m, n)
        B: (m, p)
        Returns: (particular solutions: (n, p), basis of null space: (n, k)).
        k is the dimension of the null space of A.
    """
    augmented = np.hstack((A, B))
    rref = find_ref(augmented, reduced=True)
    augmented_A, augmented_B = rref[:, :-B.shape[1]], rref[:, -B.shape[1]:]
    # find unique solutions
    pivot_columns, pivot_rows = find_pivot_columns(augmented_A, return_indices=True)
    trailing_dims = augmented_B[pivot_rows[-1] + 1:]  # dims that cannot be expressed with pivots
    if not np.allclose(trailing_dims, 0):
        raise ValueError('No exact solution.')
    unique_solutions = np.zeros((A.shape[1], B.shape[1]))
    unique_solutions[pivot_columns] = augmented_B[pivot_rows]
    # find basis of null space
    null_space_basis = kernel_space(augmented_A, is_rref=True)
    return unique_solutions, null_space_basis


def find_change_of_basis_matrix(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """ Returns the change of basis matrix from B1 to B2.
        B1 and B2 are basis vectors with respect to a common basis.
        B1: (m, n)
        B2: (m, m)
        Returns: (m, n)
    """
    return invert(B2) @ B1


# Note:
# Equivalent: A' = T^-1AS
# Similar: A' = T^-1AT
def find_equivalent_transformation_matrix(A: np.ndarray, B1: np.ndarray, B2: np.ndarray, C1: np.ndarray,
                                          C2: np.ndarray) -> np.ndarray:
    """ Returns the equivalent transformation matrix A' that represents the same linear transformation as A.
        Where A is represented with respect to B1 and C1, and A' is represented with respect to B2 and C2.
        A: (m, n) transformation matrix with respect to B1 and C1
        B1: (n, n) first basis in domain
        B2: (n, n) second basis in domain
        C1: (m, m) first basis in codomain
        C2: (m, m) second basis in codomain
        Returns: (m, n)
    """
    return find_change_of_basis_matrix(C1, C2) @ A @ find_change_of_basis_matrix(B2, B1)


def find_linind_vectors(B: np.ndarray, normalize: bool = False) -> np.ndarray:
    """ Returns a linearly independent set of vectors from B.
        If normalize is True, the vectors are normalized.
        B: (m, n)
        Returns: (m, k) where k is the dimension of the linearly independent set of vectors.
    """
    ref = find_ref(B.T)
    pivot_columns, pivot_rows = find_pivot_columns(ref, return_indices=True)
    linind = (B.T[pivot_rows]).T
    if normalize:
        linind /= np.sqrt(np.sum(linind ** 2, axis=0))
    return linind


def is_linearly_independent(B: np.ndarray) -> bool:
    """ Returns True if B is linearly independent. """
    return rk(B) == B.shape[1]
