import sympy as sp
from typing import cast

from chapter3 import *


def det(A: np.ndarray) -> float | sp.Expr:
    """ Returns the determinant of A. """
    assert A.shape[0] == A.shape[1], 'Matrix must be square.'
    n = A.shape[0]
    if n == 1:
        # noinspection PyTypeChecker
        return A[0, 0]
    else:
        # Laplace's expansion along row 0
        return sum((-1) ** j * A[0, j] * det(np.delete(np.delete(A, 0, 0), j, 1)) for j in range(n))


def tr(A: np.ndarray) -> float:
    """ Returns the trace of A. """
    assert A.shape[0] == A.shape[1], 'Matrix must be square.'
    return np.diag(A).sum()


def characteristic_polynomial(A: np.ndarray) -> np.polynomial.Polynomial:
    """ Returns the characteristic polynomial of A. """
    assert A.shape[0] == A.shape[1], 'Matrix must be square.'
    sympy_poly = cast(sp.Expr, det(A - sp.symbols('lambda') * np.eye(A.shape[0])))
    sympy_poly = cast(sp.Poly, sympy_poly.expand().as_poly())
    coeffs = np.array(sympy_poly.all_coeffs()[::-1], dtype=np.float64)
    return np.polynomial.Polynomial(coeffs)


def round_within(x: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """ Rounds the elements of x to the nearest integer if they are within tol of an integer. """
    return np.where(np.abs(x - np.round(x)) < tol, np.round(x), x)


def eigen_values(A: np.ndarray, rounding_tol: float | None = 1e-7) -> np.ndarray:
    """ Returns the real eigenvalues of A sorted in descending order.
    A: (m, n)
    rounding_tol: float, the tolerance for rounding eigenvalues to the nearest integer.
    Returns: (k) where k is the number of real eigenvalues of A, k <= min(m, n).
    """
    m, n = A.shape
    assert m == n, 'Matrix must be square.'
    if m >= 7:
        # Use the QR algorithm for large matrices.
        v = eigen_values_qr(A)
    else:
        # Solve the characteristic polynomial for small matrices.
        v = characteristic_polynomial(A).roots()
    # Keep only the real eigenvalues.
    if np.any(np.iscomplex(v)):
        if rounding_tol is not None:
            v.imag = round_within(v.imag, rounding_tol)
        v = v[v.imag == 0].real
    # Round to nearest integer.
    if rounding_tol is not None:
        v = round_within(v, rounding_tol)
    v = np.sort(v)[::-1]
    return v


def eigen_space(A: np.ndarray, eigenvalue: float) -> np.ndarray:
    """ Returns an orthonormal basis for the eigenspace of A associated with the eigenvalue.
    A: (m, n)
    eigenvalue: float, an eigenvalue of A.
    Returns: (m, k) where k is the geometric multiplicity of the eigenvalue.
    """
    assert A.shape[0] == A.shape[1], 'Matrix must be square.'
    return kernel_space(A - eigenvalue * np.eye(A.shape[0]))


def eigen_decomposition(A: np.ndarray, allow_partial: bool = False, orthonormalize: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns [D, P] from the factorization of A as A = PDP^(-1)
        where D is a diagonal matrix and P is a matrix whose columns
        are the eigenbasis of A.

    Input:
    A: (n, n) matrix with k eigenvectors.

    Output:
    D: (n, k) diagonal matrix of eigenvalues in descending order.
    P: (n, k) matrix whose columns are the eigenvectors of A.

    If allow_partial is True, then the function returns the eigenvalues and eigenvectors of A even if A is defective.
    If orthonormalize is True, then the eigenvectors are orthonormalized using the Gram-Schmidt process.
    """
    assert A.shape[0] == A.shape[1], 'Matrix must be square.'
    n = A.shape[0]
    eigenvalues = np.sort(np.unique(eigen_values(A)))[::-1]
    eigen_spaces = [eigen_space(A, v) for v in eigenvalues]
    geometric_multiplicities = [B.shape[1] for B in eigen_spaces]
    assert allow_partial or sum(
        geometric_multiplicities) == n, f'The matrix is defective, sum(ga) <> n: {sum(geometric_multiplicities)} <> {n}.'
    eigenvalues = np.repeat(eigenvalues, geometric_multiplicities)
    D = np.diag(eigenvalues)
    P = np.hstack(eigen_spaces)
    if orthonormalize:
        P, _ = qr_decomposition_gram_schmidt(P)
    return D, P


def singular_value_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns (U, D, V) from the factorization of A as A = UDV^T
        where U and V are orthogonal matrices and D is a diagonal matrix.

    Input:
    A: m x n matrix.

    Output (k = min m,n):
    U: m x k orthogonal matrix.
    D: k x k diagonal matrix.
    V: n x k orthogonal matrix.
    """
    m, n = A.shape
    if m >= n:
        D, V = eigen_decomposition(A.T @ A, orthonormalize=True)
        D = np.sqrt(D)
        _, U = eigen_decomposition(A @ A.T, orthonormalize=True)
    else:
        D, U = eigen_decomposition(A @ A.T, orthonormalize=True)
        D = np.sqrt(D)
        _, V = eigen_decomposition(A.T @ A, orthonormalize=True)
    return U, D, V


# Additional Functions
def pseudo_invert(A: np.ndarray) -> np.ndarray:
    """ Returns the Moore-Penrose pseudo-inverse of A.
    Input:
    A: m x n matrix.

    Output:
    A^+: n x m matrix.

    The Moore-Penrose pseudo-inverse of A is the unique matrix A^+ such that:
    1. A A^+ A = A
    2. A^+ A A^+ = A^+
    3. (A A^+)^T = A A^+

    If A is of full rank, then A^+ = A^(-1).
    """
    U, D, V = singular_value_decomposition(A)
    inv_D = np.zeros_like(D)
    non_zero_indices = D != 0
    inv_D[non_zero_indices] = 1 / D[non_zero_indices]
    return V @ inv_D @ U.T


def solve_linear_systems_with_pseudo_inverse(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the least squares general solutions to AX = B in the form (particular solutions, basis of null space).
    Input:
    A: m x n matrix.
    B: m x p matrix.

    Output:
    particular solutions: n x p matrix.
    basis of null space: n x ker(A) matrix.
    """
    particular_solutions = pseudo_invert(A) @ B
    null_space_basis = kernel_space(A)
    return particular_solutions, null_space_basis


def cholesky_factorization(A: np.ndarray) -> np.ndarray:
    """ Returns the Cholesky factorization of A, i.e. A = LL^T where L is a lower triangular matrix.
    Input:
    A: n x n symmetric positive definite matrix.

    Output:
    L: n x n lower triangular matrix.
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(j, n):
            if i == j:
                L[i, j] = np.sqrt(A[i, j] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L


def givens_rotation_to_zero(A: np.ndarray, target_idx: tuple[int, int], other_dim: int) -> np.ndarray:
    """ Returns the givens rotation matrix that rotates the target_dim to zero in the 2d-plane.
        A: (m, n) the matrix to be rotated
        target_idx: the index of the element to be rotated to zero
        other_dim: the index of the other dimension in the 2d-plane to rotate in
        Returns: (dims, dims)
    """
    R = np.eye(A.shape[1], dtype=np.float64)
    v = A[:, target_idx[1]].astype(np.float64)
    i, j = other_dim, target_idx[0]
    vi, vj = v[i], v[j]
    # Solving the below system for c and s.
    # vi c - vj s = sqrt(vi^2 + vj^2)  [preserve ||v||]
    # vi s + vj c = 0  [rotate vj to 0]
    c = vi / np.sqrt(vi ** 2 + vj ** 2)
    s = -vj / np.sqrt(vi ** 2 + vj ** 2)
    R[i, i], R[i, j] = c, -s
    R[j, i], R[j, j] = s, c
    return R


def find_similarity_transform_hessenberg(A: np.ndarray) -> np.ndarray:
    """ Returns the similarity transformation matrix of A in Hessenberg form (upper triangular with one sub-diagonal).
        A: (n, n) the matrix to be transformed
        Returns: (n, n)
    """
    m, n = A.shape
    assert m == n, 'A must be square'
    A = A.astype(np.float64)
    for j in range(n - 2):
        for i in range(j + 2, n):
            R = givens_rotation_to_zero(A, (i, j), j + 1)
            A = R @ A @ R.T
    return A


def qr_decomposition_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the QR decomposition of A using the Gram-Schmidt process.
        A: (m, n) the matrix to be decomposed
        Returns: (m, n), (n, n)
    """
    m, n = A.shape
    assert m == n, 'A must be square'
    A = A.astype(np.float64)
    Q = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = norm(v)
        if R[j, j] != 0:
            Q[:, j] = v / R[j, j]
    return Q, R


def householder_transform_to_zero(A: np.ndarray, column: int, row: int, B: np.ndarray=None) -> np.ndarray:
    """ Returns a Householder transformation matrix that zeros out the elements in the given column below the given row.
        If B is given, the transformation is directly applied to it and the resulting matrix is returned. """
    x = A[:, column].astype(np.float64).copy()
    x[:row] = 0
    # add the norm of the sub-vector to the element above
    # the sign of the element is used to determine the sign of the norm
    # this is done to avoid cancellation
    add = norm(x[row:])
    x[row] += np.sign(x[row]) * add
    if B is not None:
        if add == 0:
            # the row is already in the correct form
            return B
        # directly evaluate the transformation without creating the matrix
        # evaluates to: b - 2 * (x @ b) / (x @ x) * x
        # tile x to match the shape of B
        vectors = np.tile(2 * x, (B.shape[1], 1)).T
        # evaluate the dot product of x with each column of B
        scalars = (x[None, :] @ B) / dot_product(x, x)
        return B - vectors * scalars
    if add == 0:
        # the row is already in the correct form
        return np.eye(A.shape[0], dtype=np.float64)
    # create the transformation matrix
    Q = np.eye(A.shape[0], dtype=np.float64)
    Q -= 2 * np.outer(x, x) / dot_product(x, x)
    return Q


def qr_decomposition_householder(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the QR decomposition of the given matrix using the Householder transformation. """
    m, n = A.shape
    assert m == n, "Matrix must be square"
    # Q = H_1 @ H_2 @ ... @ H_n-1 @ I
    # R = H_n-1 @ ... @ H_1 @ A
    Q = np.eye(n, dtype=np.float64)
    # store each step to use for computing Q
    Rs = [A.astype(np.float64).copy()]
    for i in range(n-1):
        Rs.append(householder_transform_to_zero(Rs[-1], i, i, Rs[-1]))
    for i in range(n-2, -1, -1):
        Q = householder_transform_to_zero(Rs[i], i, i, Q)
    return Q, Rs[-1]


def eigen_values_qr(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """ Returns the eigenvalues of A using the QR algorithm.
        A: (n, n) the matrix to find the eigenvalues of
        max_iter: the maximum number of iterations
        tol: the tolerance for convergence
        Returns: (n,)
    """
    A = A.astype(np.float64)
    excess_frob_norm = np.inf
    for _ in range(max_iter):
        Q, R = qr_decomposition_gram_schmidt(A)
        A = R @ Q
        excess_frob_norm = np.sqrt(np.sum(np.tril(A, k=-1) ** 2))
        if excess_frob_norm < tol:
            break
    if excess_frob_norm > tol:
        print(f'QR algorithm did not converge after {max_iter} iterations.')
    v = np.diag(A)
    v = np.sort(v)[::-1]
    return v


def orthogonal_iteration(A: np.ndarray, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the eigenvalues and eigenvectors of the given symmetric matrix using the orthogonal iteration algorithm. """
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)
    for _ in range(max_iterations):
        Q = A @ Q
        Q, R = qr_decomposition_householder(Q)
    A = Q.T @ A @ Q
    return np.diag(A), Q


def qr_algorithm(A: np.ndarray, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the eigenvalues and eigenvectors of the given symmetric matrix using the QR algorithm. """
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    n = A.shape[0]
    U = np.eye(n, dtype=np.float64)
    for _ in range(max_iterations):
        Q, R = qr_decomposition_householder(A)
        A = R @ Q
        U = U @ Q
    return np.diag(A), U