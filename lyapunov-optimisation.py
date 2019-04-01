
def _solve_discrete_lyapunov_direct(a, q):
    lhs = kron(a, a.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = solve(lhs, q.flatten())

    return np.reshape(x, q.shape)


def _solve_discrete_lyapunov_bilinear(a, q):
    eye = np.eye(a.shape[0])
    aH = a.conj().transpose()
    aHI_inv = inv(aH + eye)
    b = np.dot(aH - eye, aHI_inv)
    c = 2*np.dot(np.dot(inv(a + eye), q), aHI_inv)
    return solve_lyapunov(b.conj().transpose(), -c)


def solve_discrete_lyapunov(a, q, method=None):
    a = np.asarray(a)
    q = np.asarray(q)
    if method is None:
        # Select automatically based on size of matrices
        if a.shape[0] >= 10:
            method = 'bilinear'
        else:
            method = 'direct'

    meth = method.lower()

    if meth == 'direct':
        x = _solve_discrete_lyapunov_direct(a, q)
    elif meth == 'bilinear':
        x = _solve_discrete_lyapunov_bilinear(a, q)
    else:
        raise ValueError('Unknown solver %s' % method)

    return x
def solve_continuous_lyapunov(a, q):
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))

    r_or_c = float

    for ind, _ in enumerate((a, q)):
        if np.iscomplexobj(_):
            r_or_c = complex

        if not np.equal(*_.shape):
            raise ValueError("Matrix {} should be square.".format("aq"[ind]))

    # Shape consistency check
    if a.shape != q.shape:
        raise ValueError("Matrix a and q should have the same shape.")

    # Compute the Schur decomp form of a
    r, u = schur(a, output='real')

    # Construct f = u'*q*u
    f = u.conj().T.dot(q.dot(u))

    # Call the Sylvester equation solver
    trsyl = get_lapack_funcs('trsyl', (r, f))

    dtype_string = 'T' if r_or_c == float else 'C'
    y, scale, info = trsyl(r, r, f, tranb=dtype_string)

    y *= scale

    return u.dot(y).dot(u.conj().T)
