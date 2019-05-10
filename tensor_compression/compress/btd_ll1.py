"""
This module implements BTD-1 or (L_r, L_r, 1) decomposition as described in

"DECOMPOSITIONS OF A HIGHER-ORDER TENSOR IN BLOCK TERMSâ€”PART III:
 ALTERNATING LEAST SQUARES ALGORITHMSâˆ—"
by LIEVEN DE LATHAUWERâ€  AND DIMITRI NIONâ€¡

In short, for order-3 tensors BTD-1 is defined as:
  T_ijk = sum_{r=1}^{R} sum_{l_r=1}^{L_r} A_{i}^{l_r} B_{j}^{l_r} C_{k}^{r}

It reduces to CPD if all ranks L_r equal 1 or to Tucker decomposition if the
length of C equals 1 and there is only 1 term in the sum.
"""

import torch as th


def khatrirao_torch(matrices, skip_matrix=None, reverse=False):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).

    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip

    reverse: bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices
        in the product.

    >>> import numpy as np
    >>> a = np.array([[1.,3.],[2.,4.]])
    >>> b = np.array([[5,6],[7,8],[9,10]])
    >>> np.sum(khatrirao((a,b), reverse=True), 1)[2]
    31.0
    """
    matrices = list(matrices)

    if skip_matrix is not None:
        if skip_matrix > -1:
            matrices = [matrices[i]
                        for i in range(len(matrices)) if i != skip_matrix]
        else:
            raise ValueError('Wrong skip_matrix: {}'.format(skip_matrix))

    if len(matrices[0].shape) == 1:
        matrices[0] = th.reshape(matrices[0], [1, -1])

    n_columns = matrices[0].shape[1]

    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if len(matrix.shape) != 2:
            raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                                 i, len(matrix.shape)))
        if matrix.shape[1] != n_columns:
            raise ValueError('All matrices must have same number of columns!'
                             'Matrix {} has {} columns != {}.'.format(
                                 i, matrix.shape[1], n_columns))

    n_factors = len(matrices)

    if reverse:
        matrices = matrices[::-1]
        # Note: we do NOT use .reverse() which would reverse matrices
        # even outside this function

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    return th.einsum(operation, matrices).reshape((-1, n_columns))


def btd1_rebuild(As, Bs, C):
    """
    Rebuilds BTD from factors in As, Bs and C

    T_ijk = sum_{r=1}^{R} sum_{l_r=1}^{L_r} A_{i}^{l_r} B_{j}^{l_r} C_{k}^{r}
    Parameters
    ----------
    As : array list
        Factors A are packed as A = [A_1, A_2, ..., A_R]
        where A_r is the r-th factor matrix in the decomposition
    Bs : array list
        Factors B are packed as B = [B_1, B_2, ..., B_R]
        where B_r is the r-th factor matrix in the decomposition
    C : array
        Factors C are packed as C = [c_1, c_2, ..., c_R]
        where c_r is the r-th vector in the decomposition
    """
    dtype = As[0].dtype
    r = len(As)
    assert len(Bs) == r
    assert C.shape[1] == r

    ni = As[0].shape[0]
    assert all([A.shape[0] == ni for A in As])

    nj = Bs[0].shape[0]
    assert all([B.shape[0] == nj for B in Bs])

    nk = C.shape[0]

    # T = th.zeros([nj * nk, ni], dtype=dtype)
    # for B, c, A in zip(Bs, C.t(), As):
    #     lr = B.shape[1]
    #     # order of T is now [nj * nk, ni]
    #     T += khatrirao_torch(
    #         [B, c[:, None].expand(nk, lr)]) @ A.t()

    # return T.t().reshape([ni, nj, nk])

    # T = th.zeros([nk * ni, nj], dtype=dtype)
    # for c, A, B in zip(C.t(), As, Bs):
    #     lr = A.shape[1]
    #     # order of T is now [nk * ni, nj]
    #     T += khatrirao_torch(
    #         [c[:, None].expand(nk, lr), A]) @ B.t()

    # return T.reshape([nk, ni, nj]).transpose(0, 1).transpose(1, 2)

    T = th.zeros([ni, nj, nk], dtype=dtype)
    for A, B, c in zip(As, Bs, C.t()):
        lr = A.shape[1]
        # order of T is now [ni * nj, nk]
        # T += khatrirao_torch([A, B]) @ c[None, :].expand(lr, nk)
        # T += khatrirao_torch([A, B]) @ th.ones(lr, 1, dtype=dtype) @ c[None, :]
        T += th.einsum('il,jl,k->ijk', [A, B, c])

    return T.reshape([ni, nj, nk])


def test_btd1_rebuild():
    """
    Tests BTD-1 reconstruction
    """
    import cpd
    alpha, beta, kappa = cpd.cpd_initialize([2, 2, 2], 3)
    t = th.tensor(cpd.cpd_rebuild([alpha, beta, kappa]))

    As = th.tensor([col[:, None] for col in alpha.T])
    Bs = th.tensor([col[:, None] for col in beta.T])
    C = th.tensor(kappa)

    t1 = btd1_rebuild(As, Bs, C)
    print(th.max(t - t1))


def btd1_als_a(T, Bs, C):
    """
    Update A matrices in the BTD-1 decomposition
    """
    nj = Bs[0].shape[0]
    nk = C.shape[0]

    N = []
    ranks = []
    for B, c in zip(Bs, C.t()):
        lr = B.shape[1]
        ranks.append(lr)
        N.append(khatrirao_torch([B, c[:, None].expand(nk, lr)]))

    AF = (T.reshape([-1, nj * nk]) @ th.pinverse(th.cat(N, 1)).t())
    return th.split(AF, ranks, 1)


def btd1_als_b(T, As, C):
    """
    Update B matrices in the BTD-1 decomposition
    """
    ni = As[0].shape[0]
    nk = C.shape[0]

    # order [nj, ni, nk]
    TT = th.transpose(T, 0, 1)
    N = []
    ranks = []
    for A, c in zip(As, C.t()):
        lr = A.shape[1]
        ranks.append(lr)
        N.append(khatrirao_torch([A, c[:, None].expand(nk, lr)]))

    BF = TT.reshape([-1, ni * nk]) @ th.pinverse(th.cat(N, 1)).t()
    Bs = []
    for B in th.split(BF, ranks, 1):
        B_orth, R = th.qr(B)
        # B_orth = B
        Bs.append(B_orth)
    return Bs


def btd1_als_c(T, As, Bs):
    """
    Update C matrix in the BTD-1 decomposition
    """
    dtype = As[0].dtype
    ni = As[0].shape[0]
    nj = Bs[0].shape[0]

    N = []
    for A, B in zip(As, Bs):
        lr = A.shape[1]
        N.append(khatrirao_torch([A, B]) @ th.ones(lr, 1, dtype=dtype))

    M = th.pinverse(th.cat(N, 1))
    C = (M @ T.reshape([ni * nj, -1])).t()
    return C


def btd1_init_random(size, ranks, rnd_function=th.randn):
    """
    Initialize BTD-1 factors randomly
    Parameters
    ----------
    size : iterable of size 3
           Size of the full tensor
    ranks : iterable
           List of ranks used in the (L, L, 1) decomposition.
           The length of this list is the number of terms
    rnd_function : func, default th.randn
           Function to draw entries from
    """
    assert len(size) == 3

    ni, nj, nk = size

    As = []
    Bs = []
    Cs = []
    for lr in ranks:
        As.append(rnd_function(ni, lr))
        Bs.append(rnd_function(nj, lr))
        Cs.append(rnd_function(nk, 1))
    return As, Bs, th.cat(Cs, 1)


def btd1_normalize(As, Bs, C, mode='reference'):
    """
    Normalize the matrices in BTD according to different conventions
    """
    modes = {'reference', 'weight-c'}
    assert mode in modes

    As_new = []
    Bs_new = []

    if mode == 'reference':
        cnorms = th.norm(C, 2, 0)
        C_new = C / cnorms

        for A, B, cn in zip(As, Bs, cnorms):
            B_orth, R = th.qr(B)
            Bs_new.append(B_orth)
            As_new.append(A @ R.t() * cn)

    elif mode == 'weight-c':

        cnorms = []
        for A, B, in zip(As, Bs):
            B_orth, R = th.qr(B)
            Bs_new.append(B_orth)
            AA = A @ R.t()
            cnorms.append(th.norm(AA, 2))
            As_new.append(A @ R.t())
        C_new = C * th.tensor(cnorms)[None, :]

    return As_new, Bs_new, C_new


def test_btd1_als_steps():
    As, Bs, C = btd1_init_random([4, 4, 4], [1, 1])

    As, Bs, C = btd1_normalize(As, Bs, C)
    t = btd1_rebuild(As, Bs, C)
    AAs = btd1_als_a(t, Bs, C)

    t2 = btd1_rebuild(AAs, Bs, C)
    print(th.max(t - t2))

    # This will produce zero only if
    # Bs were orthogonal matrices
    BBs = btd1_als_b(t, As, C)

    t3 = btd1_rebuild(As, BBs, C)
    print(th.max(t - t3))

    CC = btd1_als_c(t, As, Bs)

    t4 = btd1_rebuild(As, Bs, CC)
    print(th.max(t - t4))


def rel_difference(T_old, T_new):
    """
    Calculates relative difference between
    tensors T_old and T_new.

    nu = || T_new - T_old ||_F / || T_old ||_F
    """
    nu = th.norm(
        T_new.flatten() - T_old.flatten(), 2, 0
    ) / th.norm(T_old.flatten(), 2, 0)
    return nu


def btd1_als(T, guess, niter=100, rel_err_tol=1e-8, return_fit=False):
    """
    Compute BTD-1 decomposition by ALS iterations

    Parameters
    ----------
    T : torch.tensor
        Order-3 tensor to decompose
    guess : iterable of tensors
        List of tensors containing As, Bs, C as returned by
        btd1_init_random
    niter : int, default 100
        Maximal number of iterations
    rel_tol_err : float, default 1e-8
        Maximal relative error in the factors of the BTD-1 decomposition
    return_fit : bool, default False
        If a relative error in the recomnstructed tensor
        is returned in addition to the decomposition

    Returns
    -------
    As : list of tensors
         A factors in the BTD-1 decomposition
    Bs : list of tensors
         B factors in the BTD-1 decomposition
    C  : tensor
         C factor in the BTD-1 decomposition
    fit: float, optional if return_fit == True
         fit of the computed decomposition
    """
    assert len(T.shape) == 3

    As, Bs, C = guess
    rel_err = th.norm(C.flatten())

    for ii in range(niter):
        # Update A
        As_new = btd1_als_a(T, Bs, C)
        err_a = max(rel_difference(A, A_new) for A, A_new in
                    zip(As, As_new))
        rel_err = max([rel_err, err_a])
        As = As_new

        # Update B
        Bs_new = btd1_als_b(T, As, C)
        err_b = max(rel_difference(B, B_new) for B, B_new in
                    zip(Bs, Bs_new))
        rel_err = max([rel_err, err_b])
        Bs = Bs_new

        # Update C
        C_new = btd1_als_c(T, As, Bs)
        rel_err = max([rel_err, rel_difference(C, C_new)])
        C = C_new

        if rel_err < rel_err_tol:
            break

    result = [As, Bs, C]
    if return_fit:
        fit = rel_difference(T, btd1_rebuild(As, Bs, C))
        result.append(fit)

    return result


def test_btd1_als():
    shape = [4, 4, 4]
    ranks = [1, 1]
    print(f"Testing for:\n T.shape == {shape}, ranks == {ranks}")
    As1, Bs1, C1 = btd1_init_random(shape, ranks)
    As1, Bs1, C1 = btd1_normalize(As1, Bs1, C1)

    T = btd1_rebuild(As1, Bs1, C1)

    guess = btd1_init_random(shape, ranks)
    As2, Bs2, C2, fit = btd1_als(T, guess, niter=300, return_fit=True)

    # Calculate relative difference
    err_a = max(rel_difference(A1, A2) for A1, A2 in
                zip(As1, As2))

    err_b = max(rel_difference(B1, B2) for B1, B2 in
                zip(Bs1, Bs2))

    err_c = rel_difference(C1, C2)

    print(
        f"Relative errors:\n A: {err_a}\n B: {err_b}\n"
        f" C: {err_c}\n total tensor: {fit}"
    )
