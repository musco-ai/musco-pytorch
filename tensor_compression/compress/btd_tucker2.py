"""
This module implements BTD-2 decomposition
(or (L, M, *)-block term decomposition) for order-3 tensors as
described in

"DECOMPOSITIONS OF A HIGHER-ORDER TENSOR IN BLOCK TERMSâ€”PART III:
 ALTERNATING LEAST SQUARES ALGORITHMSâˆ—"
by LIEVEN DE LATHAUWERâ€  AND DIMITRI NIONâ€¡

The BTD-2 decomposition for order-3 tensors is defined as:
  T_ijk = sum_{r=1}^{R} sum_{l=1}^{L} sum_{m=1}^{M}
  S^{r}_{lmk} A^{r}_{il} B^{r}_{jm}

It is also a Tucker-2 decomposition
"""

import torch as th


def btd2_rebuild(Ss, As, Bs):
    """
    Rebuilds BTD-2 from a lists of terms

    T_ijk = sum_{r=1}^{R} sum_{l=1}^{L} sum_{m=1}^{M}
    S^{r}_{lmk} A^{r}_{il} B^{r}_{jm}

    Parameters
    ----------
    block_terms : list of lists
         Each block term is a list of a core and factor matrices
         [core, factor1, factor2]

    Returns
    -------
    torch.tensor
         Full tensor rebuilt from block_terms
    """
    assert len(Ss) == len(As)
    assert len(As) == len(Bs)

    ni = As[0].shape[0]
    nj = Bs[0].shape[0]
    nk = Ss[0].shape[2]
    dtype = Ss[0].dtype

    assert all(A.shape[0] == ni for A in As)
    assert all(B.shape[0] == nj for B in Bs)
    assert all(S.shape[2] == nk for S in Ss)

    T = th.zeros([ni, nj, nk], dtype=dtype)
    for block_term in zip(Ss, As, Bs):
        T += th.einsum('lmk,il,jm->ijk', block_term)

    return T


def btd2_init_random(size, ranks, rnd_function=th.randn):
    """
    Initialize Tucker-2 decomposition randomly

    Parameters
    ----------
    size : iterable of size 3
           Size of the full tensor
    ranks : iterable
           List of ranks used in the (L, M, *) decomposition.
           The length of this list is the number of terms. Each
           entry has to have length 2.
    rnd_function : func, default th.randn
           Function to draw entries from

    Returns
    -------
    (Ss, As, Bs) : lists of tensors
         Each list contains factor matrices
    """
    assert len(size) == 3
    assert all(len(rr) == 2 for rr in ranks)

    ni, nj, nk = size

    block_terms = []
    for rr in ranks:
        L, M = rr
        S = rnd_function(L, M, nk)
        A = rnd_function(ni, L)
        B = rnd_function(nj, M)
        block_terms.append([S, A, B])

    return zip(*block_terms)


def btd2_normalize(Ss, As, Bs):
    """
    Normalize Tucker-2 decomposition

    Parameters
    ----------
    Ss, As, Bs : lists of tensors
         Each list contains factor matrices

    Returns
    -------
    Ss_new, As_new, Bs_new : lists of tensors
         Each list contains transformed factor matrices
    """
    block_terms_new = []
    for block_term in zip(Ss, As, Bs):
        S, A, B = block_term
        A_orth, Ra = th.qr(A)
        B_orth, Rb = th.qr(B)
        S_new = th.einsum('lmk,il,jm->ijk', [S, Ra, Rb])
        block_terms_new.append([S_new, A_orth, B_orth])

    return zip(*block_terms_new)


def grouped(iterable, n):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    Taken from: https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    Returns only *complete* tuples, e.g. if len(s) // n == m
    then there will be only m tuples.
    """
    return zip(*[iter(iterable)]*n)


def matrix_kronecker(matrices):
    """
    Kronecker product for matrices
    """
    assert all(len(m.shape) == 2 for m in matrices)

    start = ord('a')
    num_idx = len(matrices)*2
    source = ','.join(chr(i) + chr(j)
                      for i, j in grouped(
                              range(start, start + num_idx), 2))
    dest = (''.join(chr(i) for i in range(start, start + num_idx, 2)) +
            ''.join(chr(j) for j in range(start + 1, start + num_idx, 2)))
    expr = source + '->' + dest
    res = th.einsum(expr, matrices).view(
        th.tensor([m.shape[0] for m in matrices]).prod(),
        th.tensor([m.shape[1] for m in matrices]).prod()
    )
    return res


def khatrirao_block_torch(*partitioned_matrices):
    """
    Implements the general Khatri-Rao product. Let's
    A = (a_0, a_1, ..., a_N) and B = (b_0, b_1, ..., b_N)
    be column wise partitioned matrices, then
    khatrirao_block(A, B) = (a_0 x b_0, a_1 x b_1, ..., a_N x b_N)
    where 'x' is a Kronecker product

    Parameters
    ----------
    partitioned_matrices: list of lists
          Each sublist should be a list of matrix partitions, e.g.
          [[a_0, a_1, ...], [b_0, b_1, ...], ...]
    Returns
    -------
    th.tensor
          Matrix which is a result of the operation
    """
    res = []

    for nth_parts in zip(*partitioned_matrices):
        res.append(matrix_kronecker(list(nth_parts)))

    return th.cat(res, 1)


def btd2_als_a(T, Ss, Bs):
    """
    Calculate update of the A matrices in the Tucker-2 decomposition
    """
    M = []
    lr = []
    for S, B in zip(Ss, Bs):
        lr.append(S.shape[0])
        M.append(th.einsum('lmk,jm->ljk', [S, B]))

    _, nj, nk = M[0].shape
    M = th.pinverse(th.cat(M, 0).reshape(-1, nj * nk))  # order [nj*nk, lr*r]

    AF = T.reshape(-1, nj * nk) @ M  # order [ni, lr*r]

    As = []
    for A in th.split(AF, lr, 1):
        A_orth, Ra = th.qr(A)
        As.append(A_orth)
    return As


def btd2_als_b(T, Ss, As):
    """
    Calculate update of the B matrices in the Tucker-2 decomposition
    """
    M = []
    mr = []
    for S, A in zip(Ss, As):
        mr.append(S.shape[1])
        M.append(th.einsum('lmk,il->mik', [S, A]))

    _, ni, nk = M[0].shape
    M = th.pinverse(th.cat(M, 0).reshape(-1, ni * nk))  # order [ni*nk, mr*r]

    BF = T.transpose(1, 2).reshape(ni * nk, -1).t() @ M  # order [nj, mr*r]

    Bs = []
    for B in th.split(BF, mr, 1):
        B_orth, Rb = th.qr(B)
        Bs.append(B_orth)
    return Bs


def btd2_als_s(T, As, Bs):
    """
    Calculate update of the S cores in the Tucker-2 decomposition
    """
    ni, nj, nk = T.shape

    l_m = []
    l_n = []
    l_mn = []
    for A, B in zip(As, Bs):
        l_m.append(A.shape[1])
        l_n.append(B.shape[1])
        l_mn.append(A.shape[1] * B.shape[1])

    M = th.pinverse(
        khatrirao_block_torch(As, Bs)
    )

    Ss = th.split(M @ T.reshape(-1, nk), l_mn, dim=0)
    Ss = [S.reshape([m, n, nk]) for S, m, n in zip(Ss, l_m, l_n)]
    return Ss


def test_btd2_rebuild():
    """
    Tests Tucker-2 rebuild function
    """
    import cpd
    alpha, beta, kappa = cpd.cpd_initialize([2, 2, 2], 3)
    t = th.tensor(cpd.cpd_rebuild([alpha, beta, kappa]))

    As = [th.tensor(col[:, None]) for col in alpha.T]
    Bs = [th.tensor(col[:, None]) for col in beta.T]
    Ss = [th.tensor(col[None, None, :]) for col in kappa.T]

    tt = btd2_rebuild(Ss, As, Bs)

    print(th.max(t - tt))


def test_btd2_als_steps():
    """
    Tests Tucker-2 ALS steps
    """
    shape = [4, 4, 4]
    ranks = [(1, 1), (1, 1)]

    Ss, As, Bs = btd2_init_random(shape, ranks)
    Ss, As, Bs = btd2_normalize(Ss, As, Bs)
    t = btd2_rebuild(Ss, As, Bs)

    # Test step over A factors
    As_new = btd2_als_a(t, Ss, Bs)

    t2 = btd2_rebuild(Ss, As_new, Bs)
    print(th.max(t - t2))

    # Test step over B factors
    Bs_new = btd2_als_b(t, Ss, As)

    t3 = btd2_rebuild(Ss, As, Bs_new)
    print(th.max(t - t3))

    # Test step over S factors
    Ss_new = btd2_als_s(t, As, Bs)

    t4 = btd2_rebuild(Ss_new, As, Bs)
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


def btd2_als(T, guess, niter=100, rel_err_tol=1e-8, return_fit=False):
    """
    Compute BTD-1 decomposition by ALS iterations

    Parameters
    ----------
    T : torch.tensor
        Order-3 tensor to decompose
    guess : list of lists of tensors
        List of lists, each containing S, A, B as returned by
        btd2_init_random
    niter : int, default 100
        Maximal number of iterations
    rel_tol_err : float, default 1e-8
        Maximal relative error in the factors of the BTD-1 decomposition
    return_fit : bool, default False
        If a relative error in the recomnstructed tensor
        is returned in addition to the decomposition

    Returns
    -------
    block_terms : list of lists of tensors
         list of factors [S, A, B] in the BTD-2 decomposition
    """
    assert len(T.shape) == 3

    rel_err = th.norm(T.flatten())

    Ss, As, Bs = guess
    for ii in range(niter):
        # Update A
        As_new = btd2_als_a(T, Ss, Bs)
        err_a = max(rel_difference(A, A_new) for A, A_new in
                    zip(As, As_new))
        rel_err = max([rel_err, err_a])
        As = As_new

        # Update B
        Bs_new = btd2_als_b(T, Ss, As)
        err_b = max(rel_difference(B, B_new) for B, B_new in
                    zip(Bs, Bs_new))
        rel_err = max([rel_err, err_b])
        Bs = Bs_new

        # Update S
        Ss_new = btd2_als_s(T, As, Bs)
        err_s = max(rel_difference(S, S_new) for S, S_new in
                    zip(Ss, Ss_new))
        rel_err = max([rel_err, err_s])
        Ss = Ss_new

        if rel_err < rel_err_tol:
            break

    result = [Ss, As, Bs]
    if return_fit:
        fit = rel_difference(T, btd2_rebuild(Ss, As, Bs))
        result.append(fit)

    return result


def test_btd2_als():
    shape = [4, 4, 4]
    ranks = [[1, 1], [1, 1]]
    print(f"Testing for:\n T.shape == {shape}, ranks == {ranks}")
    Ss1, As1, Bs1 = btd2_init_random(shape, ranks)
    Ss1, As1, Bs1 = btd2_normalize(Ss1, As1, Bs1)

    T = btd2_rebuild(Ss1, As1, Bs1)

    guess = btd2_init_random(shape, ranks)
    Ss2, As2, Bs2, fit = btd2_als(T, guess, niter=300, return_fit=True)

    # Calculate relative difference
    err_a = max(rel_difference(A1, A2) for A1, A2 in
                zip(As1, As2))

    err_b = max(rel_difference(B1, B2) for B1, B2 in
                zip(Bs1, Bs2))

    err_s = max(rel_difference(S1, S2) for S1, S2 in
                zip(Ss1, Ss2))

    print(
        f"Relative errors:\n A: {err_a}\n B: {err_b}\n"
        f" C: {err_s}\n total tensor: {fit}"
    )
