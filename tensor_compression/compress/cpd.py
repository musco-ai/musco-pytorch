import numpy as np


def khatrirao(matrices, skip_matrix=None, reverse=False):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).

    Parameters
    ----------
    :param matrices: ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    :param skip_matrix: None or int, optional, default is None
        if not None, index of a matrix to skip

    :param reverse: bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.

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

    if matrices[0].ndim == 1:
        matrices[0] = np.reshape(matrices[0], [1, -1])

    n_columns = matrices[0].shape[1]

    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                                 i, matrix.ndim))
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
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


def cpd_initialize(ext_sizes, rank, init_function=None):
    """
    Initialize a CPD decomposition.
    :param ext_sizes:  ndarray
        sizes of external indices
    :param rank:  int
        rank of the CPD decomposition
    :param init_function: func
        function to call for the initialization of each factor.
        Will be passed a single tuple with factor's size.
    Returns
    -------
    lam, factors: vector of factor norms and a tuple with factors
    """

    if init_function is None:
        def init_function(x): return np.random.rand(*x)

    return [init_function((size, rank))
            for size in ext_sizes]


def cpd_normalize(factors, sort=True, merge_lam=False):
    """
    Normalize the columns of factors to unit. The norms
    are returned in as ndarray

    :param factors: ndarray iterable
        factor matrices
    :param sort: bool, optional
        default True
    :param merge_lam: bool, optional, default False
        merge lam and normalized factors into
        a single tuple, as in nCPD format

    Returns
    -------
    lam: ndarray
    factors: ndarray tuple

    >>> k = cpd_initialize([2,2],3)
    >>> l, kn = cpd_normalize(k)
    >>> np.allclose(cpd_rebuild(k), ncpd_rebuild((l,) + kn))
    True

    """

    lam = np.ones([1, factors[0].shape[1]])
    new_factors = []

    for factor in factors:
        lam_factor = np.linalg.norm(factor, axis=0)
        new_factors.append(factor / lam_factor)
        lam = lam * np.reshape(lam_factor, [1, -1])

    if sort:
        order = np.argsort(lam)[::-1][0]
        lam = lam[:, order]

        for idx, factor in enumerate(new_factors):
            new_factors[idx] = factor[:, order]

    if merge_lam:
        return (lam, ) + tuple(new_factors)
    else:
        return lam, tuple(new_factors)


def ncpd_denormalize(factors, sort=True):
    """
    Convert nCPD format to the CPD format. Lambda factors are
    multiplied into factor matrices

    :param factors: ndarray iterable
        factor matrices in ncpd format
    :param sort: bool, if the factors are sorted by norm
        default True

    Returns
    -------
    lam: ndarray
    factors: ndarray tuple

    >>> k = cpd_initialize([2,2],3)
    >>> kn = cpd_normalize(k, sort=False, merge_lam=True)
    >>> kk = ncpd_denormalize(kn, sort=False)
    >>> np.allclose(cpd_rebuild(k), cpd_rebuild(kk))
    True
    >>> k = cpd_initialize([2,2],3)
    >>> kn = cpd_normalize(k, sort=True, merge_lam=True)
    >>> kk = ncpd_denormalize(kn, sort=True)
    >>> np.allclose(cpd_rebuild(k), cpd_rebuild(kk))
    True

    """

    lam = factors[0]
    new_factors = list(factors[1:])

    if sort:
        order = np.argsort(lam)[::-1][0]
        lam = lam[:, order]

        for idx, factor in enumerate(new_factors):
            new_factors[idx] = factor[:, order]

    lam_factor = np.power(lam, 1. / len(new_factors))
    for idx, factor in enumerate(new_factors):
        new_factors[idx] = np.dot(factor, np.diag(lam_factor[0, :]))

    return new_factors


def ncpd_renormalize(factors, sort=False, positive_lam=False):
    """
    Normalizes the columns of factors in nCPD format, in
    case the normalization was lost due to some operation on
    the original factors.

    :param factors: ndarray iterable
        factor matrices in ncpd format
    :param sort: bool, default False
        if the factors are sorted by norm
    :param positive_lam: bool, default False
        if negative weights lam have to be turned positive

    Returns
    -------
    lam: ndarray
    factors: ndarray tuple

    >>> k = cpd_initialize([2,2],3)
    >>> kn = cpd_normalize(k, merge_lam=True)
    >>> kk = [np.ones((1, 3)), ] + k
    >>> kt = ncpd_renormalize(kk)
    >>> np.allclose(ncpd_rebuild(kt), ncpd_rebuild(kn))
    True
    >>> kn1 = ncpd_renormalize(kn, sort=True)
    >>> np.allclose(ncpd_rebuild(kn1), ncpd_rebuild(kn))
    True
    """
    old_lam = factors[0]
    old_factors = list(factors[1:])

    lam, factors = cpd_normalize(old_factors, sort=False, merge_lam=False)

    new_lam = old_lam * lam

    if sort:
        order = np.argsort(new_lam)[::-1][0]
        new_lam = new_lam[:, order]
    if positive_lam:
        signs = np.sign(new_lam)
        factors = (factors[0] * signs, ) + factors[1:]
        new_lam = new_lam * signs

    return (new_lam, ) + factors


def ncpd_initialize(ext_sizes, rank, init_function=None):
    """
    Initialize a normalized CPD decomposition.
    :param ext_sizes:  ndarray
        sizes of external indices
    :param rank: int
        rank of the CPD decomposition
    :param init_function: func
        function to call for the initialization of each factor.
        Will be passed a single tuple with factor's size.

    Returns
    -------
    norm_and_factors: ndarray tuple

    """

    lam, factors = cpd_normalize(
        cpd_initialize(ext_sizes, rank,
                       init_function=init_function),
        sort=True)

    return (lam, ) + factors


def cpd_rebuild(factors):
    """
    Rebuild full tensor from it's CPD decomposition
    :param factors: iterable with factor matrices

    Returns
    -------
    tensor: ndarray

    >>> a = np.array([[1,3],[2,4]])
    >>> b = np.array([[5,6],[7,8],[9,10]])
    >>> c = np.array([[11,12],[13,14]])
    >>> ref = np.array([  784.,  1058.,  1332.])
    >>> np.allclose(sum(cpd_rebuild((a,b,c))[:,:,1], 1), ref)
    True
    """

    N = len(factors)
    tensor_shape = tuple(factor.shape[0] for factor in factors)
    tensor = khatrirao(tuple(factors[ii] for ii in range(
        N - 1)), reverse=False).dot(factors[N - 1].transpose())
    return tensor.reshape(tensor_shape)


def ncpd_rebuild(norm_factors):
    """
    Rebuild full tensor from it's normalized CPD decomposition
    :param norm_factors: iterable with norm and factors. First
                         iterate is the normalization vector lambda,
                         the rest are factor matrices


    >>> import numpy as np; np.random.seed(0)
    >>> k = cpd_initialize([2,2],3)
    >>> np.random.seed(0)
    >>> kn = ncpd_initialize([2,2],3)
    >>> np.allclose(cpd_rebuild(k), ncpd_rebuild(kn))
    True

    """
    lam = norm_factors[0]
    factors = norm_factors[1:]

    N = len(factors)
    tensor_shape = tuple(factor.shape[0] for factor in factors)
    t = khatrirao(tuple(factors[ii] for ii in range(
        N - 1)), reverse=False).dot((lam * factors[N - 1]).transpose())
    return t.reshape(tensor_shape)


def cpd_contract_free_cpd(factors_top, factors_bottom,
                          skip_factor=None, conjugate=False):
    """
    Contract "external" indices of two CPD decomposed tensors
    :param factors_top: iterable with CPD decomposition
                        of the top (left) tensor
    :param factors_bottom: iterable with CPD decomposition
                        of the bottom (right) tensor
    :param conjugate: conjugate the top (left) factor (default: False)
    :param skip_factor: int, default None
              skip the factor number skip_factor

    >>> import numpy as np
    >>> k1 = cpd_initialize([3,3,4], 3)
    >>> k2 = cpd_initialize([3,3,4], 3)
    >>> t1 = cpd_rebuild(k1)
    >>> t2 = cpd_rebuild(k2)
    >>> s1 = t1.conj().flatten().dot(t2.flatten()[None].T)
    >>> s2 = np.sum(cpd_contract_free_cpd(k1, k2, conjugate=True))
    >>> np.allclose(s1, s2)
    True

    """

    from functools import reduce

    if skip_factor is not None:
        factors_top = [factors_top[ii]
                       for ii in range(len(factors_top))
                       if ii != skip_factor]
        factors_bottom = [factors_bottom[ii]
                          for ii in range(len(factors_bottom))
                          if ii != skip_factor]

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    s = reduce(lambda x, y: x * y,
               map(lambda x: x[0].T.dot(x[1]),
                   zip(factors_top, factors_bottom)))

    return s


def ncpd_contract_free_ncpd(factors_top, factors_bottom,
                            skip_factor=None, conjugate=False):
    """
    Contract "external" indices of two nCPD decomposed tensors
    :param factors_top: iterable with nCPD decomposition
                        of the top(left) tensor
    :param factors_bottom: iterable with nCPD decomposition
                        of the bottom(right) tensor
    :param conjugate: conjugate the top(left) factor(default: False)
    :param skip_factor: int, default None
              skip the factor number skip_factor

    >>> k1 = cpd_initialize([3, 3, 4], 3)
    >>> kn1 = cpd_normalize(k1, sort=False, merge_lam=True)
    >>> k2 = cpd_initialize([3, 3, 4], 3)
    >>> kn2 = cpd_normalize(k2, sort=False, merge_lam=True)
    >>> s1 = cpd_contract_free_cpd(k1, k2)
    >>> s2 = ncpd_contract_free_ncpd(kn1, kn2)
    >>> np.allclose(s1, s2)
    True
    >>> s3 = ncpd_contract_free_ncpd(kn1, kn2, skip_factor=0)
    >>> np.allclose(s2, np.diag(kn1[0][0,:]) @ s3 @ np.diag(kn2[0][0,:]))
    True
    """
    if skip_factor is not None:
        if skip_factor != 0:
            skip_cpd = skip_factor - 1
        else:
            skip_cpd = None
    else:
        skip_cpd = None
    s = cpd_contract_free_cpd(factors_top[1:],
                              factors_bottom[1:],
                              skip_factor=skip_cpd,
                              conjugate=conjugate)

    lam_top = factors_top[0]
    lam_bottom = factors_bottom[0]

    if skip_factor != 0:
        return s * np.dot(lam_top.T, lam_bottom)
    else:
        return s


def cpd_symmetrize(factors, permdict, adjust_scale=True, weights=None):
    """
    Produce a symmetric CPD decomposition.
    :param factors: CPD factors
    :param permdict: dictionary of tuple: tuple pairs.
                     keys are tuples containing the permutation
                     values are tuples of any of('ident', 'neg', 'conj')
                     Identity permutation has to be excluded(added internally)
    :param weights: list, default None
                    weights of the symmetry elements.
                    If set adjust scale will turn off
    :param adjust_scale: bool, default True
                    If factors have to be scaled by the order
                    of the permutation group

    Returns
    -------
    symm_factos: ndarray list, symmetrized CPD factors

    >>> a = cpd_initialize([3, 3, 4, 4], 3)
    >>> t1 = cpd_rebuild(a)
    >>> ts = 1/4 * (t1 - t1.transpose([1, 0, 2, 3]) + np.conj(t1.transpose([1, 0, 3, 2])) - t1.transpose([0, 1, 3, 2]))
    >>> k = cpd_symmetrize(a, {(1, 0, 2, 3): ('neg', ), (1, 0, 3, 2): ('conj', ), (0, 1, 3, 2): ('neg', )})
    >>> np.allclose(ts, cpd_rebuild(k))
    True
    >>> a = cpd_initialize([3, 3, 4, 4], 3)
    >>> t1 = cpd_rebuild(a)
    >>> ts = 1/2 * (t1 + t1.transpose([1, 0, 3, 2]))
    >>> k = cpd_symmetrize(a, {(1, 0, 3, 2): ('ident', ),})
    >>> np.allclose(ts, cpd_rebuild(k))
    True
    >>> ts = 2 * t1 - t1.transpose([1, 0, 3, 2])
    >>> k = cpd_symmetrize(a, {(1, 0, 3, 2): ('neg', ),}, weights=[2, 1])
    >>> np.allclose(ts, cpd_rebuild(k))
    True

    """

    nsym = len(permdict) + 1
    nfactors = len(factors)

    if weights is not None:
        if len(weights) != nsym:
            raise ValueError('len(weights) != len(permdict)+1')
        weights = [pow(w, 1 / nfactors) for w in weights]
        adjust_scale = False
    else:
        weights = [1 for ii in range(len(factors))]

    if adjust_scale:
        scaling_factor = pow(1 / nsym, 1 / nfactors)
    else:
        scaling_factor = 1

    new_factors = []

    def ident(x):
        return x

    def neg(x):
        return -1 * x

    def conj(x):
        return np.conj(x)

    def make_scaler(weight):
        return lambda x: x * weight

    from functools import reduce

    n_factors = len(factors)

    new_factors = []
    for idx, factor in enumerate(factors):
        new_factor = factor * scaling_factor * weights[0]
        for (perm, operations), weight in zip(permdict.items(), weights[1:]):
            transforms = []
            for operation in operations:
                if operation == 'ident':
                    transforms.append(ident)
                elif operation == 'neg':
                    if n_factors % 2 == 0 and idx == 0:  # We don't want to negate
                        transforms.append(ident)        # even number of times
                    else:
                        transforms.append(neg)
                elif operation == 'conj':
                    transforms.append(conj)
                else:
                    raise ValueError(
                        'Unknown operation: {}'.format(
                            operation))
            transforms.append(make_scaler(weight))

            new_factor = np.hstack(
                (new_factor,
                 scaling_factor *
                 reduce((lambda x, y: y(x)), transforms, factors[perm[idx]])))
        new_factors.append(new_factor)

    return new_factors


def ncpd_symmetrize(norm_factors, permdict, weights=None):
    """
    Produce a symmetric nCPD decomposition.
    :param norm_factors: norm and normalized CPD factors
    :param permdict: dictionary of tuple: tuple pairs.
                     keys are tuples containing the permutation
                     values are tuples of any of('ident', 'neg', 'conj')
                     Identity permutation has to be excluded(added internally)
    :param weights: list, default None
                    weights of the symmetry elements.

    Returns
    -------
    symm_norm_factos: ndarray list, symmetrized nCPD factors

    >>> a = ncpd_initialize([3, 3, 4], 3)
    >>> t1 = ncpd_rebuild(a)
    >>> ts = 1/2 * (t1 + t1.transpose([1, 0, 2]))
    >>> k = ncpd_symmetrize(a, {(1, 0, 2): ('ident', )})
    >>> np.allclose(ts, ncpd_rebuild(k))
    True
    >>> ts = (2 * t1 - 3 * t1.transpose([1, 0, 2]))
    >>> k = ncpd_symmetrize(a, {(1, 0, 2): ('neg', )}, [2, 3])
    >>> np.allclose(ts, ncpd_rebuild(k))
    True
    """

    lam = norm_factors[0]
    factors = norm_factors[1:]

    nsym = len(permdict) + 1
    if weights is not None:
        if len(weights) != nsym:
            raise ValueError('len(weights) != len(permdict)+1')
        scaling_factor = 1
    else:
        weights = [1 for ii in range(nsym)]
        scaling_factor = 1 / nsym

    new_factors = cpd_symmetrize(factors, permdict, adjust_scale=False,
                                 weights=None)
    new_lam = scaling_factor * np.hstack(
        (lam * weight for weight in weights))

    return [new_lam, ] + new_factors


def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of `tensor`
    with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in
           ``range(0, tensor.ndim)``. If -1 is passed, then
           `tensor` is flattened

    Returns
    -------
    ndarray
        unfold_tensor of shape ``(tensor.shape[mode], -1)``
    """
    if mode > -1:
        return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))
    elif mode == -1:
        return tensor.reshape([1, -1])
    else:
        raise ValueError('Wrong mode: {}'.format(mode))


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape
        ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`

    >>> a = np.random.rand(3, 12)
    >>> b = fold(a, 0, [3, 3, 4])
    >>> b.shape == (3, 3, 4)
    True
    >>> np.allclose(fold(a.ravel(), -1, [3, 3, 4]), b)
    True
    """
    if mode > -1:
        full_shape = list(shape)
        mode_dim = full_shape.pop(mode)
        full_shape.insert(0, mode_dim)
        return np.moveaxis(unfolded_tensor.reshape(full_shape), 0, mode)
    elif mode == -1:
        return unfolded_tensor.reshape(shape)
    else:
        raise ValueError('Wrong mode: {}'.format(mode))


def als_contract_cpd(factors_top, tensor_cpd,
                     skip_factor, conjugate=False,
                     tensor_format='cpd'):
    """
    Performs the first part of the ALS step on an (already CPD decomposed)
    tensor, which is to contract "external" indices of the tensor with all
    CPD factors of the decomposition except of one, denoted by skip_factor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor_cpd: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    :param tensor_format: str, default 'cpd'
               Format of the decomposition ('cpd' or 'ncpd') for both
               tensor_cpd and factors_top
    Returns
    -------
           matrix
    """
    if tensor_format == 'cpd':
        contractor = cpd_contract_free_cpd
    elif tensor_format == 'ncpd':
        contractor = ncpd_contract_free_ncpd
    else:
        raise ValueError('Unknown tensor_format: {}'.format(tensor_format))

    s = contractor(factors_top, tensor_cpd,
                   skip_factor=skip_factor, conjugate=conjugate)
    return np.dot(tensor_cpd[skip_factor], s.T)


def als_contract_dense(factors_top, tensor,
                       skip_factor, conjugate=False,
                       tensor_format='cpd'):
    """
    Performs the first part of the ALS step on a dense
    tensor, which is to contract "external" indices of the tensor with all
    nCPD factors of the decomposition except of one, denoted by skip_factor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor: tensor to contract with
    :param skip_factor: int
              skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    :param tensor_format: str, default 'cpd'
               Format of the decomposition ('cpd' or 'ncpd')
    Returns
    -------
        matrix

    >>> kn = ncpd_initialize([3, 3, 4], 4)
    >>> k = kn[1:]
    >>> t = cpd_rebuild(k)
    >>> s1 = als_contract_dense(kn, t, skip_factor=0, tensor_format='ncpd')
    >>> s2 = np.dot(t.ravel(), khatrirao(k))
    >>> np.allclose(s1, s2)
    True
    >>> kn1 = ncpd_renormalize(kn)
    >>> s3 = als_contract_dense(kn, t, skip_factor=1, tensor_format='ncpd')
    >>> s4 = als_contract_dense(kn1, t, skip_factor=1, tensor_format='ncpd')
    >>> np.allclose(s3, s4)
    True
    """

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    if tensor_format == 'cpd':
        mode = skip_factor
    elif tensor_format == 'ncpd':
        mode = skip_factor - 1
    else:
        raise ValueError('Unknown tensor_format: {}'.format(tensor_format))

    return np.dot(unfold(tensor, mode=mode),
                  khatrirao(factors_top, skip_matrix=skip_factor))


def als_pseudo_inverse(factors_top, factors_bottom,
                       skip_factor, conjugate=False,
                       thresh=1e-10):
    """
    Calculates the pseudo inverse needed in the ALS algorithm.

    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param factors_bottom: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
              skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    :param thresh: float, default: 1e-10
               threshold used to calculate pseudo inverse
    Returns
    -------
          matrix

    >>> a = cpd_initialize([3, 3, 4], 3)
    >>> b = cpd_initialize([3, 3, 4], 4)
    >>> r = cpd_contract_free_cpd(a, b, skip_factor=2)
    >>> s = als_pseudo_inverse(a, b, skip_factor=2)
    >>> np.allclose(np.linalg.pinv(r), s)
    True
    >>> a = ncpd_initialize([3, 3, 4], 3)
    >>> b = ncpd_initialize([3, 3, 4], 4)
    >>> r = ncpd_contract_free_ncpd(a, b, skip_factor=2)
    >>> s = als_pseudo_inverse(a, b, skip_factor=2)
    >>> np.allclose(np.linalg.pinv(r), s)
    True
    """

    rank1 = factors_top[0].shape[1]
    rank2 = factors_bottom[0].shape[1]

    if conjugate:
        factors_top = [factor.conjugate() for factor in factors_top]

    pseudo_inverse = np.ones((rank1, rank2))
    for ii, (factor1, factor2) in enumerate(zip(factors_top, factors_bottom)):
        if ii != skip_factor:
            pseudo_inverse *= np.dot(factor1.T, factor2)

    return np.linalg.pinv(pseudo_inverse, thresh)


def als_step_cpd(factors_top, tensor_cpd,
                 skip_factor, conjugate=False,
                 tensor_format='cpd'):
    """
    Performs one ALS update of the factor skip_factor
    for a CPD decomposed tensor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor_cpd: iterable with CPD decomposition
               of the bottom (right) tensor
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate the top (left) factors
    :param tensor_format: str, default 'cpd'
               Format of the decomposition ('cpd' or 'ncpd')

    Returns
    -------
          matrix - updated factor skip_factor
    """

    r = als_contract_cpd(factors_top, tensor_cpd,
                         skip_factor, conjugate=conjugate,
                         tensor_format=tensor_format)
    s = als_pseudo_inverse(factors_top, factors_top,
                           skip_factor, conjugate=conjugate,
                           thresh=1e-10)
    return np.dot(r, s)


def als_step_dense(factors_top, tensor,
                   skip_factor, conjugate=False,
                   tensor_format='cpd'):
    """
    Performs one ALS update of the factor skip_factor
    for a full tensor
    :param factors_top: iterable with CPD decomposition
               of the top (left) tensor
    :param tensor: ndarray, tensor to decompose
    :param skip_factor: int
               skip the factor number skip_factor
    :param conjugate: bool, default: False
               conjugate CPD factors (needed for complex CPD)
    :param tensor_format: str, default 'cpd'
               Format of the decomposition ('cpd' or 'ncpd')
    Returns
    -------
          matrix - updated factor skip_factor
    """

    r = als_contract_dense(factors_top, tensor,
                           skip_factor, conjugate=conjugate,
                           tensor_format=tensor_format)
    s = als_pseudo_inverse(factors_top, factors_top,
                           skip_factor, conjugate=conjugate,
                           thresh=1e-10)
    return np.dot(r, s)


def als_cpd(guess, tensor_cpd, complex_cpd=False, max_cycle=100, tensor_format='cpd'):
    """
    Run an ALS algorithm on the CPD decomposed tensor
    :param guess: iterable with initial guess for CPD decomposition
    :param tensor_cpd: iterable with CPD decomposition
                       of the target tensor
    :param complex_cpd: bool, default: False
               if complex decomposition is done
               (guess should also be complex, or this will not be enforced)
    :param max_cycle: maximal number of iterations
    :param tensor_format: str, default 'cpd'
               Format of the decomposition ('cpd' or 'ncpd')

    Returns
    -------
    Iterable with CPD decomposition

    >>> a = cpd_initialize([3, 3, 4], 3)
    >>> b = cpd_initialize([3, 3, 4], 10)
    >>> k = als_cpd(b, a, max_cycle=100)
    >>> np.allclose(cpd_rebuild(a), cpd_rebuild(k), 1e-10)
    True
    >>> a = ncpd_initialize([3, 3, 4], 3)
    >>> b = ncpd_initialize([3, 3, 4], 10)
    >>> k = als_cpd(b, a, max_cycle=100, tensor_format='ncpd', complex_cpd=True)
    >>> np.allclose(ncpd_rebuild(a), ncpd_rebuild(k), 1e-9)
    True
    """
    factors = [factor for factor in guess]  # copy the guess

    for iteration in range(max_cycle):
        for idx in range(len(factors)):
            factor = als_step_cpd(factors, tensor_cpd,
                                  skip_factor=idx,
                                  conjugate=complex_cpd,
                                  tensor_format=tensor_format)
            factors[idx] = factor
        if tensor_format == 'ncpd':
            factors = list(ncpd_renormalize(
                factors, sort=False, positive_lam=True))
    return factors


def als_dense(guess, tensor, complex_cpd=False, max_cycle=100,
              tensor_format='cpd'):
    """
    Run an ALS algorithm on a dense tensor

    :param guess: iterable with initial guess for CPD decomposition
    :param tensor: ndarray, target tensor
    :param complex_cpd: bool, default: False
               if complex decomposition is done
               (guess should also be complex, or this will not be enforced)
    :param max_cycle: maximal number of iterations
    :param tensor_format: str, default 'cpd'
               Format of the decomposition ('cpd' or 'ncpd')

    Returns
    -------
    Iterable with CPD decomposition

    >>> a = cpd_rebuild(cpd_initialize([3, 3, 4], 3))
    >>> b = cpd_initialize([3, 3, 4], 10)
    >>> k = als_dense(b, a, max_cycle=100)
    >>> np.allclose(a, cpd_rebuild(k), 1e-9)
    True
    >>> ac = ncpd_initialize([3, 3, 4], 3)
    >>> a = ncpd_rebuild(ncpd_symmetrize(ac, {(1, 0, 2): ('neg', )}))
    >>> b = ncpd_initialize([3, 3, 4], 10)
    >>> k = als_dense(b, a, max_cycle=100, tensor_format='ncpd', complex_cpd=True)
    >>> np.allclose(a, ncpd_rebuild(k), 1e-9)
    True
    """
    factors = [factor for factor in guess]  # copy the guess

    for iteration in range(max_cycle):
        for idx in range(len(factors)):
            factor = als_step_dense(factors, tensor,
                                    skip_factor=idx,
                                    conjugate=complex_cpd,
                                    tensor_format=tensor_format)
            factors[idx] = factor
        if tensor_format == 'ncpd':
            # Turning off sorting is important, otherwise no convergence
            # Normalization improves convergence of lam. Also, lams may become
            # negative if not forced positive at each iteration - not sure
            # what this means
            factors = list(ncpd_renormalize(
                factors, sort=False, positive_lam=True))
    return factors


def _demonstration_symmetry_rank():  # pragma: nocover
    """
    1. This function demonstrates that a symmetrized
    tensor has a rank which is larger than the
    rank of its unsymmetric part. If we set the rank of the
    unsymmetric part to R1, the errors in the CPD decomposition
    of the symmetrized tensor with a guess having rank R1 are not small,
    thus the rank of symmetrized tensor may be larger than
    the rank of the unsymmetric part.

    2. There is no difference between decomposing full tensor
    or its CPD decomposition, although there should be
    a large gain in speed for decomposed tensors.

    3. Symmetrized factors, which produce symmetrized tensor
    exactly, preserve all their weights during nCPD ALS if
    used as an inital guess. This means that ALS procedure
    does not break symmetry.
    However, a random initial guess can have different weights
    and yield an exact decomposition. This also shows that the form
    of CPD may not be unique if factors are not independent, as is
    in the case of symmetrized factors. Otherwise CPD is unique.

    4. The minimal rank of the symmetrized tensor may be smaller than
    the rank of its CPD done with symmetrized factors as an inital guess.
    For example, a symmetrized
    CPD can be rank 6 and yield the symmetrized tensor exactly, but
    there may exist an exact CPD of rank 4 of the same tensor.
    """
    # Initialize unsymmetric cpd, unsymmetric tensor and symmetric tensor
    a = cpd_initialize([3, 3, 4], 3)
    t1 = cpd_rebuild(a)
    ts = 1 / 2 * (t1 - t1.transpose([1, 0, 2]))
    # Symmetrize cpd
    k = cpd_symmetrize(a, {(1, 0, 2): ('neg', )})
    # Check symmetrized CPD yeilds symmetrized tensor
    z0 = ts - cpd_rebuild(k)
    # Initialize unsymmetric ncpd and its symmetrized version
    an = cpd_normalize(a, sort=True, merge_lam=True)
    kn = cpd_normalize(k, sort=True, merge_lam=True)
    # Check symmetrized ncpd yeilds symmetric tensor
    z1 = ts - ncpd_rebuild(kn)
    # Calculate cpd of the same tensor in dense and CPD forms
    # with original unsymmetric factors as inital guess 
    n = als_cpd(an, kn, max_cycle=10000, tensor_format='ncpd')
    m = als_dense(an, ts, max_cycle=10000, tensor_format='ncpd')
    # Check that the rank of symmetrized tensor may be bigger
    z2 = ts - ncpd_rebuild(n)
    z3 = ts - ncpd_rebuild(m)
    # Check that ALS preserves weights
    l = als_dense(kn, ts, max_cycle=10000, tensor_format='ncpd')
    z4 = ts - ncpd_rebuild(l)
    # Check that there may exist an unsymmetric tensor with smaller
    # rank which decomposes symmetric tensor
    kp = ncpd_initialize([3, 3, 4], 5)
    o = als_dense(kp, ts, max_cycle=10000, tensor_format='ncpd')
    z5 = ts - ncpd_rebuild(o)

    print('Symm tensor - symm CPD: {}'.format(np.linalg.norm(z0.ravel())))
    print('Symm tensor - symm nCPD: {}'.format(np.linalg.norm(z1.ravel())))
    print('Symm nCPD - optimized original nCPD: {}'.format(np.linalg.norm(z2.ravel())))
    print('Symm tensor - optimized original nCPD: {}'.format(np.linalg.norm(z3.ravel())))
    print('Symm tensor - optimized symm nCPD: {}'.format(np.linalg.norm(z4.ravel())))
    print('Symm tensor - optimized nCPD of lower rank: {}'.format(np.linalg.norm(z5.ravel())))


def recompress_ncpd_tensor(tensor_ncpd, new_rank,
                           max_cycle=100, return_fit=True, tensor_format = 'ncpd'):
    """
    Given a tensor in the nCPD format and a new rank, calculate the
    best nCPD approximation to the given tensor.

    :param tensor_ncpd: iterable with the tensor in the nCPD format
    :param new_rank: int, new value of the rank
    :param max_cycle: int, default 100, number of cycles to calculate
                      the new CPD decomposition
    :param return_fit: bool, default True, if the Frobenius norm of
                      the difference tensor is returned

    >>> t = ncpd_initialize([2, 2, 2], 3)
    >>> q, fit = recompress_ncpd_tensor(t, 4, return_fit=True)
    >>> np.allclose(fit, 0)
    True
    """
    if tensor_format == 'ncpd':
        sizes = [factor.shape[0] for factor in tensor_ncpd[1:]]
    elif tensor_format == 'cpd':
        sizes = [factor.shape[0] for factor in tensor_ncpd[:]]
    
    is_complex = any([np.iscomplex(factor).any() for factor in tensor_ncpd])
    
    if tensor_format == 'ncpd':
        guess = ncpd_initialize(sizes, new_rank)
    elif tensor_format == 'cpd':
        guess = cpd_initialize(sizes, new_rank)
        
        
    new_tensor = als_cpd(
        guess, tensor_ncpd,
        complex_cpd=is_complex,
        max_cycle=max_cycle, tensor_format= tensor_format)

    result = new_tensor

    if return_fit:       
        if tensor_format == 'ncpd':
            tensor_full = ncpd_rebuild(tensor_ncpd)
            tensor_full_new = ncpd_rebuild(new_tensor)
        elif tensor_format == 'cpd':
            tensor_full = cpd_rebuild(tensor_ncpd)
            tensor_full_new = cpd_rebuild(new_tensor)
            
        fit = np.linalg.norm((tensor_full - tensor_full_new).flatten())/np.linalg.norm(tensor_full.flatten())
        

        result = (result, fit)

    return result
