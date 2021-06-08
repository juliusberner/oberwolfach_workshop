# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow import keras

from .utils import blockdiag, gen2string


# parametrizations of simple functions
SIMPLE_P = {
    "abs": [
        (np.array([[1.0, -1]]), np.array([0.0, 0.0])),
        (np.array([[1.0], [1.0]]), np.array([0.0])),
    ],
    "hat": [
        (np.array([[1.0, 1.0, 1.0]]), np.array([0.0, -0.5, -1.0])),
        (np.array([[2.0], [-4.0], [2.0]]), np.array([0.0])),
    ],
    "id_1": [
        (np.array([[1.0, -1.0]]), np.array([0.0, 0.0])),
        (np.array([[1.0], [-1.0]]), np.array([0.0])),
    ],
}


# parametrization class
class P:
    def __init__(self, name_or_parameters, name="parametrization", eps=1e-6):
        if isinstance(name_or_parameters, str):
            name = name_or_parameters
            name_or_parameters = SIMPLE_P[name_or_parameters]
        self.eps = eps
        self.name = name
        self.parameters = name_or_parameters

    @property
    def weights(self):
        return [w for w, _ in self.parameters]

    @property
    def biases(self):
        return [b for _, b in self.parameters]

    @property
    def in_dim(self):
        return self.weights[0].shape[0]

    @property
    def arch(self):
        return [self.in_dim] + [len(b) for b in self.biases]

    @property
    def out_dim(self):
        return self.arch[-1]

    @property
    def width(self):
        return max(self.arch)

    @property
    def depth(self):
        return len(self.arch) - 1

    @property
    def magnitude(self):
        return max([max(np.max(w), np.max(b)) for w, b in self.parameters])

    @property
    def num_parameters(self):
        return sum(
            [
                self.arch[i] * self.arch[i + 1] + self.arch[i + 1]
                for i in range(self.depth)
            ]
        )

    @property
    def connectivity(self):
        return sum(
            [
                (np.abs(w) > self.eps).sum() + (np.abs(b) > self.eps).sum()
                for w, b in self.parameters
            ]
        )

    def __getitem__(self, idx):
        return self.parameters[idx]

    def __setitem__(self, idx, value):
        self.parameters[idx] = value

    def __repr__(self):
        return f"{self.name}: {self.parameters}"

    def __str__(self):
        return f"{self.name}: {self.parameters}"

    def describe(self):
        return (
            f"Depth: {self.depth}, Width: {self.width}, "
            f"Magnitude: {self.magnitude}, #Parameters: {self.num_parameters}, "
            f"Connectivity: {self.connectivity}"
        )


# affine linear layer
# e.g. AffineLayer(w,b)(x)=xw+b
class AffineLayer(keras.layers.Layer):
    def __init__(self, w, b, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            initial_value=tf.constant(w, dtype="float32"), trainable=True
        )
        self.b = tf.Variable(
            initial_value=tf.constant(b, dtype="float32"), trainable=True
        )

    def call(self, x):
        return tf.matmul(x, self.w) + self.b


# realization map
class R(keras.Model):
    def __init__(self, parametrization, act=nn.relu):
        super().__init__(name=f"{act.__name__}-realization of {parametrization.name}")
        self.parametrization = parametrization
        self.act = act
        self.affine_layers = [
            AffineLayer(*p, name=f"layer_{l}")
            for l, p in enumerate(parametrization.parameters)
        ]

    def call(self, x):
        for layer in self.affine_layers[:-1]:
            x = self.act(layer(x))
        return self.affine_layers[-1](x)


# derivative map dR(Pa)(x)/dx via automatic differentiation
class D(keras.Model):
    def __init__(self, parametrization, act=nn.relu):
        super().__init__(
            name=f"{act.__name__}-derivative map of {parametrization.name}"
        )
        self.realization = R(parametrization, act=act)

    def call(self, x):
        x = tf.convert_to_tensor(x)

        with tf.GradientTape(persistent=True) as t:
            t.watch(x)
            ys = tf.unstack(self.realization(x), axis=-1)
        grads = [t.gradient(y, x) for y in ys]
        return grads[0] if len(grads) == 1 else tf.stack(grads, axis=-1)


# affine linear network
def affine(w, b=None, name=None):
    if b is None:
        b = np.zeros((w.shape[1],))
    return P([(w, b)], name="affine" if name is None else name)


# network concatenation
# e.g. pa_list = [pa1, pa2, pa3]
# R(conc(pa_list))(x)=R(pa3)(R(pa2)(R(pa1)(x)))
# the order is reversed for easier usage
def conc(pa_list, name=None):
    if name is None:
        name = " ○ ".join([pa.name for pa in pa_list[::-1]])
    if len(pa_list) == 1:  # catch exceptional cases in other functions
        return pa_list[0]
    if len(pa_list) == 2:
        pa1 = pa_list[0]
        pa2 = pa_list[1]
        w = np.matmul(pa1.weights[-1], pa2.weights[0])
        b = np.matmul(pa1.biases[-1], pa2.weights[0]) + pa2.biases[0]
        return P(pa1.parameters[:-1] + [(w, b)] + pa2.parameters[1:], name=name)
    else:
        return conc([conc(pa_list[:-1]), pa_list[-1]], name=name)


# identity network with possible efficient scaling
# e.g. dim=2, depth=4, scale = np.array([16,81])
# then R(Identity(dim, depth, scale))(x)=(16x_1,81x_2)
# efficient: coefficients with magnitude |scale_i|**(1/depth)
def identity(dim=1, depth=1, scale=1, name=None):
    id = np.eye(dim)
    name = f"{scale}*id_{dim}" if name is None else name
    if depth == 1:
        return affine(scale * id, name=name)
    elif depth > 1:
        factor = np.abs(scale) ** (1 / depth)
        pa_list = [
            _pos_elong(affine(m), depth, factor=factor)
            for m in [factor * id, -factor * id]
        ]
        ind = np.arange(dim)
        return affine_combination(
            [np.sign(scale) * id, -np.sign(scale) * id],
            pa_list=pa_list,
            ind_list=[ind, ind],
            name=name,
        )
    else:
        raise ValueError("Depth must be a natural number.")


# positive elongation (with factor) helper function
def _pos_elong(pa, depth, factor=1):
    pa_pos = identity(pa.out_dim, 1, scale=factor)
    return P(pa.parameters + [pa_pos.parameters[0] for _ in range(depth - pa.depth)])


# network elongation
def elongation(pa, depth, name=None):
    if pa.depth == depth:
        return pa
    else:
        return conc(
            [pa, identity(pa.out_dim, depth - pa.depth + 1)],
            name=pa.name if name is None else name,
        )


# parallelization helper function for same depth
def _par_same(pa_list):
    if len(pa_list) == 2:
        return P(
            [
                (blockdiag([w1, w2]), np.block([b1, b2]))
                for (w1, b1), (w2, b2) in zip(
                    pa_list[0].parameters, pa_list[1].parameters
                )
            ]
        )
    else:
        return _par_same([_par_same(pa_list[:-1]), pa_list[-1]])


# parallelization with indexed input
# e.g. pa_list = [p1,p2], ind_list = [(2,0),(1,3)]
# then: R(par(pa_list, ind_list))(x_0,x_1,x_2,x_3)=(R(p1)(x_2,x_0),R(p2)(x_1,x_3))
def par(pa_list, ind_list=None, in_dim=None, name=None):
    depth = max([pa.depth for pa in pa_list])
    pa = _par_same([elongation(pa, depth) for pa in pa_list])
    if ind_list is None:
        dims = [pa.in_dim for pa in pa_list]
        ind_list = np.array_split(np.arange(sum(dims)), np.cumsum(dims)[:-1])
    else:
        in_dim = max(max(t) for t in ind_list) + 1 if in_dim is None else in_dim
        perms = [np.zeros((in_dim, pa.in_dim)) for pa in pa_list]
        for perm, ind in zip(perms, ind_list):
            perm[ind, np.arange(len(ind))] = 1
        pa_perms = affine(np.block([perms]))
        pa = conc([pa_perms, pa], name=f"[{', '.join([pa.name for pa in pa_list])}]")
    _name = gen2string(
        [
            n + gen2string((f"x_{i}" for i in ind))
            for n, ind in zip([pa.name for pa in pa_list], ind_list)
        ]
    )
    pa.name = f"(x → {_name})" if name is None else name
    return pa


# sparse network concatenation
def sparse_conc(pa_list, name=None):
    if len(pa_list) == 2:
        id = identity(pa_list[0].out_dim, 2)
        return conc([pa_list[0], id, pa_list[1]], name=name)
    else:
        return sparse_conc([sparse_conc(pa_list[:-1]), pa_list[-1]], name=name)


# (affine) linear combination
# e.g. pa_list = [p1,p2], coeff_list = [a,b]
# then: R(affine(pa_list, coeff_list))(x)=a*R(p1)(x)+b*R(p2)(x))
# see parallelization for explanation of ind_list
def affine_combination(
    coeff_list, constant=None, pa_list=None, ind_list=None, name=None
):
    block = np.block([coeff_list]).transpose()
    pa_comb = affine(
        block,
        constant,
        name=f"(x → <x,{coeff_list}>" + (")" if constant is None else f"+{constant})"),
    )
    if pa_list is None:
        return pa_comb
    else:
        return conc([par(pa_list, ind_list=ind_list), pa_comb], name=name)


# squaring helper function (interpolates the squaring function up to precision 4**(-k))
def _square(k):
    if isinstance(k, int):
        if k <= 1:
            return identity()
        elif k > 1:
            pa_triang = P("hat")
            pa_inp = _pos_elong(identity(), 2)
            pa_sub_list = [
                _pos_elong(affine_combination([-(2.0 ** (-2 * m)), 1]), 2)
                for m in range(1, k - 1)
            ]
            pa = [par([pa_triang, pa_inp], ind_list=[[0], [0]])]
            pa += [
                par([pa_triang, pa_sub], ind_list=[[0], [0, 1]])
                for pa_sub in pa_sub_list
            ]
            return conc(pa + [affine_combination([-(2.0 ** (-2 * (k - 1))), 1])])

    else:
        raise ValueError("k must be an integer.")


# approximation of squaring function on [-bound,bound]
# up to error eps in Sobolev W^{1,\infty} norm
def square(eps, bound=1.0):
  k = int(np.ceil(2 * np.log2(bound / eps) + 1))
  depth = int(np.ceil(np.log2(bound))) if bound > 1 else 1
  return conc(
    [
      affine(np.array([[1 / bound]])),
      P("abs"),
      _square(k),
      identity(1, depth, scale=bound ** 2),
    ],
    name="squaring",
  )


# approximation of multiplication function on [-bound,bound]
# up to error eps in Sobolev W^{1,\infty} norm
def mult(eps, bound=1.0):
    pa_list = [
        conc([affine_combination([1.0, a]), square(2 * eps, bound=2 * bound)])
        for a in [1.0, -1.0]
    ]
    return affine_combination(
        [0.25, -0.25], pa_list=pa_list, ind_list=[[0, 1], [0, 1]], name="multiplication"
    )


# approximation of monomials function (x,x^2,...,x^(2^k)) on [-bound,bound]
# up to error eps in Sobolev W^{1,\infty} norm
def _monomial(power, eps, bound=1.0):
    pa_list_all = []
    eta = eps / (4.0 ** (power ** 2) * bound ** (2 ** (power + 1)))
    eta_list = [4 ** (k ** 2) * eta for k in range(1, power + 1)]
    new_bound = 2 * bound ** (2 ** power)
    for k in range(power):
        pa_list = []
        ind_list = []
        pa_square = square(eta_list[k], bound=new_bound)
        pa_mult = mult(eta_list[k], bound=new_bound)
        for i in range(2 ** (k + 1)):
            if i <= 2 ** k - 1:
                pa_list.append(identity())
                ind_list.append([i])
            elif i % 2:
                pa_list.append(pa_square)
                ind_list.append([(i - 1) // 2])
            else:
                pa_list.append(pa_mult)
                ind_list.append([(i - 2) // 2, i // 2])
        pa_list_all.append(par(pa_list, ind_list=ind_list))
    return conc(pa_list_all)


# monomial extension: arbitrary degree deg, optional with constant 1
def monomial(deg, eps, bound=1.0, const=True):
    power = int(np.ceil(np.log2(deg)))
    if const:
        e = np.zeros((deg + 1,))
        e[0] = 1
        pa_zero = P(
            [(np.block([np.zeros((2 ** power, 1)), np.eye(2 ** power, deg)]), e)]
        )
    else:
        pa_zero = affine(np.eye(2 ** power, deg))
    return conc([_monomial(power, eps, bound), pa_zero], name="monomials")


# approximation of polynomial p[0]*x^(N-1)+ ... +p[N-1] on [-bound,bound]
# up to error eps in Sobolev W^{1,\infty} norm
def poly(p, eps, bound=1.0):
    p_len = len(p)
    coeff_sum = np.sum(np.abs(p))
    eta = eps / coeff_sum
    depth = int(np.ceil(np.log2(coeff_sum)))
    pa_scale = identity(p_len, depth, scale=p[::-1])
    return conc(
        [
            monomial(p_len - 1, eta, bound),
            pa_scale,
            affine_combination(np.ones((1, p_len))),
        ],
        name="polynomial",
    )