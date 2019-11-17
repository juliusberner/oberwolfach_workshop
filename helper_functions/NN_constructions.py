# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, 
                        print_function, unicode_literals)

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import nn

# Parametrization class
class P():

  def __init__(self, name_or_parameters, name='parametrization', eps=1e-6):
    if isinstance(name_or_parameters, str):
      name = name_or_parameters
      name_or_parameters = SimpleParametrizations[name_or_parameters]
    self.eps = eps
    self.name = name
    self.parameters = name_or_parameters

  @property
  def weights(self):
    return [W for W, _ in self.parameters]
  @property
  def biases(self):
    return [B for _, B in self.parameters]
  @property
  def in_dim(self):
    return self.weights[0].shape[0]
  @property
  def arch(self):
    return [self.in_dim]+[len(B) for B in self.biases]
  @property
  def out_dim(self):
    return self.arch[-1]
  @property
  def width(self):
    return max(self.arch)
  @property
  def depth(self):
    return len(self.arch)-1
  @property
  def pa_max(self):
    return max([max(np.max(W),np.max(B)) for W, B in self.parameters])
  @property
  def num_parameters(self):
    return sum([self.arch[i]*self.arch[i+1]+self.arch[i+1] 
                             for i in range(self.depth)])
  @property
  def connectivity(self):
    return sum([(np.abs(W)>self.eps).sum()+(np.abs(B)>self.eps).sum() 
                            for W, B in self.parameters])
    
  def __getitem__(self, idx):
      return self.parameters[idx]

  def __setitem__(self, idx, value):
      self.parameters[idx] = value

  def __repr__(self):
    return str(self.parameters)
  
  def __str__(self):
    return str(self.parameters)

  def attributes_print(self):
    print('Depth: {}, Width: {}, Max: {}, #Parameters: {}, Connectivity: {}'.format(
        self.depth,self.width,self.pa_max, self.num_parameters,self.connectivity))

# Parametrizations of simple functions
SimpleParametrizations = {'abs': [(np.array([[1.,-1]]), np.array([0.,0.])), 
                                  (np.array([[1.],[1.]]), np.array([0.]))],
                          'triangle': [(np.array([[1.,1.,1.]]), 
                                  np.array([0.,-0.5,-1.])), 
                                 (np.array([[2.],[-4.],[2.]]), 
                                  np.array([0.]))]}


# affine linear layer
# e.g. Affine(W,B)(x)=xW+B
class Affine(layers.Layer):

  def __init__(self, W, B):
    super(Affine, self).__init__()
    self.w = tf.Variable(initial_value=tf.constant(W,dtype='float32'),
                         trainable=True)
    self.b = tf.Variable(initial_value=tf.constant(B,dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

# realization map
class R(tf.keras.Model):

  def __init__(self, name_or_P, act=nn.relu):
    if isinstance(name_or_P, str):
      name_or_P=P(name_or_P)
    super(R, self).__init__(name=name_or_P.name)
    self.parametrization = name_or_P
    self.act = act
    self.affine_maps = [Affine(W, B) 
                        for W, B in self.parametrization.parameters]

  def call(self, input_tensor, training=False):

    x = input_tensor
    for affine in self.affine_maps[:-1]:
      x = affine(x)
      x = self.act(x)
    return self.affine_maps[-1](x)

# derivative map (for now: only supports Pa with Pa.out_dim = 1) 
# dR(Pa)(x)/dx via automatic differentiation
def D(Pa, act=nn.relu):
  def der(x):
    with tf.GradientTape() as t:
      x = tf.convert_to_tensor(x)
      t.watch(x)
      y = R(Pa, act = act)(x)
      return t.gradient(y, x)
  return der

# blockdiagonal matrix
def blockdiag(matrix_list):
  if len(matrix_list)==2:
    m1 = matrix_list[0]
    m2 = matrix_list[1]
    return np.block([[m1,np.zeros((m1.shape[0],m2.shape[1]))],
                     [np.zeros((m2.shape[0],m1.shape[1])),m2]])
  else:
    return blockdiag([blockdiag(matrix_list[:-1]),matrix_list[-1]])

# linear network
def Lin(W):
  zeros = np.zeros((W.shape[1],))
  return P([(W,zeros)])

# network concatenation
# e.g. Pa_list = [Pa1, Pa2, Pa3]
# R(Conc(Pa_list))(x)=R(Pa3)(R(Pa2)(R(Pa1)(x)))
# the order is reversed for easier usage
def Conc(Pa_list):
  if len(Pa_list)==1: #catch exceptional cases in other functions
    return Pa_list[0]
  if len(Pa_list)==2:
    Pa1=Pa_list[0]
    Pa2=Pa_list[1]
    W=np.matmul(Pa1.weights[-1],Pa2.weights[0])
    B=np.matmul(Pa1.biases[-1],Pa2.weights[0])+Pa2.biases[0]
    return P(Pa1.parameters[:-1]+[(W,B)]+Pa2.parameters[1:])
  else:
    return Conc([Conc(Pa_list[:-1]),Pa_list[-1]])

# positive elongation (with factor) helper function
def _pos_elong(Pa, L, factor = 1):
  Pa_pos = Identity(Pa.out_dim, 1, scale = factor)
  return P(Pa.parameters+[Pa_pos.parameters[0] for _ in range(L-Pa.depth)])

# (affine) linear combination
# e.g. Pa_list = [P1,P2], coeff_list = [A,B]
# then: R(Affine(Pa_list, coeff_list))(x)=A*R(P1)(x)+B*R(P2)(x))
# see parallelization for explanation of ind_list
def Affine_Comb(coeff_list, Pa_list = None, ind_list = None, affine = None):
  block = np.block([coeff_list]).transpose()
  if affine == None:
    Pa_comb = Lin(block)
  else:
    Pa_comb = P([(block,affine)])
  if Pa_list == None:
    return Pa_comb
  else: 
    return Conc([Par(Pa_list, ind_list=ind_list),Pa_comb])

# identity network with possible efficient scaling
# e.g. dim=2, L=4, scale = np.array([16,81]) 
# then R(Identity(dim, L, scale))(x)=(16x_1,81x_2)
# efficient: coefficients with magnitude |scale_i|**(1/L)
def Identity(dim, L, scale = 1):
  id = np.eye(dim)
  if L==1:
    return Lin(scale*id)
  elif L>1:
    factor = np.abs(scale)**(1/L)
    Pa_list = [_pos_elong(Lin(m), L, factor=factor) 
              for m in [factor*id, -factor*id]]
    ind = np.arange(dim)
    return Affine_Comb([np.sign(scale)*id, -np.sign(scale)*id], Pa_list=Pa_list, ind_list=[ind, ind])
  else:
    raise ValueError('L must be a natural number greater than 0.')

# update SimpleParametrizations
SimpleParametrizations.update({'identity': Identity(1,1).parameters})

# parallelization helper function for same depth
def _par_same(Pa_list):
  if len(Pa_list)==2:
    return P([(blockdiag([W1,W2]),np.block([B1,B2])) 
            for (W1, B1), (W2, B2) 
            in zip(Pa_list[0].parameters,Pa_list[1].parameters)])
  else:
    return _par_same([_par_same(Pa_list[:-1]),Pa_list[-1]])

# parallelization with indexed input
# e.g. Pa_list = [P1,P2], ind_list = [(2,0),(1,3)]
# then: R(Par(Pa_list, ind_list))(x_0,x_1,x_2,x_3)=(R(P1)(x_2,x_0),R(P2)(x_1,x_3))
def Par(Pa_list, ind_list = None, in_dim = None):
  L = max([Pa.depth for Pa in Pa_list])
  Pa = _par_same([Elongation(Pa, L) for Pa in Pa_list])
  if ind_list == None:
    return Pa
  else:
    if in_dim == None:
      in_dim = max([max(ind) for ind in ind_list])+1
    perms = [np.zeros((in_dim, Pa.in_dim)) for Pa in Pa_list]
    for perm, ind in zip(perms, ind_list):
      perm[ind,np.arange(len(ind))] = 1
    Pa_perms = Lin(np.block([perms]))
    return Conc([Pa_perms,Pa])

# sparse network concatenation
def Sparse_Conc(Pa_list):
  if len(Pa_list)==2:
    Id = Identity(Pa_list[0].out_dim, 2)
    return Conc([Pa_list[0],Id,Pa_list[1]])
  else:
    return Sparse_Conc([Sparse_Conc(Pa_list[:-1]),Pa_list[-1]])

# network elongation
def Elongation(Pa, L): 
  if Pa.depth == L:
    return Pa
  else:
    return Conc([Pa,Identity(Pa.out_dim,L-Pa.depth+1)])

# squaring helper function (interpolates the squaring function up to precision 4**(-k))
def _square(k):
  if isinstance(k,int):
    if k<=1 :
      return P('identity')
    else:
      Pa_triang = P('triangle')
      Pa_inp = _pos_elong(P('identity'), 2)
      Pa_sub_list = [_pos_elong(Affine_Comb([-2.**(-2*m),1]), 2) for m in range(1,k-1)]
      Pa_first = Par([Pa_triang, Pa_inp], ind_list=[[0],[0]])
      Pa_middle_list = [Par([Pa_triang, Pa_sub], ind_list=[[0],[0,1]]) 
                        for Pa_sub in Pa_sub_list]
      return Conc([Pa_first]+Pa_middle_list+[Affine_Comb([-2.**(-2*(k-1)),1])])
  else:
    raise ValueError('k must be a natural number.')

# approximation of squaring function on [-B,B] 
# up to error eps in Sobolev W^{1,\infty} norm
def Square(eps, B = 1.):
  k = int(np.ceil(2*np.log2(B/eps)+1))
  L = int(np.ceil(np.log2(B))) if B>1 else 1
  return Conc([Lin(np.array([[1/B]])),P('abs'),
                _square(k),Identity(1, L, scale=B**2)])
  
# approximation of multiplication function on [-B,B]
# up to error eps in Sobolev W^{1,\infty} norm
def Mult(eps, B = 1.):
  Pa_list = [Conc([Affine_Comb([1., a]), Square(2*eps, B = 2*B)]) 
            for a in [1.,-1.]]
  return Affine_Comb([0.25,-0.25], Pa_list=Pa_list, ind_list=[[0,1],[0,1]])

# approximation of monomials function (x,x^2,...,x^(2^2)) on [-B,B]
# up to error eps in Sobolev W^{1,\infty} norm
def _Monomial(K, eps, B = 1.):
  Pa_list_all=[]
  Pa_id = P('identity')
  eta = eps/(4.**(K**2)*B**(2**(K+1)))
  eta_list = [4**(k**2)*eta for k in range(1,K+1)]  
  E = 2*B**(2**K)
  for k in range(K):
    Pa_list=[]
    ind_list=[]
    Pa_square = Square(eta_list[k], B = E)
    Pa_mult = Mult(eta_list[k], B = E)
    for i in range(2**(k+1)):
      if i<=2**k-1:
        Pa_list.append(Pa_id)
        ind_list.append([i])
      elif i%2:
        Pa_list.append(Pa_square)
        ind_list.append([(i-1)//2])
      else:
        Pa_list.append(Pa_mult)
        ind_list.append([(i-2)//2,i//2])
    Pa_list_all.append(Par(Pa_list, ind_list=ind_list))
  return Conc(Pa_list_all)

# Monomial extension: arbitrary degree deg, optional with constant 1
def Monomial(deg, eps, B=1.0, const=True):
  K = int(np.ceil(np.log2(deg)))
  if const:
    e = np.zeros((deg+1,))
    e[0] = 1
    Pa_zero = P([(np.block([np.zeros((2**K,1)),np.eye(2**K,deg)]),e)])
  else:
    Pa_zero = Lin(np.eye(2**K,deg))
  return Conc([_Monomial(K, eps, B), Pa_zero])

# approximation of polynomial p[0]*x^(N-1)+ ... +p[N-1] on [-B,B]
# up to error eps in Sobolev W^{1,\infty} norm
def Poly(p, eps, B = 1.):
  p_len = len(p)
  coeff_sum = np.sum(np.abs(p))
  eta = eps/coeff_sum
  L = int(np.ceil(np.log2(coeff_sum)))
  Pa_scale = Identity(p_len, L, scale=p[::-1])
  return Conc([Monomial(p_len-1, eta, B), Pa_scale, 
               Affine_Comb(np.ones((1,p_len)))])
