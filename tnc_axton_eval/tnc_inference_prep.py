from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import scipy as scp
from utils import timeit
import logging
import sys

from prior_prep import (
    tuple_combis_mic as tuple_combis,
    reac_map_mic as reac_map,
    assign_startvals,
    startvals_map,
)
from propagate_prep import (
    prepare_propagate,
    prepare_jacobian,
)
import optim_prep
from optim_prep import (
    prepare_chisquare,
    prepare_chisquare_gradient,
    prepare_chisquare_and_gradient,
    prepare_chisquare_hessian,
    logger as optim_prep_logger,
)
from expdata import (
    all_exp_dt as exp_dt
)
from gmapy.tf_uq.custom_distributions import BaseDistribution

basepath = Path(__file__).resolve().parent

logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
optim_prep_logger.setLevel(logging.INFO)

# NOTE: directly starting with UNC_MODE='scale_by_propvals'
#       would not give nice results (formally converged yes, but weird numbers).
#       First, good values should be found with UNC_MODE='scale_by_expvals',
#       and the result as starting values in the fit with UNC_MODE='scale_by_propvals'

optim_prep.UNC_MODE = 'scale_by_expvals'

seed = np.random.randint(0, 1000, 1).item()
np.random.seed(seed)
startvals = np.random.uniform(1, 2, len(tuple_combis))

# select experimental datasets
failure_reacs = []
selected_exp_idx = []
for i in range(len(exp_dt)):
    exp_dt_row = exp_dt.iloc[i:i+1].reset_index()
    propfun = prepare_propagate(reac_map, exp_dt_row)
    print('dataset: ' + str(exp_dt_row.iloc[0]['No']))
    reac = exp_dt_row.iloc[0]['MeasureFunc']
    print(reac)
    try:
        propfun(startvals)
        selected_exp_idx.append(i)
    except:
        failure_reacs.append(reac)

# ['FLEM', 'CAP(34)', 'FFH(39,39)', 'FFH(39,39)/FFH(34,35)', 'FFH(39,39)/FFH(33,33)', 'FFH(39,39)/FFH(33,35)', 'F1BIG', 'FH1(39,39)', 'FH1(34,35)', 'FH1(34,35)', 'FH1(39,39)', 'FH1(39,39)/FH1(34,35)',
#  'FH1(39,39) / FH1(34,35)', 'F1CAB', 'F2CAB', 'F3CAB', 'F4CAB', 'F5CAB', 'CA(40)', 'CA(42)', 'GC116(39)', 'GC116(40)', 'GC116(41)', 'GC116(42)', 'GA116(39)', 'GA116(41)']


red_exp_dt = exp_dt.loc[selected_exp_idx].reset_index(drop=True)

propagate = prepare_propagate(reac_map, red_exp_dt)
jacobian = prepare_jacobian(propagate)
assign_startvals(startvals, startvals_map, reac_map)
startvals_tf = tf.constant(np.sqrt(np.abs(startvals)), dtype=tf.float64)

expvals = red_exp_dt['InputValue'].to_numpy()

# preapre the relative covariance matrix
# relcov_linop = tf.linalg.LinearOperatorDiag(
#     np.square(red_exp_dt['Uncertainty'] / 100.0), is_positive_definite=True
# )

ags_index = np.loadtxt(basepath / 'tnc_cov_data/thermalcst.mic')
cov_info = np.loadtxt(basepath / 'tnc_cov_data/ags.mic')
assert np.all(ags_index[:,0] == cov_info[:,0])

no_gilles = ags_index[:,0].astype(int).tolist()
no_georg = red_exp_dt['No'].tolist()

stat_uncs = cov_info[:,2]
partial_uncs = cov_info[:,3:]

rel_covmat = sum([x.reshape(-1,1) @ x.reshape(1,-1) for x in partial_uncs.T])
rel_covmat += np.diag(np.square(stat_uncs))

full_covmat = np.diag(np.square(red_exp_dt['Uncertainty'] / 100))
idcs = [no_georg.index(n) for n in no_gilles if n in no_georg]
sel = [n in no_georg for n in no_gilles]
full_covmat[np.ix_(idcs, idcs)] = rel_covmat[np.ix_(sel, sel)]

relcov_linop = tf.linalg.LinearOperatorFullMatrix(full_covmat, is_positive_definite=True, is_square=True)

# prepare the fit quantities
trafo = tf.square
chisquare = prepare_chisquare(propagate, expvals, relcov_linop, trafo=trafo)
chisquare_and_gradient = prepare_chisquare_and_gradient(chisquare)
chisquare_hessian = prepare_chisquare_hessian(chisquare)

func_and_grad_tf = tf.function(chisquare_and_gradient)
func_hessian_tf = tf.function(chisquare_hessian)


gma_to_axt_map = {
    'MT:1-R1:8': 'FIS(35)',
    'MT:1-R1:9': 'FIS(39)',
    'MT:1-R1:11': 'WGA(33)',
    'MT:1-R1:12': 'WGF(33)',
    'MT:1-R1:13': 'SCA(33)',
    'MT:1-R1:14': 'FIS(33)',
    'MT:1-R1:15': 'CA(33)',
    'MT:1-R1:16': 'NUB(33)',
    'MT:1-R1:17': 'WGA(35)',
    'MT:1-R1:18': 'WGF(35)',
    'MT:1-R1:19': 'SCA(35)',
    'MT:1-R1:20': 'CA(35)',
    'MT:1-R1:21': 'NUB(35)',
    'MT:1-R1:22': 'WGA(39)',
    'MT:1-R1:23': 'WGF(39)',
    'MT:1-R1:24': 'SCA(39)',
    'MT:1-R1:25': 'CA(39)',
    'MT:1-R1:26': 'NUB(39)',
    'MT:1-R1:27': 'WGA(41)',
    'MT:1-R1:28': 'WGF(41)',
    'MT:1-R1:29': 'SCA(41)',
    'MT:1-R1:30': 'FIS(41)',
    'MT:1-R1:31': 'CA(41)',
    'MT:1-R1:32': 'NUB(41)',
    'MT:1-R1:33': 'NUB(52)',
    'MT:3-R1:9-R2:8': 'FIS(39)/FIS(35)',
    'MT:3-R1:30-R2:8': 'FIS(41)/FIS(35)',
    'MT:3-R1:14-R2:8': 'FIS(33)/FIS(35)',
    'MT:3-R1:15-R2:14': 'CA(33)/FIS(33)',
    'MT:3-R1:20-R2:8': 'CA(35)/FIS(35)',
    'MT:3-R1:25-R2:9': 'CA(39)/FIS(39)',
}


axt_to_gma_map = {v: k for k, v in gma_to_axt_map.items()}


def prepare_distribute_params(gma_reacs, axt_reacs, reacs, axt_to_gma_map=None):
    def distribute_params(params_tf):
        reacs_t = reacs
        if axt_to_gma_map is not None:
            reacs_t = [axt_to_gma_map.get(r, r) for r in reacs]
        src_idcs1 = [i for i, r in enumerate(reacs_t) if r in gma_reacs]
        src_idcs2 = [i for i, r in enumerate(reacs) if r in axt_reacs]
        gather1 = tf.gather(params_tf, src_idcs1)
        gather2 = tf.gather(params_tf, src_idcs2)
        reacs_gather1 = [reacs_t[idx] for idx in src_idcs1]
        reacs_gather2 = [reacs[idx] for idx in src_idcs2]
        tar_idcs1 = [[gma_reacs.index(r)] for r in reacs_gather1]
        tar_idcs2 = [[axt_reacs.index(r)] for r in reacs_gather2]
        scatter1 = tf.scatter_nd(tar_idcs1, gather1, [len(gma_reacs)])
        scatter2 = tf.scatter_nd(tar_idcs2, gather2, [len(axt_reacs)])
        return scatter1, scatter2
    return distribute_params


def prepare_combine_hessians(gma_reacs, axt_reacs, reacs, axt_to_gma_map=None):
    def combine_hessians(gma_hess, axt_hess):
        reacs_t = reacs
        if axt_to_gma_map is not None:
            reacs_t = [axt_to_gma_map.get(r, r) for r in reacs]
        idcs1 = [reacs_t.index(r) for r in gma_reacs]
        idcs1_2d = [(i1, i2) for i1 in idcs1 for i2 in idcs1]
        exp_gma_hess = tf.scatter_nd(idcs1_2d, tf.reshape(gma_hess, [-1]), [len(reacs)]*2)
        idcs2 = [reacs.index(r) for r in axt_reacs]
        idcs2_2d = [(i1, i2) for i1 in idcs2 for i2 in idcs2]
        exp_axt_hess = tf.scatter_nd(idcs2_2d, tf.reshape(axt_hess, [-1]), [len(reacs)]*2)
        return exp_gma_hess + exp_axt_hess
    return combine_hessians


class AxtonChiSquareDist(BaseDistribution):

    def log_prob(self, x):
        return (-0.5)*chisquare(x)

    def log_prob_hessian(self, x):
        return (-0.5)*chisquare_hessian(x)


class GmaAxtDist(BaseDistribution):
    def __init__(self, gma_reacs, gma_dist, axt_reacs, axt_dist, reacs):
        assert len(np.unique(reacs)) == len(reacs)
        reacs_t = [axt_to_gma_map.get(r,r) for r in reacs]
        assert all(r in reacs_t for r in gma_reacs)
        assert all(r in reacs for r in axt_reacs)
        self._gma_dist = gma_dist
        self._axt_dist = axt_dist
        self._distribute_params = self._prepare_distribute_params(gma_reacs, axt_reacs, reacs, axt_to_gma_map)
        self._combine_hessians = prepare_combine_hessians(gma_reacs, axt_reacs, reacs, axt_to_gma_map)

    def log_prob(self, x):
        gma_inp, axt_inp = self._distribute_params(x)
        gma_log_prob = self._gma_dist.log_prob(gma_inp)
        axt_log_prob = self._axt_dist.log_prob(axt_inp)
        return gma_log_prob + axt_log_prob

    def log_prob_hessian(self, x):
        gma_inp, axt_inp = self._distribute_params(x)
        gma_hess = self._gma_dist.log_prob_hessian(gma_inp)
        axt_hess = self._axt_dist.log_prob_hessian(axt_inp)
        return self._combine_hessian(gma_hess, axt_hess)


# FOR AD-HOC TESTING
# axt_to_gma_map = {'b': 'u'}
#
# reacs1 = ('a', 'u', 'c', 'd')
# reacs2 = ('c', 'd', 'e', 'f')
# reacs = ('a', 'b', 'c', 'd', 'e', 'f')
# params_tf = tf.range(len(reacs))
#
# distribute_params = prepare_distribute_params(reacs1, reacs2, reacs, axt_to_gma_map)
# distribute_params(params_tf)
#
# hess1 = tf.constant([[i*4+j for j in range(4)] for i in range(4)])
# hess2 = tf.constant([[i*4+j for j in range(4)] for i in range(4)])
#
# combine_hessians = prepare_combine_hessians(reacs1, reacs2, reacs, axt_to_gma_map)
# combine_hessians(hess1, hess2)
