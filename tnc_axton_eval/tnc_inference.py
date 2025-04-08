import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from utils import timeit

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
from optim_prep import (
    prepare_chisquare,
    prepare_chisquare_gradient,
    prepare_chisquare_and_gradient,
    prepare_chisquare_hessian,
)
from expdata import (
    exp_dt
)
from gmapy.tf_uq.inference import determine_MAP_estimate


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
startvals_tf = tf.constant(startvals, dtype=tf.float64)

trafo = tf.square
chisquare = prepare_chisquare(propagate, red_exp_dt, trafo=trafo) 
chisquare_and_gradient = prepare_chisquare_and_gradient(chisquare)
chisquare_hessian = prepare_chisquare_hessian(chisquare)

func_and_grad_tf = tf.function(chisquare_and_gradient)
func_hessian_tf = tf.function(chisquare_hessian)

optres = determine_MAP_estimate(startvals, func_and_grad_tf, func_hessian_tf, ret_optres=True)
opt_params = (trafo(optres.position)).numpy()

# res_list = []
res_list.append(pd.DataFrame({
    'NAME': [f'{x} {y}' for x, y in tuple_combis],
    'POST': opt_params,
    'SEED': seed,
}))


res_list[-1]

# comparison with experiments
# red_exp_dt['PRED'] = propagate(opt_params)
# red_exp_dt['RES'] = (red_exp_dt['PRED'] - red_exp_dt['InputValue']) / red_exp_dt['InputValue'] / (red_exp_dt['Uncertainty']/100)
# red_exp_dt
