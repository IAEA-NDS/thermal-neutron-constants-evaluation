import tensorflow as tf
import pandas as pd
import numpy as np
from expdata import exp_dt
from quantity_grammar import prepare_propagator
from quantities import prepare_funcs
from utils import timeit


tuple_combis = tuple(
    (x, y) for x in ('SCA', 'SCR', 'ABS', 'FIS', 'NUB', 'WGA', 'WGF')
    for y in (33, 35, 39, 41)
)
tuple_combis += (('NUB', 52),)
tuple_combis += (('HLF', 33),)
tuple_combis += (('HLF', 39),)

reac_map = {
    n: i for i, n in enumerate(tuple_combis)
}

prior_mesh = tf.constant(np.arange(len(tuple_combis)), dtype=tf.float64) 

def get_numpy_el(x, y):
    idx = reac_map[(x,y)]
    return prior_mesh[idx]

funcs = prepare_funcs(get_numpy_el)
# orig_cap = funcs.CAP
# funcs.CAP = lambda r: orig_cap(r) if ('CAP',r) not in reac_map else prior_mesh[reac_map[('CAP',r)]]
# orig_f1cab = funcs.F1CAB
# funcs.F1CAB = lambda: prior_mesh[reac_map[('F1CAB',)]]

propfun = prepare_propagator(funcs)

failure_reacs = []
selected_exp_idx = []
for i, row in exp_dt.iterrows():
    print('dataset: ' + str(row['No']))
    reac = row['MeasureFunc']
    print(reac)
    try:
        propfun(reac)
        selected_exp_idx.append(i)
    except:
        failure_reacs.append(reac)

# >>> failure_reacs
# ['FLEM', 'CAP(34)', 'FFH(39,39)', 'FFH(39,39)/FFH(34,35)', 'FFH(39,39)/FFH(33,33)', 'FFH(39,39)/FFH(33,35)', 'F1BIG', 'FH1(39,39)', 'FH1(34,35)', 'FH1(34,35)', 'FH1(39,39)', 'FH1(39,39)/FH1(34,35)',
#  'FH1(39,39) / FH1(34,35)', 'F1CAB', 'F2CAB', 'F3CAB', 'F4CAB', 'F5CAB', 'CA(40)', 'CA(42)', 'GC116(39)', 'GC116(40)', 'GC116(41)', 'GC116(42)', 'GA116(39)', 'GA116(41)']

red_exp_dt = exp_dt.loc[selected_exp_idx].reset_index(drop=True)


def propagate(params):
    getter = lambda x, y: params[reac_map[(x,y)]] 
    funcs = prepare_funcs(getter)
    propfun = prepare_propagator(funcs)
    tf_results = []
    for i, row in red_exp_dt.iterrows():
        reac = row['MeasureFunc']
        tf_results.append(propfun(reac))
    return tf.stack(tf_results, axis=0)


def jacobian(params):
    with tf.GradientTape(persistent=False) as tape:
        result = propagate(params)
    return tape.jacobian(result, params, experimental_use_pfor=True)


params = tf.Variable(prior_mesh+10)

propfun = propagate
jacfun = jacobian

# optimization

def chisquare(params):
    propvals = propfun(params)
    expvals = tf.constant(red_exp_dt['InputValue'], dtype=tf.float64)
    uncs = tf.constant(red_exp_dt['Uncertainty']/100, dtype=tf.float64)
    absuncs = uncs*propvals
    diff = (0.5) * tf.square(expvals - propvals) / tf.square(absuncs)
    return tf.reduce_sum(diff)


def chisquare_gradient(params):
    with tf.GradientTape() as tape:
        tape.watch(params)
        y = chisquare(params)
    return tape.gradient(y, params)


def func_and_grad(params):
    with tf.GradientTape() as tape:
        tape.watch(params)
        y = chisquare(params)
    g = tape.gradient(y, params)
    return y, g


def chisquare_hessian(params):
    # propvals = propfun(params)
    # jacmat = jacfun(params)
    # uncs = tf.constant(red_exp_dt['Uncertainty']/100, dtype=tf.float64)
    # covmat = tf.linalg.LinearOperatorDiag(tf.square(uncs*propvals), is_positive_definite=True)
    # return tf.transpose(jacmat) @ covmat.solve(jacmat)
    with tf.GradientTape() as t2:
        t2.watch(params)
        with tf.GradientTape() as t1:
            t1.watch(params)
            y = chisquare(params)
        g = t1.gradient(y, params)
    h = t2.jacobian(g, params)
    return h


approx_neg_chisquare_hessian_tf = tf.function(approx_neg_chisquare_hessian)
res = timeit(approx_neg_chisquare_hessian_tf)(params)



from gmapy.tf_uq.inference import determine_MAP_estimate


func_and_grad_tf = tf.function(func_and_grad)
func_hessian_tf = tf.function(chisquare_hessian)

optres = determine_MAP_estimate(params, func_and_grad_tf, func_hessian_tf, ret_optres=True)

optres.position.numpy()

pd.DataFrame({
    'NAME': [f'{x} {y}' for x, y in tuple_combis],
    'POST': optres.position.numpy()
})

