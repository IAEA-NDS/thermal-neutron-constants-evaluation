import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from expdata import exp_dt
from quantity_grammar import prepare_propagator
from quantities import prepare_funcs
from utils import timeit


# tuple_combis = tuple(
#     (x, y) for x in ('SCA', 'SCR', 'ABS', 'FIS', 'NUB', 'WGA', 'WGF')
#     for y in (33, 35, 39, 41)
# )
tuple_combis = tuple(
    (x, y) for x in ('SCA', 'SCR', 'ABS', 'FIS', 'NUB', 'WGA', 'WGF')
    for y in (33, 35, 39, 41)
)
tuple_combis += (('NUB', 52),)
# tuple_combis += (('WGF', 33),)
# tuple_combis += (('WGF', 35),)
# tuple_combis += (('WGF', 39),)
# tuple_combis += (('WGF', 41),)
# tuple_combis += (('WGA', 33),)
# tuple_combis += (('WGA', 35),)
# tuple_combis += (('WGA', 39),)

# for debugging

reac_map = {
    n: i for i, n in enumerate(tuple_combis)
}

prior_mesh = tf.constant(np.arange(len(tuple_combis))[::-1], dtype=tf.float64)

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

# optimization

try:
    startvals_map
except NameError:
    startvals_map = {}


np.random.seed(48)
startvals = np.random.uniform(0.1, 2, len(reac_map))
for k, idx in reac_map.items():
    if k in startvals_map:
        startvals[idx] = startvals_map[k]

startvals = tf.constant(startvals, dtype=tf.float64)
startpreds = propagate(startvals)


myprop = lambda x: propagate(tf.square(x))


def chisquare(params):
    propvals = myprop(params)
    expvals = tf.constant(red_exp_dt['InputValue'], dtype=tf.float64)
    uncs = tf.constant(red_exp_dt['Uncertainty']/100, dtype=tf.float64)
    absuncs = uncs * expvals
    diff = (0.5) * tf.square(expvals - propvals) / tf.square(absuncs)
    return tf.reduce_sum(diff)


def chisquare_gradient(params):
    with tf.GradientTape() as tape:
        tape.watch(params)
        y = chisquare(params)
    return tape.gradient(y, params)


def chisquare_and_gradient(params):
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


chisquare_hessian_tf = tf.function(chisquare_hessian)
res = timeit(chisquare_hessian_tf)(params)


from gmapy.tf_uq.inference import determine_MAP_estimate


func_and_grad_tf = tf.function(chisquare_and_gradient)
func_hessian_tf = tf.function(chisquare_hessian)

optres1 = determine_MAP_estimate(startvals, func_and_grad_tf, func_hessian_tf, ret_optres=True)
optres2 = tfp.optimizer.bfgs_minimize(func_and_grad_tf, params)
np.max(optres1.position - optres2.position)


opt_params = np.square(optres1.position.numpy())

pd.DataFrame({
    'NAME': [f'{x} {y}' for x, y in tuple_combis],
    'POST': opt_params
})
red_exp_dt['PRED'] = propagate(opt_params)
red_exp_dt['RES'] = (red_exp_dt['PRED'] - red_exp_dt['InputValue']) / red_exp_dt['InputValue'] / (red_exp_dt['Uncertainty']/100)
red_exp_dt


# startvals_map = {k: optres1.position.numpy()[idx] for k, idx in reac_map.items()}
startvals_map.update({k: optres1.position.numpy()[idx] for k, idx in reac_map.items()})
# startvals_map[('WGF', 33)] = 1.0
# startvals_map[('WGF', 35)] = 1.0
# startvals_map[('WGF', 39)] = 1.0
# startvals_map[('WGF', 41)] = 1.0
# startvals_map[('FIS', 41)] = np.sqrt(1011.0)

