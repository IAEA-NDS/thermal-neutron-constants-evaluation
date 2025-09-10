import tensorflow as tf
from quantities import prepare_funcs
from quantity_grammar import prepare_propagator

# some "derived" quantities in Axton's report are
# actually primary quantities (CAP(34). The following
# function allows to override the computation of the
# derived function by a value

def modify_funcs(funcs, param_vec, reac_map):
    orig_cap = funcs.CAP
    funcs.CAP = lambda r: (
        orig_cap(r) if ('CAP',r) not in reac_map
        else param_vec[reac_map[('CAP',r)]]
    )


def prepare_element_getter(param_vec, reac_map):
    def get_numpy_el(x, y):
        idx = reac_map[(x,y)]
        return param_vec[idx]
    return get_numpy_el 


def prepare_propagate(reac_map, exp_dt):
    def propagate(params):
        getter = prepare_element_getter(params, reac_map)
        funcs = prepare_funcs(getter)
        modify_funcs(funcs, params, reac_map)
        propfun = prepare_propagator(funcs)
        tf_results = []
        for i, row in exp_dt.iterrows():
            reac = row['MeasureFunc']
            tf_results.append(propfun(reac))
        return tf.stack(tf_results, axis=0)
    return propagate


def prepare_jacobian(propagate_fun):
    def jacobian(params):
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(params)
            result = propagate_fun(params)
        return tape.jacobian(result, params, experimental_use_pfor=True)
    return jacobian
