import tensorflow as tf
from .quantity_grammar import prepare_propagator


# Define quantities that are considered atomic (i.e. not derived from others) 
PRIMARY_QUANTITIES = (
    'ABS', 'FIS', 'SCA', 'SCR', 'WGA', 'WGF', 'GC116', 'GA116', 'HLF', 'NUB'
)


class Bunch(object):
    """Class to store dictionary."""
    def __init__(self, adict):
        """Initialize class with dictionary."""
        self.__dict__.update(adict)
        super().__setattr__('_varnames', set(adict))

    def __setattr__(self, name, value):
        """Set member variable in class.""" 
        self._varnames.add(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        """Delete member variable from class."""
        self._varnames.remove(name)
        super().__delattr__(name)

    def is_valid(self, name):
        """Check if member variable with given name exists."""
        return name in self._varnames


def prepare_funcs(getter, primary_quantities=None):
    """Create Bunch object with quantity functions.

    Create a Bunch object with quantity functions, e.g.,
    `ABS(r)`, `FA(r)` with `r` being an isotope indicator,
    e.g. 39 for Pu-239. These functions are defined in the 
    Axton 1986 report [1, page 23].

    [1] https://nds.iaea.org/standards/Reports/Axton-GE-PH-01-86.pdf

    Args:
        getter (callable): The getter function expects two args
            `x` (quantity name) and `y` (isotope identification number)
            and shall return a function that takes a 1D nd.array
            and returns the value at a specific index of the array,
            with the index determined by the `(x, y)` combination.
            For instance, `(x, y)` could be `(ABS, 39)`.

    Returns:
        Bunch: A Bunch object with quantity functions, e.g. `ABS(r)`.
    """
    if primary_quantities is None:
        primary_quantities = PRIMARY_QUANTITIES

    funcs = {q: lambda r, q=q: getter(q, r) for q in primary_quantities}
    f = Bunch(funcs)
    # define funcs for derived quantities
    f.FA = lambda r: f.ABS(r) * f.WGA(r)
    f.FF = lambda r: f.FIS(r) * f.WGF(r)
    f.CA = lambda r: f.ABS(r) - f.FIS(r)
    # check with Gilles if type in Axton report and GA == WGA 
    f.CAP = lambda r: (f.ABS(r) * f.WGA(r)) - (f.FIS(r)*f.WGF(r))  
    f.ETA = lambda r: f.NUB(r) * f.FIS(r) / f.ABS(r)
    f.F1ETA = lambda r: f.NUB(r) * f.FF(r) / f.FA(r)
    f.F2ETA = lambda r: f.NUB(r) * f.FF(r)
    f.F3ETA = lambda r: (f.NUB(r) * f.FF(r)) - f.FA(r)
    f.FH1 = lambda r1, r2: f.FIS(r2) * f.HLF(r1)
    f.FFH = lambda r1, r2: f.FF(r2) * f.HLF(r1)
    # define special functions
    f.FLEM = lambda: f.CAP(33) / (f.FA(33) - f.CAP(34))  
    f.F1CAB = lambda: (f.CA(40) * f.GC116(40)) - (f.CA(42) * f.GC116(42))
    f.F2CAB = lambda: (f.ABS(39) * f.GA116(39)) - (f.CA(42) * f.GC116(42)) 
    f.F3CAB = lambda: (f.ABS(39) * f.GA116(39)) / (f.CA(39) * f.GC116(39))
    f.F4CAB = lambda: (f.ABS(41) * f.GA116(41) - f.ABS(39) * f.GA116(39)) / f.F1HLF()
    f.F5CAB = lambda: f.ABS(41) * f.GA116(41) / (f.CA(41) * f.GC116(41) * f.F2HLF()) 
    f.F1BIG = lambda: f.FF(41) / (f.FFH(39, 39) * f.F3HLF())
    f.F1HLF = lambda: 1 + 0.12966 * (f.HLF(41) - 14.05)
    f.F2HLF = lambda: 1 + 0.0225 * (f.HLF(41) - 14.05)
    f.F3HLF = lambda: 1 + 0.00395 * (14.5 - f.HLF(41)) / (14.5-12.9)  # Is this correct?
    return f


def prepare_element_getter(param_vec, reac_map):
    """Prepare getter func to get element from param_vec.

    Args:
        param_vec (array-like): 1D array (e.g. `tf.tensor` or `nd.array`)
        reac_map (dict): Dictionary mapping `(x, y)` tuples to an index 

    Returns:
        callable[[Any, Any], Any]: A function f(x, y) that takes
            two arguments and returns the element in `param_vec`
            at the index associated with `(x, y)` in `reac_map`.
    """
    def get_element(x, y):
        idx = reac_map[(x,y)]
        return param_vec[idx]
    return get_element


def prepare_propagate(reac_map, exp_dt, modify_funcs=None):
    """Prepare a propagate function.

    Construct a function that maps a parameter
    vector to a vector corresponding to experimental
    quantities. 

    Args:
        reac_map (dict): Map from tuple to index in vector.
        exp_dt (pd.DataFrame): Data frame with experimental features.
        modify_funcs (callable[[Bunch, array_like, dict], None]):
            Function to modify funcs Bunch object inplace.

    Returns:
        callable[[array_like], array_like]
    """
    def propagate(params):
        getter = prepare_element_getter(params, reac_map)
        funcs = prepare_funcs(getter)
        if modify_funcs is not None:
            modify_funcs(funcs, params, reac_map)
        propfun = prepare_propagator(funcs)
        tf_results = []
        for i, row in exp_dt.iterrows():
            reac = row['MeasureFunc']
            tf_results.append(propfun(reac))
        return tf.stack(tf_results, axis=0)
    return propagate


def prepare_jacobian(propagate_fun):
    """Prepare a Jacobian function.

    Construct a function computing the Jacobian
    for `propagate_fun`.

    Args:
        propagate_fun (callable[array_like]): Vector-valued function 

    Returns:
        callable[[array_like], array_like]

    """
    def jacobian(params):
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(params)
            result = propagate_fun(params)
        return tape.jacobian(result, params, experimental_use_pfor=True)
    return jacobian
