"""Iterative Generalized Least Squares solver."""

import re
import numpy as np
import tensorflow as tf
from .quantities import prepare_funcs, prepare_element_getter, Bunch
from .quantity_grammar import prepare_propagator


# =========================================================================
# Expression conversion
# =========================================================================

def convert_measurement_expr(expr):
    """Convert Axton Table 1 notation to parseable grammar notation.

    Converts argument format:
      "FA 39" -> "FA(39)"
      "33 FFH 35" -> "FFH(33,35)"

    The JSON data should already use "/" for division and "+" for
    addition, so no operator disambiguation is needed.
    """
    # Step 1: Convert 2-arg notation "NUM FUNC NUM" -> "FUNC(NUM,NUM)"
    expr = re.sub(r'(\d+)\s+([A-Za-z]\w*)\s+(\d+)', r'\2(\1,\3)', expr)

    # Step 2: Convert 1-arg notation "FUNC NUM" -> "FUNC(NUM)"
    expr = re.sub(r'([A-Za-z]\w+)\s+(\d+)', r'\1(\2)', expr)

    return expr


# =========================================================================
# Forward model
# =========================================================================

def build_param_space(initial_params, fixed_params=None):
    """Build parameter vector, index map, and metadata.

    Parameters
    ----------
    initial_params : dict
        Maps (quantity, nuclide) -> initial value for free parameters.
    fixed_params : dict, optional
        Maps (quantity, nuclide) -> fixed value.

    Returns
    -------
    param_vec : np.ndarray
        1D array of parameter values (free params first, then fixed).
    reac_map : dict
        Maps (quantity, nuclide) -> index in param_vec.
    free_indices : np.ndarray
        Indices of free parameters in param_vec.
    """
    if fixed_params is None:
        fixed_params = {}

    free_keys = list(initial_params.keys())
    fixed_keys = list(fixed_params.keys())

    param_vec = np.array(
        [initial_params[k] for k in free_keys]
        + [fixed_params[k] for k in fixed_keys]
    )

    reac_map = {}
    for i, k in enumerate(free_keys):
        reac_map[k] = i
    offset = len(free_keys)
    for i, k in enumerate(fixed_keys):
        reac_map[k] = offset + i

    free_indices = np.arange(len(free_keys))

    return param_vec, reac_map, free_indices


def get_measurement_exprs(data, measurement_nos=None):
    """Extract measurement expressions from JSON data, optionally filtered.

    Parameters
    ----------
    data : dict
        Dictionary representation of the axton1986_data.json file.
    measurement_nos : list of int, optional
        If given, return only expressions for these measurement IDs
        (in the order provided). If None, return all.

    Returns
    -------
    exprs : list of str
        Measurement function strings.
    """
    by_no = {m['no']: m['measured_function'] for m in data['measurements']}
    if measurement_nos is None:
        return [m['measured_function'] for m in data['measurements']]
    return [by_no[no] for no in measurement_nos]


def _make_getter(params_tf, reac_map):
    """Create a getter that retrieves values from a TF tensor."""
    def getter(quantity, nuclide):
        key = (quantity, nuclide)
        if key in reac_map:
            return params_tf[reac_map[key]]
        raise KeyError(f"Unknown parameter: {key}")
    return getter


def _override_derived_as_primary(funcs, getter, reac_map):
    """Override derived quantities that are treated as primary parameters.

    For nuclides 40, 42: CA is a free parameter (not derived from ABS-FIS).
    For nuclide 34: CAP is a free parameter.
    """
    original_CA = funcs.CA
    funcs.CA = lambda r: (
        getter('CA', r) if ('CA', r) in reac_map else original_CA(r)
    )

    original_CAP = funcs.CAP
    funcs.CAP = lambda r: (
        getter('CAP', r) if ('CAP', r) in reac_map else original_CAP(r)
    )


def create_propagate_fn(reac_map, measurement_exprs):
    """Create a function that maps parameters to predicted measurements.

    Parameters
    ----------
    reac_map : dict
        Maps (quantity, nuclide) -> index in parameter vector.
    measurement_exprs : list of str
        Measurement function strings in Axton Table 1 notation.

    Returns
    -------
    propagate : callable
        Function mapping a TF tensor of parameters to a TF tensor
        of predicted measurement values.
    """
    converted_exprs = [convert_measurement_expr(e) for e in measurement_exprs]

    def propagate(params_tf):
        getter = _make_getter(params_tf, reac_map)
        funcs = prepare_funcs(getter)
        _override_derived_as_primary(funcs, getter, reac_map)
        propfun = prepare_propagator(funcs)
        results = []
        for expr in converted_exprs:
            results.append(propfun(expr))
        return tf.stack(results)

    return propagate


# =========================================================================
# GLS solver
# =========================================================================

def compute_jacobian(propagate_fn, params, free_indices):
    """Compute the Jacobian of propagate_fn w.r.t. free parameters.

    Returns the absolute Jacobian matrix (n_meas x n_free_params).
    """
    params_tf = tf.Variable(params, dtype=tf.float64)
    with tf.GradientTape() as tape:
        predictions = propagate_fn(params_tf)
    full_jac = tape.jacobian(predictions, params_tf)
    # Extract columns for free parameters only
    return full_jac.numpy()[:, free_indices]


def iterative_gls(measured_values, rel_covmat, propagate_fn,
                  initial_params, free_indices,
                  max_iter=20, tol=1e-8):
    """Run iterative Generalized Least Squares.

    Implements the procedure from Axton 1986 Section 2, working
    in relative quantities throughout.

    Parameters
    ----------
    measured_values : np.ndarray
        Vector of measured values (length n).
    rel_covmat : np.ndarray
        Relative covariance matrix (n x n) in percent squared.
    propagate_fn : callable
        Maps parameter vector (TF tensor) to predicted values (TF tensor).
    initial_params : np.ndarray
        Full parameter vector (free + fixed).
    free_indices : np.ndarray
        Indices of free parameters within the parameter vector.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the max absolute relative correction.

    Returns
    -------
    result : dict with keys:
        'params': final parameter vector
        'free_values': fitted values of free parameters
        'cov_free': covariance matrix of free parameters (relative, pct^2)
        'chi2': chi-squared value
        'n_iter': number of iterations performed
        'weighted_residuals': weighted residuals r'
        'converged': bool
    """
    params = initial_params.copy()
    n_meas = len(measured_values)
    n_free = len(free_indices)

    # Convert relative covmat from percent^2 to fraction^2
    Z = rel_covmat / 1e4
    Z_inv = np.linalg.inv(Z)

    converged = False
    for iteration in range(max_iter):
        # Compute predictions
        params_tf = tf.constant(params, dtype=tf.float64)
        predictions = propagate_fn(params_tf).numpy()

        # Relative observation vector: y = (measured - predicted) / predicted
        y = (measured_values - predictions) / predictions

        # Compute absolute Jacobian and convert to relative form
        J_abs = compute_jacobian(propagate_fn, params, free_indices)

        # Relative Jacobian: A_ij = J_abs_ij * param_j / prediction_i
        free_vals = params[free_indices]
        A = J_abs * free_vals[np.newaxis, :] / predictions[:, np.newaxis]

        # Solve normal equations: b = (A^T Z^-1 A)^-1 A^T Z^-1 y
        AtZi = A.T @ Z_inv
        AtZiA = AtZi @ A
        AtZiy = AtZi @ y
        b = np.linalg.solve(AtZiA, AtZiy)

        # Update free parameters (relative corrections)
        params[free_indices] = params[free_indices] * (1 + b)

        max_correction = np.max(np.abs(b))
        if max_correction < tol:
            converged = True
            break

    # Final predictions and diagnostics
    params_tf = tf.constant(params, dtype=tf.float64)
    predictions = propagate_fn(params_tf).numpy()
    y_final = (measured_values - predictions) / predictions

    # Recompute Jacobian at final point
    J_abs = compute_jacobian(propagate_fn, params, free_indices)
    free_vals = params[free_indices]
    A = J_abs * free_vals[np.newaxis, :] / predictions[:, np.newaxis]

    # Parameter covariance (relative, fraction^2)
    AtZiA = A.T @ Z_inv @ A
    V_free = np.linalg.inv(AtZiA)

    # Cholesky factor R such that R^T R = Z^-1
    R = np.linalg.cholesky(Z_inv).T
    y_prime = R @ y_final
    r_prime = y_prime - (R @ A) @ np.linalg.solve(AtZiA, A.T @ Z_inv @ y_final)
    chi2 = float(r_prime @ r_prime)

    # Convert parameter covariance to percent^2
    V_free_pct = V_free * 1e4

    return {
        'params': params,
        'free_values': params[free_indices],
        'cov_free': V_free_pct,
        'chi2': chi2,
        'dof': n_meas - n_free,
        'n_iter': iteration + 1,
        'weighted_residuals': r_prime,
        'predictions': predictions,
        'converged': converged,
    }
