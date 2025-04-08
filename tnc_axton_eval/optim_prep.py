import tensorflow as tf
import logging


logger = logging.getLogger(__name__)


UNC_MODE = 'scale_by_expvals'


def prepare_chisquare(propagate_fun, exp_dt, trafo=None):
    def chisquare(params):
        if trafo is not None:
            params = tf.square(params)
        propvals = propagate_fun(params)
        expvals = tf.constant(exp_dt['InputValue'], dtype=tf.float64)
        uncs = tf.constant(exp_dt['Uncertainty']/100, dtype=tf.float64)
        if UNC_MODE == 'scale_by_expvals':
            logger.info('scale by expvals')
            absuncs = uncs * expvals
        else:
            absuncs = uncs * propvals
            logger.info('scale by propvals')
        diff = (0.5) * tf.square(expvals - propvals) / tf.square(absuncs)
        return tf.reduce_sum(diff)
    return chisquare


def prepare_chisquare_gradient(chisquare_fun):
    def chisquare_gradient(params):
        with tf.GradientTape() as tape:
            tape.watch(params)
            y = chisquare_fun(params)
        return tape.gradient(y, params)
    return chisquare_gradient


def prepare_chisquare_and_gradient(chisquare_fun):
    def chisquare_and_gradient(params):
        with tf.GradientTape() as tape:
            tape.watch(params)
            y = chisquare_fun(params)
        g = tape.gradient(y, params)
        return y, g
    return chisquare_and_gradient


def prepare_chisquare_hessian(chisquare_fun):
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
                y = chisquare_fun(params)
            g = t1.gradient(y, params)
        h = t2.jacobian(g, params)
        return h
    return chisquare_hessian
