import tensorflow as tf
import logging


logger = logging.getLogger(__name__)


UNC_MODE = 'scale_by_expvals'


def prepare_chisquare(propagate_fun, expvals, relcov_linop, trafo=None):
    def chisquare(params):
        if trafo is not None:
            params = tf.square(params)
        propvals = propagate_fun(params)
        if UNC_MODE == 'scale_by_expvals':
            logger.info('scale by expvals')
            scale_op = tf.linalg.LinearOperatorDiag(
                expvals, is_positive_definite=True
            )
        else:
            logger.info('scale by propvals')
            scale_op = tf.linalg.LinearOperatorDiag(
                propvals, is_positive_definite=True
            )

        cov_linop = tf.linalg.LinearOperatorComposition(
            [scale_op, relcov_linop, scale_op],
            is_positive_definite=True, is_non_singular=True,
            is_self_adjoint=True, is_square=True
        )
        absdiff = expvals - propvals
        return tf.reduce_sum(absdiff * cov_linop.solvevec(absdiff))
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
