import tnc_inference_prep as prep
import numpy as np
import pandas as pd
import scipy as scp
import tensorflow as tf
from gmapy.tf_uq.inference import (
    determine_MAP_estimate,
    generate_MCMC_chain,
)
from gmapy.mcmc_inference import compute_effective_sample_size

# do a MAP
optres = determine_MAP_estimate(prep.startvals_tf, prep.func_and_grad_tf, prep.func_hessian_tf, ret_optres=True)
opt_params = (prep.trafo(optres.position)).numpy()

# do MCMC
log_prob = lambda x: (-prep.chisquare(x))
chain, _ = generate_MCMC_chain(
    optres.position, log_prob, prep.chisquare_hessian,
    num_leapfrog_steps=3, step_size=0.001, num_burnin_steps=5000,  num_results=20000
)
opt_params_mcmc = np.mean(prep.trafo(chain).numpy(), axis=0)
opt_params_mcmc_uncs = np.std(prep.trafo(chain).numpy(), axis=0)


post_df = pd.DataFrame({
    'NAME': [f'{x} {y}' for x, y in prep.tuple_combis],
    'START': np.square(prep.startvals_tf),
    'POST': opt_params,
    'POST_MCMC': opt_params_mcmc,
    'POST_MCMC_UNC': opt_params_mcmc_uncs,
    'SEED': prep.seed,
})

#
# To compute capture CA (capture) cross sections
#
# t0 = post_df[post_df.NAME.str.match('ABS')].reset_index(drop=True)
# t0['NAME'] = t0['NAME'].str.replace('ABS', 'CA')
# t1 = post_df[post_df.NAME.str.match('ABS')].reset_index(drop=True)
# t2 = post_df[post_df.NAME.str.match('FIS')].reset_index(drop=True)
# t0['START'] = t1['START'] - t2['START']
# t0['POST'] = t1['POST'] - t2['POST_MCMC']
# t0['POST'] = t1['POST'] - t2['POST_MCMC']
# post_df = pd.concat([post_df, t0], ignore_index=True)

#
# To compute CA uncertainty (ad-hoc)
#
# np.std(prep.trafo(chain[:,9]).numpy() - prep.trafo(chain[:,13]).numpy())
# plt.hist(prep.trafo(chain[:,9]).numpy() - prep.trafo(chain[:,13]).numpy(), bins=30)
# plt.show()

#
# To plot posterior distribution
#
# import matplotlib.pyplot as plt
# for idx in range(26):
#     plt.title(post_df.loc[idx, 'NAME'])
#     plt.hist(np.square(chain.numpy()[:,idx]), bins=200)
#     plt.axvline(post_df.at[idx, 'START'], c='b')
#     plt.axvline(post_df.at[idx, 'POST'], c='r')
#     plt.axvline(post_df.at[idx, 'POST_MCMC'], c='g')
#     # plt.xlim(14.51, 14.58)
#     print(post_df.loc[idx])
#     plt.show()


# method 1 (exact, analytic)
chisquare_notrafo = prep.prepare_chisquare(prep.propagate, prep.expvals, prep.relcov_linop)
chisquare_hessian_notrafo = prep.prepare_chisquare_hessian(chisquare_notrafo)
invpostcov = chisquare_hessian_notrafo(tf.constant(opt_params))
postcov = np.linalg.inv(invpostcov)
postuncs = np.sqrt(postcov.diagonal())
post_df['RELUNC1'] = postuncs / post_df['POST'] * 100

# method 2 (sandwhich, analytic)
propfun = prep.prepare_propagate(prep.reac_map, prep.red_exp_dt)
propvals = propfun(opt_params).numpy()
jacfun = prep.prepare_jacobian(propfun)
J = jacfun(tf.constant(opt_params))
expcov = prep.relcov_linop.to_dense().numpy() * (propvals.reshape(-1, 1) @ propvals.reshape(1,-1))
inv_expcov = np.linalg.inv(expcov)
postcov = np.linalg.inv(J.numpy().T @ inv_expcov @ J.numpy())
postuncs = np.sqrt(postcov.diagonal())
post_df['RELUNC2'] = postuncs / post_df['POST'] * 100

# method 3 (MCMC)
postuncs_mcmc = np.std(prep.trafo(chain).numpy(), axis=0)
post_df['RELUNC_MCMC'] = postuncs_mcmc / post_df['POST_MCMC'] * 100


import matplotlib.pyplot as plt
# curchain = prep.trafo(chain[:,14]).numpy()
# scp.stats.skew(curchain)
# scp.stats.kurtosis(curchain)
# compute_effective_sample_size(curchain)
# plt.hist(curchain, bins=50)
# plt.show()

fis_df = post_df[post_df['NAME'].str.startswith('FIS')].reset_index(drop=True)
abs_df = post_df[post_df['NAME'].str.startswith('ABS')].reset_index(drop=True)
cap_df = fis_df.copy()
cap_df['NAME'] = cap_df['NAME'].str.replace('FIS', 'CA')
cap_df['POST'] = abs_df['POST'] - fis_df['POST']
cap_df['POST_MCMC'] = abs_df['POST_MCMC'] - fis_df['POST_MCMC']

red_post_df = post_df[~post_df['NAME'].str.startswith('ABS') & ~post_df['NAME'].str.startswith('HLF')]
red_post_df = pd.concat([red_post_df, cap_df], axis=0, ignore_index=True)
red_post_df


# comparison with experiments
# red_exp_dt['PRED'] = propagate(opt_params)
# red_exp_dt['RES'] = (red_exp_dt['PRED'] - red_exp_dt['InputValue']) / red_exp_dt['InputValue'] / (red_exp_dt['Uncertainty']/100)
# red_exp_dt

idcs = post_df[post_df.NAME.str.startswith('FIS')].index
for idx in idcs:
    curchain = prep.trafo(chain[:,idx]).numpy()
    curskew = scp.stats.skew(curchain)
    post_df.loc[idx, 'SKEW'] = curskew
    curkurt = scp.stats.kurtosis(curchain)
    post_df.loc[idx, 'KURT'] = curkurt


