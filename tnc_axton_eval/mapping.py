import pandas as pd
import numpy as np
import lark
from expdata import exp_dt
from quantity_grammar import grammar

# get source indices
# get target indices
# called atomic propagate constructor with source and target indices


# define source selectors

inpvar = np.arange(100)
dt = pd.DataFrame.from_records([
    {'REAC': 'MT:20-R1:10'},
    {'REAC': 'MT:20-R1:11'},
    {'REAC': 'MT:24-R1:11'},
])


reac_parser = lark.Lark(grammar)


reacstr = exp_dt.iloc[0]['MeasureFunc']


reac_parser.parse(reacstr)










