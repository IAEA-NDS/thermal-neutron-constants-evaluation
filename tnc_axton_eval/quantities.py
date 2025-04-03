class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        super().__setattr__('_varnames', set(adict))

    def __setattr__(self, name, value):
        self._varnames.add(name)
        super().__setattr__(name, value)

    def is_valid(self, name):
        return name in self._varnames


primary_quantities = ('ABS', 'FIS', 'SCA', 'SCR', 'WGA', 'WGF', 'GC116', 'GA116', 'HLF', 'NUB')
primary_mt_map = {v: k for k, v in enumerate(primary_quantities, start=20)} 

# define selectors for primary quantities
def prepare_funcs(getter, override=None):
    primary_quantities = (
        'ABS', 'FIS', 'SCA', 'SCR', 'WGA',
        'WGF', 'GC116', 'GA116', 'HLF', 'NUB'
    )
    funcs = {q: lambda r, q=q: getter(q, r) for q in primary_quantities}
    f = Bunch(funcs)
    # define funcs for derived quantities
    f.FA = lambda r: f.ABS(r) - f.WGA(r)
    f.FF = lambda r: f.FIS(r) - f.WGF(r)  
    f.CA = lambda r: f.ABS(r) - f.FIS(r)
    f.CAP = lambda r: (f.ABS(r) * f.WGA(r)) - (f.FIS(r)*f.WGF(r))  # check with Gilles if type in Axton report and GA == WGA 
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
    f.F3HLF = lambda: 1 + 0.00395 * (14.5 - f.HLF(41)) / (14.5-12.9)
    return f
