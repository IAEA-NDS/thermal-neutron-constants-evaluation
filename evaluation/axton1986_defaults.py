"""Default parameter definitions for the Axton 1986 evaluation.

The 38 free parameters and their fitted values from Table 4,
plus fixed half-life reference data from Table 6 (Reich 1985).
"""

# The 38 free parameters and their fitted values from Table 4
DEFAULT_INITIAL_PARAMS = {
    ('SCA', 33): 12.1861, ('SCA', 35): 15.9828,
    ('SCA', 39): 7.8966,  ('SCA', 41): 12.1878,
    ('SCR', 33): 10.9169, ('SCR', 35): 14.2071,
    ('SCR', 39): 6.8000,  ('SCR', 41): 11.1717,
    ('ABS', 33): 576.2174, ('ABS', 35): 681.8303,
    ('ABS', 39): 1018.9667, ('ABS', 41): 1373.2025,
    ('FIS', 33): 530.6953, ('FIS', 35): 582.7836,
    ('FIS', 39): 747.6236, ('FIS', 41): 1011.8730,
    ('NUB', 33): 2.4950, ('NUB', 35): 2.4334,
    ('NUB', 39): 2.8822, ('NUB', 41): 2.9463, ('NUB', 52): 3.7676,
    ('WGA', 33): 0.9995, ('WGA', 35): 0.9789,
    ('WGA', 39): 1.0782, ('WGA', 41): 1.0442,
    ('WGF', 33): 0.9955, ('WGF', 35): 0.9774,
    ('WGF', 39): 1.0555, ('WGF', 41): 1.0445,
    ('CA', 40): 289.3296, ('CA', 42): 18.5144,
    ('CAP', 34): 95.8369,
    ('GC116', 39): 1.3265, ('GC116', 40): 1.0860,
    ('GC116', 41): 1.1085, ('GC116', 42): 1.1335,
    ('GA116', 39): 1.1846, ('GA116', 41): 1.1073,
}

# Fixed reference data (half-lives from Table 6, Reich 1985)
DEFAULT_FIXED_PARAMS = {
    ('HLF', 33): 1.592,   # 233U, ×1E5 years
    ('HLF', 34): 2.457,   # 234U, ×1E5 years
    ('HLF', 39): 2.411,   # 239Pu, ×1E4 years
    ('HLF', 41): 14.35,   # 241Pu, years
}
