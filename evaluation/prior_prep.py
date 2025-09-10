import tensorflow as tf


# independent parameters for fitting
# for the microscopic case (tuple_combis_mic)
# and microscopic + macroscopic case (tuple_combis_mac)

tuple_combis_mic = tuple(
    (x, y) for x in ('SCA', 'SCR', 'ABS', 'FIS', 'NUB')
    for y in (33, 35, 39, 41)
)

tuple_combis_mac = tuple(
    (x, y) for x in ('SCA', 'SCR', 'ABS', 'FIS', 'NUB', 'WGA', 'WGF')
    for y in (33, 35, 39, 41)
)


add_tuple_combis = tuple()
add_tuple_combis += (('NUB', 52),)
add_tuple_combis += (('HLF', 33),)
add_tuple_combis += (('HLF', 34),)
add_tuple_combis += (('HLF', 39),)
add_tuple_combis += (('HLF', 41),)
add_tuple_combis += (('CAP', 34),)

tuple_combis_mic += add_tuple_combis
tuple_combis_mac += add_tuple_combis


# mapping of quantities to indices in a vector

reac_map_mic = {
    n: i for i, n in enumerate(tuple_combis_mic)
}

reac_map_mac = {
    n: i for i, n in enumerate(tuple_combis_mac)
}


# start values for the optimization process

startvals_map = {
    ('SCA',33):    12.307300,
    ('SCA',35):    16.243115,
    ('SCA',39):     8.010771,
    ('SCA',41):    12.037969,
    ('SCR',33):    11.127555,
    ('SCR',35):    14.539451,
    ('SCR',39):     7.196032,
    ('SCR',41):    10.957713,
    ('ABS',33):   575.321483,
    ('ABS',35):   681.206917,
    ('ABS',39):  1014.286652,
    ('ABS',41):  1379.294684,
    ('FIS',33):   530.547892,
    ('FIS',35):   583.096931,
    ('FIS',39):   745.356348,
    ('FIS',41):  1015.927048,
    ('NUB',33):     2.490351,
    ('NUB',35):     2.428843,
    ('NUB',39):     2.880714,
    ('NUB',41):     2.944448,
    ('NUB',52):     3.766402,
    ('CAP',34):     95.8369,
}


def assign_startvals(startvals_vec, startvals_map, reac_map):
    for k, idx in reac_map.items():
        if k in startvals_map:
            startvals_vec[idx] = startvals_map[k]
