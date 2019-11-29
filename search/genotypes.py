from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip',
    'conv',
    'conv_downup',
    'conv_2x',
    'conv_2x_downup',
]
