# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from lib.datasets.motchallenge_pair import motchallenge as motchallenge_pair
from lib.datasets.motchallenge_single import motchallenge as motchallenge_single
from lib.datasets.motchallenge_sequential import motchallenge as motchallenge_sequential
from lib.datasets.citypersons_single import citypersons as citypersons_single
from lib.datasets.citypersons_pair import citypersons as citypersons_pair
from lib.datasets.motchallenge_tracking_pair import motchallenge as motchallenge_tracking_pair
from lib.datasets.motchallenge_appearance_pair import motchallenge as motchallenge_appearance_pair

import numpy as np

# set up for motchallenge
for database in ['motchallenge_single']:
    for phase in ['train', 'val']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: motchallenge_single(phase=phase, name=name))

for database in ['motchallenge_pair']:
    for phase in ['train', 'val']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: motchallenge_pair(phase=phase, name=name))

for database in ['motchallenge_sequential']:
    for phase in ['train', 'val']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: motchallenge_sequential(phase=phase, name=name))

# set up for citypersons
for database in ['citypersons_single']:
    for phase in ['train', 'val', 'test']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: citypersons_single(phase=phase, name=name))

for database in ['citypersons_pair']:
    for phase in ['train', 'val', 'test']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: citypersons_pair(phase=phase, name=name))

for database in ['motchallenge_tracking_pair']:
    for phase in ['train', 'val']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: motchallenge_tracking_pair(phase=phase, name=name))

for database in ['motchallenge_appearance_pair']:
    for phase in ['train', 'val']:
        name = database + '_' + phase
        __sets[name] = (lambda name=name, phase=phase: motchallenge_appearance_pair(phase=phase, name=name))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
