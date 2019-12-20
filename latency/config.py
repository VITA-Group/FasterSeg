# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'FasterSeg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

C.num_classes = 19
C.layers = 16
""" Latency Config """
C.mode = "student" # "teacher" or "student"
if C.mode == "teacher":
    ##### train teacher model only ####################################
    C.arch_idx = [0] # 0 for teacher
    C.branch = [2]
    C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    C.stem_head_width = [(1, 1)]
    C.load_path = "fasterseg" # path to the searched directory
    C.load_epoch = "last" # "last" or "int" (e.g. "30"): which epoch to load from the searched architecture
    C.Fch = 12
    C.image_height = 1024
    C.image_width = 2048
    C.save = "%dx%d_teacher"%(C.image_height, C.image_width)
elif C.mode == "student":
    ##### train student with KL distillation from teacher ##############
    C.arch_idx = [1] # 1 for student
    C.branch = [2]
    C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    C.stem_head_width = [(8./12, 8./12),]
    C.load_path = "fasterseg" # path to the searched directory
    C.teacher_path = "fasterseg" # where to load the pretrained teacher's weight
    C.load_epoch = "last" # "last" or "int" (e.g. "30")
    C.Fch = 12
    C.image_height = 1024
    C.image_width = 2048
    C.save = "%dx%d_student"%(C.image_height, C.image_width)

########################################
