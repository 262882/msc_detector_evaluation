#!/usr/bin/env python3
"""
Determine mAP scores for detector
"""

import os
import sys
import glob

sys.path.append(os.path.join(sys.path[0], '../tooling/'))
from myeval import myevaluatemodels
from mymodels import cascade_classifier

display = False
verbose = True

models_list = glob.glob(os.path.join('../models/vj/trash/','*.xml'))
evaluate_models = [[model, cascade_classifier,  {"scale_factor":1.1, "min_neighbours":3}] for model in models_list] 

#evaluate_models = [
#    ['../models/ballcascade_8_0.25.xml', cascade_classifier, {"scale_factor":1.1, "min_neighbours":1}],
#]

dataset_dirs = [
    "../data/coco_train_sml_01_False_matchdrill/",
    "../data/coco_validation_sml_01_False_matchdrill/",
]

result_dir = "../results/train_vj/"

myevaluatemodels(evaluate_models, dataset_dirs, result_dir, display=display, verbose=verbose)
