#!/usr/bin/env python3
"""
Determine mAP scores for detector
"""

import os
import sys
import glob

sys.path.append(os.path.join(sys.path[0], '../tooling/'))
from myeval import myevaluatemodels
from mymodels import nanodet

display = False
verbose = True

models_list = glob.glob(os.path.join('../models/nanodet/','*.onnx'))
evaluate_models = [[model, nanodet, {"resolution":int(model[-8:-5]), "num_classes":80}] for model in models_list] 

dataset_dirs = [
    "../data/coco_validation_sml_01_False_matchdrill/",
]

result_dir = "../results/train_nanodet/"
myevaluatemodels(evaluate_models, dataset_dirs, result_dir, display=display, verbose=verbose)
