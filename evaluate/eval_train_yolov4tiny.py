#!/usr/bin/env python3
"""
Determine mAP scores for detector
"""

import os
import sys

sys.path.append(os.path.join(sys.path[0], '../tooling/'))
from myeval import myevaluatemodels
from mymodels import yolov4tiny

display = False
verbose = False

evaluate_models = [
    ['../models/pretrained_yolov4-tiny.onnx', yolov4tiny, {"resolution":416, "num_classes":80}],
]

dataset_dirs = [
    "../data/coco_test_sml_01_False_drill/",  # No match, no occl
    "../data/coco_test_sml_01_False_matchdrill/",  # No occl
    "../data/coco_test_sml_01_FalseTrue_matchdrill/",  # Full
]

result_dir = "../results/train_yolov4tiny/"

myevaluatemodels(evaluate_models, dataset_dirs, result_dir, display=display, verbose=verbose)