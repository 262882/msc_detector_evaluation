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
    ['../models/yolov4-tiny/pretrained_yolov4-tiny.onnx', yolov4tiny, {"resolution":416, "num_classes":80}],
    #['../models/yolov4-tiny/ball-yolov4-tiny_320.onnx', yolov4tiny, {"resolution":320, "num_classes":1}],
    ['../models/yolov4-tiny/ball-yolov4-tiny_416.onnx', yolov4tiny, {"resolution":416, "num_classes":1}],
    #['../models/yolov4-tiny/ball-yolov4-tiny_480.onnx', yolov4tiny, {"resolution":480, "num_classes":1}],
]

dataset_dirs = [
    "../data/coco_validation_sml_01_False_matchdrill/",
]

result_dir = "../results/train_yolov4tiny/"

myevaluatemodels(evaluate_models, dataset_dirs, result_dir, display=display, verbose=verbose)