#!/usr/bin/env python3
"""
Determine mAP scores for detector
"""

import os
import sys

sys.path.append(os.path.join(sys.path[0], '../tooling/'))
from myeval import myevaluatemodels
from mymodels import nanodet, pretrained_yolov5s, cascade_classifier, yolox

display = False

evaluate_models = [
    ['../none/pretrained_yolov5s.none', pretrained_yolov5s, {}],
    ['../models/pretrained_nanodet-plus-m_416.onnx', nanodet, {"resolution":416, "num_classes":80}],
    ['../models/ball-nanodet-plus-m_416.onnx', nanodet, {"resolution":416, "num_classes":80}],
    ['../models/ballcascade_10_0.35.xml', cascade_classifier, {"scale_factor":1.04, "min_neighbours":2}],
]

dataset_dirs = [
    "../data/coco_test_sml_01_False_drill/",  # No match, no occl
    "../data/coco_test_sml_01_False_matchdrill/",  # No occl
    "../data/coco_test_sml_01_FalseTrue_matchdrill/",  # Full
]

result_dir = "../results/"

myevaluatemodels(evaluate_models, dataset_dirs, result_dir, display=display, verbose=False)