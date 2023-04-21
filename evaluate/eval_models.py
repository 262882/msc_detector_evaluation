#!/usr/bin/env python3
"""
Determine mAP scores for detector
"""

from PIL import Image
import json
import numpy as np
import torch
import cv2
import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.join(sys.path[0], '../tooling/'))
from myloader import CocoDetection
from mymodels import nanodet, pretrained_yolov5s

display = True

evaluate_models = [
    ['../none/pretrained_yolov5s.none', pretrained_yolov5s, 0],
    ['../models/finedet_map93_416.onnx', nanodet, 416],
    ['../models/nanodet-plus-m-1.5x_416.onnx', nanodet, 416],
]

dataset_dirs = [
    "../data/coco_validation_sml_01_False_drill/",  # No match, no occl
    "../data/coco_validation_sml_01_False_matchdrill/",  # No occl
    "../data/coco_validation_sml_01_FalseTrue_matchdrill/"  # Full
]

for eval_model in evaluate_models:
        
    model_name = eval_model[0][eval_model[0].rfind('/')+1:-5]
    resFile = "../results/" + model_name + "_stats.json"
    model = eval_model[1](eval_model[0], eval_model[2])

    stats = {}
    x_loop_must_break = False
    for set_no, dir in enumerate(dataset_dirs):
        set_name = dir[dir[:-1].rfind('/')+1:-1]
        print('Evaluating', model_name, "using dataset", set_no)
        img_dir = os.path.abspath(dir)
        annotation = img_dir + "/annot.json"
        dataset = CocoDetection(root=img_dir, annFile=annotation)

        preds = []
        for idx, id in enumerate(dataset.ids):
            img, target = dataset[idx]

            # Prepare prediction 
            detections = model.forward(img)
            
            if len(detections)>0:
                for detect in detections:  # [x_min, y_min, x_max, y_max, score, class]
                    scores = detect[4]
                    boxes = detect[0:4]
                    labels = detect[5]
                    
                    # Consider only ball detections
                    if labels == 32:
                        labels = 37  # Convert prediction to coco
                        pass
                    else:
                        continue
                    
                    if (display):
                        # Display the resulting frame
                        frame = cv2.rectangle(np.asarray(img), (int(detect[0]), int(detect[1])), (int(detect[2]), int(detect[3])),(0, 255, 0), 2)
                        cv2.imshow('Frame',frame)

                        # Press Q on keyboard to  exit
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            x_loop_must_break = True
                            break

                    # from xyxy to xywh
                    boxes[2] = boxes[2]-boxes[0]
                    boxes[3] = boxes[3]-boxes[1]
                        
                    preds.append(
                        dict(
                            image_id = id,
                            category_id=labels,
                            bbox=boxes,
                            score=float(scores),
                        )
                    )
            if x_loop_must_break:
                break
                
        workingFile = "./detection_cocoresults_set.json"
        with open(workingFile, 'w') as out_file:
            json.dump(preds, out_file)

        cocoGt = COCO(annotation)
        cocoDt = cocoGt.loadRes(workingFile)
        os.remove(workingFile)

        cocoEval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
        cocoEval.params.imgIds  = dataset.ids  # Evaluate all images
        cocoEval.params.useCats = True
        cocoEval.params.catIds = 37  # Ball

        cocoEval.evaluate()
        cocoEval.accumulate()
        print(set_name)
        cocoEval.summarize()

        stat_sum = {
            'AP_{@[IoU=0.50:0.95]-all}': cocoEval.stats[0],
            'AP_{@[IoU=0.50:0.95]-small}': cocoEval.stats[3],
            'AP_{@[IoU=0.50:0.95]-medium}': cocoEval.stats[4],
        }
        stats[set_name] = stat_sum

        if x_loop_must_break:
            break

    detector_res = {model_name:stats}

    with open(resFile, 'w') as out_file:
        json.dump(detector_res, out_file)