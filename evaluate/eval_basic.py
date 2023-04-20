#!/usr/bin/env python3
"""
Determine mAP scores for detector
"""

from PIL import Image
import torch
import json
import numpy as np
import cv2
import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.join(sys.path[0], '../tooling/'))
from myloader import CocoDetection

display = True

resFile = "../results/basic_stats.json"

dataset_dirs = [
    "../data/coco_validation_sml_01_False_drill/",  # No match, no occl
    "../data/coco_validation_sml_01_False_matchdrill/",  # No occl
    "../data/coco_validation_sml_01_FalseTrue_matchdrill/"  # Full
]

model_name = 'yolov5s'
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

stats = {}
x_loop_must_break = False
for set_no, dir in enumerate(dataset_dirs):
    set_name = dir[dir[:-1].rfind('/')+1:-1]
    print('Evaluating dataset', set_no)
    img_dir = os.path.abspath(dir)
    annotation = img_dir + "/annot.json"
    dataset= CocoDetection(root=img_dir, annFile=annotation)

    preds = []
    for idx, id in enumerate(dataset.ids):
        img, target = dataset[idx]

        # Prepare prediction 
        result = model(img)
        detections = result.xyxy[0]  # Batch size of 1
        
        if len(detections)>0:
            for detect in detections:  # [x_min, y_min, x_max, y_max, score, class]
                scores = detect[4].tolist()
                
                boxes = detect[0:4].type(torch.int64).tolist()
                labels = detect[5].type(torch.int64).tolist()
                
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
                        image_id = target[0]['image_id'],
                        category_id=labels,
                        bbox=boxes,
                        score=round(scores,3),
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