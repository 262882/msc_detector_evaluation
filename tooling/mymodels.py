"""
Vision models 
"""

import onnxruntime as rt
import cv2
import numpy as np
import torch

class pretrained_yolov5s():

    def __init__(self, dir, res):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def forward(self, img):
        result = self.model(img)
        detections = result.xyxy[0]  # Assume batch size of 1
        return (detections.cpu()).tolist()

class nanodet():

    def __init__(self, dir, res):
        providers = ['CPUExecutionProvider']
        sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self.session = rt.InferenceSession(dir, sess_options=sess_options, providers=providers)
        self.outname = [i.name for i in self.session.get_outputs()] 
        self.inname = [i.name for i in self.session.get_inputs()]
        self.mod_res = res
        self.num_classes = 80

    def _normalize(self, in_img): 
        in_img = np.asarray(in_img).astype(np.float32) / 255
        MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
        out_img = in_img - MEAN / STD
        return out_img
    
    def _softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _post_process(self, predict_results):
        detections = []
        class_scores = predict_results[:,:self.num_classes]
        bbox_predicts = predict_results[:,self.num_classes:]
        PROJECT = np.arange(8)

        max_ind = np.argmax(class_scores[:,32])
        bbox_predict = bbox_predicts[max_ind]

        bbox_predict = bbox_predict.reshape(-1, 8)
        bbox_predict = self._softmax(bbox_predict, axis=1)
        bbox_predict = np.dot(bbox_predict, PROJECT).reshape(-1, 4)
        stride = 8
        bbox_predict *= stride
        bbox_predict = np.append(bbox_predict[0], max_ind)
        return bbox_predict

        """
        bboxes, scores, class_ids = self._get_bboxes_single(
            class_scores,
            bbox_predicts,
            1,
            rescale=False,
        )
        """
        #return bboxes.astype(np.int32), scores, class_ids


    def forward(self, img):
        preds = []
        norm_img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(norm_img, 
                        size = (self.mod_res, self.mod_res),
                        swapRB=False, crop=False) 
        inp = {self.inname[0]:blob}
        layer_output = self.session.run(self.outname, inp)[0][0]  

        detections = self._post_process(layer_output)        

        coco_ind = 32  # Ball
        stride = 8  # Stride: 8,16,32
        cell_count = self.mod_res//stride

        #max_inds = np.argwhere(layer_output[:(cell_count)**2, coco_ind] > 0.2)
        max_inds = np.argwhere(layer_output[:, coco_ind] > 0.5)

        l,t,r,b = detections[:4]*norm_img.shape[0]/self.mod_res
        ind = detections[4]
        x = (ind%cell_count)*norm_img.shape[0]/cell_count
        y = (ind//cell_count)*norm_img.shape[0]/cell_count

        preds.append([x-l, y-t, x+r, y+b, layer_output[int(ind), coco_ind], coco_ind])

        """
        for ind in max_inds:
            ind=ind[0]

            # Find pixel representation
            x = (ind%cell_count+0.5)*norm_img.shape[0]/cell_count
            y = (ind//cell_count+0.5)*norm_img.shape[0]/cell_count

            # Find max prediction as result
            box = np.max(np.reshape(layer_output[ind,-32:],(4,8)),axis=1)
            l,t,r,b = (box*norm_img.shape[0]/cell_count)
            print("lrtb",l,r,t,b)

            preds.append([x+l, y+t, x-r, y-b, layer_output[ind, coco_ind], coco_ind, x, y])
        """

        return preds

class yolox():
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, dir, res):
        providers = ['CPUExecutionProvider']
        sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self.session = rt.InferenceSession(dir, sess_options=sess_options, providers=providers)
        self.outname = [i.name for i in self.session.get_outputs()] 
        self.inname = [i.name for i in self.session.get_inputs()]
        self.res = res

    def _normalize(self, in_img): 
        in_img = np.asarray(in_img).astype(np.float32) / 255
        MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
        out_img = in_img - MEAN / STD
        return out_img

    def forward(self, img):
        detections = []
        norm_img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(norm_img, 
                        size = (self.res, self.res),
                        swapRB=False, crop=False) 
        inp = {self.inname[0]:blob}
        layer_output = self.session.run(self.outname, inp)[0][0]          

        coco_ind = 32  # Ball
        stride = 8  # Stride: 8,16,32
        cell_count = self.res//stride

        #max_inds = np.argwhere(layer_output[:(cell_count)**2, coco_ind] > 0.2)
        max_inds = np.argwhere(layer_output[:, coco_ind] > 0.5)

        for ind in max_inds:
            ind=ind[0]

            # Find pixel representation
            x = (ind%cell_count+0.5)*norm_img.shape[0]/cell_count
            y = (ind//cell_count+0.5)*norm_img.shape[0]/cell_count

            # Find max prediction as result
            box = np.max(np.reshape(layer_output[ind,-32:],(4,8)),axis=1)
            l,t,r,b = (box*norm_img.shape[0]/cell_count)
            print("lrtb",l,r,t,b)

            detections.append([x-l, y-t, x+r, y+b, layer_output[ind, coco_ind], coco_ind, x, y])

        return detections
