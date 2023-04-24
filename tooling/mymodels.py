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
    
class cascade_classifier():

    def __init__(self, dir, res):
        self.model = cv2.CascadeClassifier(dir)

    def _preprocess(self, img):
        grey_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_16S
        kernel_size = 3
        edge_img = cv2.Laplacian(grey_img, ddepth, ksize=kernel_size)
        final = np.array((edge_img>100)*255, dtype='uint8')

        return final

    def _post_process(self, input):
        preds = []
        bboxes, rejectLevels, levelWeights = input
        if len(bboxes)>0:
            ind_max = np.argmax(levelWeights)
            bboxes = [bboxes[ind_max]]
            for (x, y, w, h) in bboxes:
                preds.append([float(x), float(y), float(x+w), float(y+h), float(1), int(32)])

        return preds

    def forward(self, img):
        scale_factor = 1.1  # how much the image size is reduced at each image scale
        min_neighbours = 1 # how many neighbors each candidate rectangle should have to retain it
        
        img = self._preprocess(img)

        output = self.model.detectMultiScale3(img, 
                                              scale_factor, 
                                              min_neighbours,
                                              outputRejectLevels = True
                                              )
        detections_pre = self._post_process(output)
            
        return detections_pre

class nanodet():

    def __init__(self, dir, res, num_classes=80):
        providers = ['CPUExecutionProvider']
        sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self.session = rt.InferenceSession(dir, sess_options=sess_options, providers=providers)
        self.outname = [i.name for i in self.session.get_outputs()] 
        self.inname = [i.name for i in self.session.get_inputs()]
        self.mod_res = res
        self.num_classes = num_classes
        self.strides = [8, 16, 32]

        self.grid_points = []
        self.grid_strides = []
        for stride in self.strides:
            grid_point = self._make_grid_point(
                (int(res/stride),
                 int(res/stride)),
                stride,
            )
            self.grid_points.extend(grid_point)
            self.grid_strides.extend([stride]*len(grid_point))  

    def _make_grid_point(self, grid_size, stride):
        grid_height, grid_width = grid_size

        shift_x = np.arange(0, grid_width) * stride
        shift_y = np.arange(0, grid_height) * stride

        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()

        cx = xv #  + 0.5 * (stride - 1)
        cy = yv #  + 0.5 * (stride - 1)

        return np.stack((cx, cy), axis=-1)

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
        preds = []
        class_ind = 32
        score_thr=0.05
        PROJECT = np.arange(8)

        class_scores = predict_results[:,:self.num_classes]
        bbox_predicts = predict_results[:,self.num_classes:]

        max_inds = np.argmax(class_scores[:,class_ind])  # Assume only one instance
        if class_scores[max_inds, class_ind] > score_thr:
            pred_inds = [[int(max_inds)]]
        else:
            pred_inds = []
        #pred_inds = np.argwhere(class_scores[:, class_ind] > score_thr)  # Process multiple instances

        if len(pred_inds)>0:
            for ind in pred_inds:
                ind = ind[0]
                x, y = self.grid_points[ind]
                stride = self.grid_strides[ind]

                bbox_max = bbox_predicts[ind]
                bbox_max = bbox_max.reshape(-1, 8)
                bbox_max = self._softmax(bbox_max, axis=1)
                bbox_max = np.dot(bbox_max, PROJECT).reshape(-1, 4)
                bbox_max *= stride
                l,t,r,b = bbox_max[0]

                preds.append([x-l, y-t, x+r, y+b, class_scores[ind, class_ind], class_ind])

        return preds

    def forward(self, img):
        preds = []
        norm_img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(norm_img, 
                        size = (self.mod_res, self.mod_res),
                        swapRB=False, crop=False) 
        inp = {self.inname[0]:blob}
        layer_output = self.session.run(self.outname, inp)[0][0]  

        detections_pre = self._post_process(layer_output)   
        for detection in detections_pre:

            x1, y1, x2, y2 = np.array(detection[:4])*norm_img.shape[0]/self.mod_res
            score = detection[-2]
            ind = detection[-1]
            preds.append([x1, y1, x2, y2, score, ind])
            
        return preds

class yolox():

    def __init__(self, dir, res, num_classes=80):
        providers = ['CPUExecutionProvider']
        sess_options = rt.SessionOptions()  # https://onnxruntime.ai/docs/performance/tune-performance.html
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self.session = rt.InferenceSession(dir, sess_options=sess_options, providers=providers)
        self.outname = [i.name for i in self.session.get_outputs()] 
        self.inname = [i.name for i in self.session.get_inputs()]
        self.mod_res = res
        self.num_classes = num_classes
        self.strides = [8, 16, 32]

        self.grid_points = []
        self.grid_strides = []
        for stride in self.strides:
            grid_point = self._make_grid_point(
                (int(res/stride),
                 int(res/stride)),
                stride,
            )
            self.grid_points.extend(grid_point)
            self.grid_strides.extend([stride]*len(grid_point))  

    def _make_grid_point(self, grid_size, stride):
        grid_height, grid_width = grid_size

        shift_x = np.arange(0, grid_width) * stride
        shift_y = np.arange(0, grid_height) * stride

        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()

        cx = xv #+ 0.5 * (stride - 1)
        cy = yv #+ 0.5 * (stride - 1)

        return np.stack((cx, cy), axis=-1)
    
    def _softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def _pre_process(self, img):
        blob = cv2.dnn.blobFromImage(img, size = (self.mod_res, self.mod_res), # Resolution multiple of 32
                                swapRB=False, crop=False) 
        return blob


    def _post_process(self, predict_results):
        preds = []
        class_ind = 32
        score_thr=0.05
        PROJECT = np.arange(8)

        class_scores = predict_results[:,6:]
        bbox_predicts = predict_results[:,:4]
        bbox_score = predict_results[:,5]

        max_inds = np.argmax(bbox_score)  # Assume only one instance
        top_classes = np.argmax(class_scores, axis=1)
        top_class_x = np.where(top_classes==32)[0]

        pred_inds = []
        if top_class_x.size > 0:
            top_ind_x = top_class_x[np.argmax(bbox_score[top_class_x])]

            if bbox_score[top_ind_x] > score_thr:
                pred_inds = [[int(top_ind_x)]]

        #pred_inds = np.argwhere(class_scores[:, class_ind] > score_thr)  # Process multiple instances

        if len(pred_inds)>0:
            for ind in pred_inds:
                ind = ind[0]
                x, y = self.grid_points[ind]
                stride = self.grid_strides[ind]
                p_x, p_y, p_w, p_h = bbox_predicts[ind,:4]
                l_w, l_h = stride*np.exp(np.array([p_w, p_h]))
                l_x, l_y = stride*np.array([p_x,p_y])

                preds.append([x+l_x-l_w//2, y+l_y-l_h//2, x+l_x+l_w//2, y+l_y+l_h//2, bbox_score[ind], np.argmax(class_scores[ind])])

        return preds

    def forward(self, img):
        preds = []
        norm_img = np.asarray(img)
        inp = {self.inname[0]:self._pre_process(norm_img)}
        layer_output = self.session.run(self.outname, inp)[0][0]  

        detections_pre = self._post_process(layer_output)   
        for detection in detections_pre:

            x1, y1, x2, y2 = np.array(detection[:4])*norm_img.shape[0]/self.mod_res
            score = detection[-2]
            ind = detection[-1]
            preds.append([x1, y1, x2, y2, score, ind])
            
        return preds
