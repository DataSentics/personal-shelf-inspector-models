import os
from argparse import Namespace

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path
from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    plot_one_box, strip_optimizer, set_logging, increment_dir
from utils.torch_utils import select_device, load_classifier, time_synchronized

from detect import detect

# TODO: check and/or edit in a way that detect returns list of predictions if lit of images on input


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class Model():
    def __init__(self, weights, device="cpu"):
        self.weights = weights
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.save_txt = False
        self.view_img = False
        self.augment = False
        self.update = False
        self.device = select_device(device)
        self.img_size = 640
        self.classes = None
        self.agnostic_nms = False
        self.save_img = False
        self.name = ""       
        self.webcam = False
        self.half = self.device.type != 'cpu'
        self.save_dir = Path('runs/detect')
        self.load()
        
    def load(self):
 
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()
                                    
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        # print("Model loaded")
        
    def detect(self,
               images,
               min_confidence=0.0,
               mode="return"):
        """ 
        :param images: a list of images in np.array, RGB format.
        :param min_confidence: a threshold for minimum acceptable confidence
            all detections with lower confidence will be discarded
        :param mode: str wirh desired mode. Modes available: 'return' (returns detections), 'draw' (draws detections and returns image)

        returns: dict containing:
            rois: np.array of shape [n_detection,4] each row: [rows_top_left, col_top_left, rows_bottom_right, col_bottom_row],
            cls: np. array of class ids,
            conf: np.array of confidences  
        """
        
        assert mode in ["return", "draw"], "Parameter mode should be one of: 'return', 'draw', 'return_rois'"
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        for im0 in images:
            img = letterbox(im0, new_shape=self.img_size)[0]
            ## n channels as first dimension
            
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]

            # Apply nonmaxsupression
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, img0)
            print(f"{round(t2-t1,3)} s")
                             
            # Process detections
            for i, det in enumerate(pred):  # detections per image -> needed? we process one image 
                cnt = 0

                save_path = str(self.save_dir )
                txt_path = str(self.save_dir / 'labels' ) 
                # s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    #return det
                    # transform detections
                    rois = []
                    conf = []
                    cls = []
                    for *xyxy, con, cl in reversed(det):
                        if con >= min_confidence:
                            # reorder, [rows, cols, rows, cols]
                            rois.append([xyxy[1],xyxy[0],xyxy[3],xyxy[2]]) 
                            conf.append(con)
                            cls.append(int(cl))
                        
                    pred = {
                        "rois":np.array(rois),
                        "conf":np.array(conf),
                        "cls":np.array(cls)
                           }

                if mode == "return":
                    return pred
                elif mode == "return_rois":
                    return pred["rois"]
                elif mode == "draw":
                    return self.draw_detections(im0, det)
                
    def draw_detections(self, img, det):
        im = np.array(deepcopy(img))
        if (det == None) or (len(det) == 0):
            print("No detections found!")
            
        for *xyxy, con, cls in reversed(det):
            label = '%s %.2f' % (self.names[int(cls)], con)
            plot_one_box(xyxy, im, label=label, color=self.colors[int(cls)], line_thickness=3)
#         for i,d in enumerate(det["rois"]):
#             cl = det["cls"][i]
#             conf = det["conf"][i]
#             label = '%s %.2f' % (self.names[cl], conf)
#             # TODO: does not use line_thinkness because it failes with mysterious TypeError
#             plot_one_box(d, im, label=label, color=self.colors[int(cl)])       
        return im
                
          
        
    
