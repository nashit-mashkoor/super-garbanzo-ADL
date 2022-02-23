import os
import cv2
import time
import json
import torch
import requests
import torch2trt
import PIL.Image
import numpy as np
import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule

import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import jetson.inference
import jetson.utils

class TRTDetector():

    def __init__(self):
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

        self.num_parts = len(human_pose['keypoints'])
        self.num_links = len(human_pose['skeleton'])

        self.WIDTH = 256
        self.HEIGHT = 256

        self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()

        self.MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
        self.OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        self.model = trt_pose.models.densenet121_baseline_att(self.num_parts, 2 * self.num_links).cuda().eval()

        if not os.path.exists(self.OPTIMIZED_MODEL):
            self.model.load_state_dict(torch.load(self.MODEL_WEIGHTS))
            self.model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(self.model_trt.state_dict(), self.OPTIMIZED_MODEL)

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')

        self.parse_objects = ParseObjects(self.topology)
        self.X_compress = 1280.0 / self.WIDTH * 1.0
        self.Y_compress = 720.0 / self.HEIGHT * 1.0
        self.draw_objects = DrawObjects(self.topology)


    def preprocess(self, image):
        self.device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]


    def get_keypoint(self, humans, hnum, peaks, w=0, h=0):
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]

        for j in range(C-1):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
            else:
                peak = (j, None, None)
                kpoint.append(peak)

        return kpoint


    def get_bb(self, kp_set, width_height):
        bb_list = []
        for i in range(kp_set.shape[0]):
            x = []
            y = []
            for j in range(kp_set[i].shape[0]):
                if kp_set[i, j, 0] is not None:
                    x.append(kp_set[i, j, 0])
                if kp_set[i, j, 1] is not None:
                    y.append(kp_set[i, j, 1])
            if len(x) > 0 and len(y) > 0:
                # keypoint bounding box
                x1, x2 = np.min(x), np.max(x)
                y1, y2 = np.min(y), np.max(y)
                if x2 - x1 < 5.0/width_height[0]:
                    x1 -= 2.0/width_height[0]
                    x2 += 2.0/width_height[0]
                if y2 - y1 < 5.0/width_height[1]:
                    y1 -= 2.0/width_height[1]
                    y2 += 2.0/width_height[1]
                bb_list.append(((int(round(x1 * self.WIDTH * X_compress)), int(round(y1 * self.HEIGHT * Y_compress))), (int(round(x2 * self.WIDTH * X_compress)), int(round(y2 * self.HEIGHT * Y_compress)))))
        if len(bb_list) == 0:
            return []
        return bb_list


    def changeXtoY(img, array):
        w, h = img.size
        updatedArray = []
        index = 0
        for elem in array:
            updatedArray.append([elem[2], elem[1]])
            index += 1
        return updatedArray


    def getTrtPose(self, img, org):
        data = self.preprocess(img)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        human_keypoints = []
        for i in range(counts[0]):
            kpoint = self.get_keypoint(objects, i, peaks, org.size[0], org.size[1])
            kp = self.changeXtoY(org, kpoint)
            human_keypoints.append(kp)

        return human_keypoints


    def get_trt_keypoints(self, src, width_height):
        start = time.time()
        pilimg = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pilimg = PIL.Image.fromarray(pilimg)
        orgimg = pilimg.copy()
        image = cv2.resize(src, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)

        img = image.copy()
        w, h = orgimg.size
        Points = self.getTrtPose(img, orgimg)
        Points = np.array(Points)
        bboxes = self.get_bb(Points, width_height)
        for i in range(Points.shape[0]):
            for j in range(Points[i].shape[0]):
                if Points[i, j, 0] is not None:
                    Points[i, j, 0] = Points[i, j, 0]/w
                if Points[i, j, 0] is None:
                    Points[i, j, 0] = Points[i, j, 0]
                if Points[i, j, 1] is not None:
                    Points[i, j, 1] = Points[i, j, 1]/h
                if Points[i, j, 1] is None:
                    Points[i, j, 1] = Points[i, j, 1]

        end = time.time()
        return Points, bboxes, width_height

class NvidiaDetector():
    def __init__(self, network="ssd-mobilenet-v2", threshold=0.95):
        self.net = jetson.inference.detectNet(network, threshold)
    
    def detect(self, img):
        img = jetson.utils.cudaFromNumpy(img)
        detections = self.net.Detect(img) 
        height, width = img.shape[:2]
        width_height = (width, height) 
        classes = []
        boxes = []
        scores = []
        for detect in detections:

            print(detect)
            if detect.ClassID == 1:
                # Temporary
                scores.append(detect.Confidence)
                classes.append(detect.ClassID)
                boxes.append([detect.Left, detect.Top, detect.Right, detect.Bottom])
        return classes, boxes, scores

    

def main():
    cam = cv2.VideoCapture('test1.mp4')
    network="ssd-mobilenet-v2"
    threshold=0.95
    net = jetson.inference.detectNet(network, threshold)
    while True:
        success, image = cam.read()
        if not success:
            break
        height, width = image.shape[:2]
        print(type(image))
        width_height = (width, height) 
        img = jetson.utils.cudaFromNumpy(image)
        detections = net.Detect(img) 
        for detect in detections:
            cv2.rectangle(image,(int(detect.Left),int(detect.Top)),(int(detect.Left+detect.Width),int(detect.Top+detect.Height)),(0,255,0),2)
        cv2.imshow('Test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def get_person_bboxes(inp_img):
    pass
    

def main_static():
    detector= NvidiaDetector()
    image = cv2.imread('download2.jpeg')
    print((image))
    height, width = image.shape[:2]
    width_height = (width, height) 
    classes, boxes, scores = detector.detect(image)
    print(classes,boxes, scores)

main_static()