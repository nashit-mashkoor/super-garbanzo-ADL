from cProfile import label
from functools import partial
import numpy as np
from PIL import Image
import cv2
import torch
import time
from detector import TRTDetector, NvidiaDetector

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection # Another option is slowfast_r50_detection
from visualization import VideoVisualizer
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' # or 'cpu'
video_model = slow_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)

# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)

detector= NvidiaDetector()

# Window name in which image is displayed
window_name = 'Image'
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, detector):
    classes, boxes, scores = detector.detect(image)
    if len(boxes) != 0:
        boxes = torch.FloatTensor(boxes)

    return boxes

def ava_inference_transform(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes

# Create an id to label name mapping
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
# # Load the video
# encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path('test3.mp4')
# video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.95)

# print('Completed loading encoded video.')

# # Video predictions are generated at an internal of 1 sec from 90 seconds to 100 seconds in the video.
clip_duration = 2 # Duration of clip used for each inference step.
time_stamp_range = range(1, 2) # time stamps in video for which clip is sampled.

def get_prediction(encoded_vid, clip_duration, time_stamp_range):
    gif_imgs = []
    predictions = []

    for time_stamp in time_stamp_range:
        print("Generating predictions for complete video clip")

        # Generate clip around the designated time stamps
        inp_imgs = encoded_vid.get_clip(
            time_stamp - clip_duration/2.0, # start second
            time_stamp + clip_duration/2.0  # end second
        )
        
        inp_imgs = inp_imgs['video']
        
        
        # Generate people bbox predictions using Nvidia SSD v1 Inception
        # We use the the middle image in each clip to generate the bounding boxes.
        inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
        inp_img = inp_img.permute(1,2,0)


        predicted_boxes = get_person_bboxes(inp_img, detector)
        
        if len(predicted_boxes) == 0:
            print("Skipping clip no frames detected at time stamp: ", time_stamp)
            continue

        # Preprocess clip and bounding boxes for video action recognition.
        inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
        # Prepend data sample id for each bounding box.
        # For more details refere to the RoIAlign in Detectron2
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)

        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        preds = video_model(inputs.unsqueeze(0).to(device), inp_boxes.to(device))


        preds= preds.to('cpu')
        # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
        preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
        class_pred = np.argmax(preds.detach().numpy())
        
        if class_pred in allowed_class_ids:
            predictions.append(label_map[class_pred])
        else:
            predictions.append(None)

            # Plot predictions on the video and save for later visualization.
        inp_imgs = inp_imgs.permute(1,2,3,0)
        inp_imgs = inp_imgs/255.0
        out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
        gif_imgs += out_img_pred

    return predictions, gif_imgs

# # suppose you want to start reading from frame no 500
# frame_set_no = 500 
cap = cv2.VideoCapture('test.mp4')
test_ret, test_frame=cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
demo_out = cv2.VideoWriter(f'sample.avi', fourcc, 20.0, (600,  900))
final_pred = []
if not test_ret:
    print('Video file not opened')
# Define the codec and create VideoWriter object
frame_chunk = 40 # Rounds about 2 sec video Depends on the platform the system is running
while True:
    # Temporary output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output.avi', fourcc, 20.0, (600,  900))
    current_frame = []
    for i in range(frame_chunk):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.resize(frame, (600,  900))
        current_frame.append(frame)
        out.write(frame)
    out.release()
    
    try:
        # Predictions
        encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path('output.avi')
        video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.95)
        predictions, gif_imgs = get_prediction(encoded_vid, clip_duration, time_stamp_range)
        height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]
        print(predictions)
    except:
        print('Frame Captured Failed')
    
    c = Counter(predictions)
    # Visualised output
    if len(predictions) == 0:
        label = 'No activity detected'
    else:
        label = 'Activity Detected: '+str(c.most_common(1)[0][0])
        print(label)
    for f in current_frame:
        vis_frame = cv2.putText(f, label, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        # cv2.imshow('Output', vis_frame)
        demo_out.write(vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
demo_out.release()
cv2.destroyAllWindows()
