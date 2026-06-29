import cv2
import json
import torch
import torchvision.transforms as transforms
import PIL.Image
import os
import numpy as np # <-- ADDED for Kalman Filter math
import trt_pose.coco
import trt_pose.models
from torch2trt import torch2trt
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import time 

# ---------------------------------------------------------
# NEW: Kalman Filter Class for Individual Joints
# ---------------------------------------------------------
class JointKalmanFilter:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        # State: [x, y, velocity_x, velocity_y]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # We only measure x and y (not velocity)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        
        # Physics model: next_x = x + velocity_x
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        
        # TUNE THESE FOR SMOOTHNESS VS LAG
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise       # Q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise # R
        
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            # First time seeing this joint, snap directly to it
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        
        # 1. Predict where the joint should be
        self.kf.predict()
        
        # 2. Update with the noisy TensorRT measurement
        measurement = np.array([[x], [y]], np.float32)
        estimated = self.kf.correct(measurement)
        
        # Return the smoothed x and y
        return float(estimated[0][0]), float(estimated[1][0])

# 1. Setup paths and device
TOPOLOGY_PATH = r'C:\Users\eehit\Desktop\RoboticsECSE\Research\MusicAI\Hand_Tracking\trt_pose_hand\preprocess\hand_pose.json'
WEIGHTS_PATH = r'C:\Users\eehit\Desktop\RoboticsECSE\Research\MusicAI\Hand_Tracking\trt_pose_hand\model\hand_pose_resnet18_att_244_244.pth'
ENGINE_PATH = r'C:\Users\eehit\Desktop\RoboticsECSE\Research\MusicAI\Hand_Tracking\trt_pose_hand\model\resnet18_hand_pose_224x224_trt.pth'
device = torch.device('cuda')

# 2. Load the hand topology
with open(TOPOLOGY_PATH, 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

# 3. Initialize the parsing and drawing tools
parse_objects = ParseObjects(
    topology, 
    cmap_threshold=0.25,  
    link_threshold=0.17,  
    cmap_window=5         
)
draw_objects = DrawObjects(topology)

# NEW: Initialize an array of 21 Kalman Filters (one for each joint)
# If it's too laggy, increase process_noise. If it's too jittery, increase measurement_noise.
filters = [JointKalmanFilter(process_noise=1e-3, measurement_noise=1e-2) for _ in range(num_parts)]

# 4. Engine Optimization / Loading
if not os.path.exists(ENGINE_PATH):
    print("Engine not found! Compiling PyTorch model to TensorRT...")
    print("This will maximize your RTX 4060 and take 3-10 minutes. Please wait...")
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    data = torch.zeros((1, 3, 224, 224)).cuda()
    model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), ENGINE_PATH)
    print("Compilation finished! Engine saved.")

print("Loading TensorRT engine...")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(ENGINE_PATH))
print("Engine loaded successfully!")

# 5. Define Image Preprocessing
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    global device
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# 6. Boot up the Webcam
print("Starting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    img_resized = cv2.resize(frame, (224, 224))
    data = preprocess(img_resized)
    
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    
    # counts: array of detected joints
    # peaks: tensor containing the (y, x) coordinates of those joints
    counts, objects, peaks = parse_objects(cmap, paf)
    
    # ---------------------------------------------------------
    # NEW: Apply Kalman Filter to the raw peaks before drawing
    # ---------------------------------------------------------
    for i in range(num_parts):
        # Check if this specific joint (c) was actually detected in this frame
        if int(counts[0, i]) > 0:
            # Extract raw (y, x) coordinates. 
            # Note: trt_pose normalizes coordinates between 0 and 1
            raw_y = float(peaks[0, i, 0, 0])
            raw_x = float(peaks[0, i, 0, 1])
            
            # Run through this joint's specific filter
            smooth_x, smooth_y = filters[i].update(raw_x, raw_y)
            
            # Overwrite the raw peaks with our smoothed coordinates
            peaks[0, i, 0, 0] = smooth_y
            peaks[0, i, 0, 1] = smooth_x
        else:
            # If the joint is hidden/lost, reset its filter so it doesn't 
            # wildly predict its location when it reappears
            filters[i].initialized = False
    # ---------------------------------------------------------
    
    # Draw the skeleton onto the resized image (now using smoothed peaks!)
    draw_objects(img_resized, counts, objects, peaks)
    
    display_img = cv2.resize(img_resized, (frame.shape[1], frame.shape[0]))
    cv2.imshow('TensorRT Hand Tracking', display_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()