import cv2
import json
import torch
import torchvision.transforms as transforms
import PIL.Image
import os
import trt_pose.coco
import trt_pose.models
from torch2trt import torch2trt
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import time 

# 1. Setup paths and device
# Note: Ensure these paths match your folder structure!
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
#parse_objects = ParseObjects(topology)q
parse_objects = ParseObjects(
    topology, 
    cmap_threshold=0.25,  # Minimum confidence for a joint (increase to filter noise)
    link_threshold=0.20,  # Minimum confidence for the "bone" connecting joints
    cmap_window=3         # The pixel radius used to find the highest confidence peak
)
draw_objects = DrawObjects(topology)

# 4. Engine Optimization / Loading
if not os.path.exists(ENGINE_PATH):
    print("Engine not found! Compiling PyTorch model to TensorRT...")
    print("This will maximize your RTX 4060 and take 3-10 minutes. Please wait...")
    
    # Load the base PyTorch model
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    
    # Create dummy data to trace the network
    data = torch.zeros((1, 3, 224, 224)).cuda()
    
    # Compile the engine
    model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    
    # Save the engine so we never have to compile it again
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
    # OpenCV uses BGR, PyTorch uses RGB
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
    
    # --- 1. NEW: Crop a perfect square from the center of the webcam ---
    h, w = frame.shape[:2]
    size = min(h, w) # Find the shortest side to make our square
    
    # Calculate the top-left corner to center the square
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    
    # Slice the image array to grab just that center square
    square_frame = frame[y1:y1+size, x1:x1+size]
    
    # --- 2. Resize our undistorted square to the 224x224 model requirement ---
    img_resized = cv2.resize(square_frame, (224, 224))
    
    # Format the data for the GPU
    data = preprocess(img_resized)
    
    # Run Inference
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    
    # Parse and Draw
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(img_resized, counts, objects, peaks)
    
    # --- 3. NEW: Scale up evenly for display (Do NOT stretch back to rectangular!) ---
    # We resize it to 600x600 so it is large enough to see, but remains a square.
    display_img = cv2.resize(img_resized, (600, 600), interpolation=cv2.INTER_CUBIC)
    
    # Show the video feed
    cv2.imshow('TensorRT Hand Tracking', display_img)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()