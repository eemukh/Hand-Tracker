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
#parse_objects = ParseObjects(topology)
parse_objects = ParseObjects(
    topology, 
    cmap_threshold=0.12,  # Minimum confidence for a joint (increase to filter noise)
    link_threshold=0.10,  # Minimum confidence for the "bone" connecting joints
    cmap_window=11         # The pixel radius used to find the highest confidence peak
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

VIDEO_PATH = r'C:\Users\eehit\Desktop\RoboticsECSE\Research\MusicAI\Hand_Tracking\VideoSamples\DnnPianoTest.mov'

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: File not found at {VIDEO_PATH}")
    # List files in the directory to see if there's a typo
    dir_path = os.path.dirname(VIDEO_PATH)
    print(f"Files in directory: {os.listdir(dir_path)}")
else:
    print(f"File confirmed at: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: OpenCV could not open the video file. Check codec/format.")
    # If this fails, try: cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
else:
    print("Successfully opened video file.")

# Get video properties
fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video <= 0: fps_video = 30 
frame_delay = int(1000 / fps_video)

print(f"Processing video: {VIDEO_PATH} at {fps_video} FPS")

window_name = 'TensorRT Hand Tracking'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allows manual resizing/fullscreen
# Optional: Set a starting size that's larger than 224
cv2.resizeWindow(window_name, 1280, 720)

while cap.isOpened():
    start_time = time.time() # Track start of processing
    
    ret, frame = cap.read()
    if not ret:
        print("End of video file or error loading frame.")
        break
    
    # --- Processing Logic (Same as before) ---
    img_resized = cv2.resize(frame, (224, 224))
    data = preprocess(img_resized)
    
    with torch.no_grad():
        cmap, paf = model_trt(data)
    
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(img_resized, counts, objects, peaks)
    # -----------------------------------------

    # Resize up to HD or your screen's resolution
    # INTER_CUBIC looks better for upscaling than the default linear
    display_img = cv2.resize(img_resized, (1280, 720), interpolation=cv2.INTER_CUBIC)

    # # Show the large image
    cv2.imshow(window_name, display_img)

    # # Display results
    # cv2.imshow('TensorRT Video Inference', img_resized)
    
    # Calculate how long processing took to maintain real-time playback
    # If processing was faster than the frame rate, wait the difference
    elapsed = int((time.time() - start_time) * 1000)
    wait_time = max(1, frame_delay - elapsed)
    
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()