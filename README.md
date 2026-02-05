# Visual-based Surgery Skill Evaluation System

### Dataset Source Statement

This project uses the SurgPose dataset, a publicly available dataset for articulated robotic surgical tool pose estimation and tracking. 
SurgPose provides semantic keypoints for surgical instrument videos,
facilitating visual-based pose estimation and trajectory analysis. 
For full details, refer to the official GitHub and original paper.

Dataset repository: https://github.com/zijianwu1231/SurgPose  
Paper: https://arxiv.org/abs/2502.11534


### Overview
Base on the operation video of monocular endoscope surgical robot,
this project detects and evaluates the motion trajectory of surgical instruments through computer vision technology,
and provides visualization and intelligent feedback agent modules to assist in surgical skill training.

## 1. Project Structure
```
VisualSurgerySystem/
├── data/                 
│   ├── dataset/         # original dataset, not uploaded
│   │   ├── 000000/
│   │   │   ├── regular/
│   │   │   │   ├── left_video.mp4
│   │   │   │   └── right_video.mp4
│   │   │   ├── bbox_left.json
│   │   │   ├── bbox_right.json
│   │   │   ├── keypoints_left.yaml
│   │   │   └── keypoints_right.yaml
│   │   ├── 000001/
│   │   ├── ...
│   │   └── 000033/
│   ├── meta/             
│   ├── sample/           
│   └── temp/             # intermediate output cache, not uploaded
├── output/               # result
│   ├── audit/            
│   ├── detections/        
│   ├── frames/
│   ├── tracks/
│   ├── metrics/
│   └── figures/ 
├── src/                  # all .py
├── README.md             # project description
├── requirements.txt      # list of library
└── .gitignore            # Git ignore configuration
```
#### XXXXXX[description]

## 2. Environment Configuration

### 2.1 Install dependencies
```commandline
pip install -r requirements.txt
```
### 2.2 Run modules
```commandline
python src/1_extract_frames.py
python src/2.1_detect_and_visualize.py
...
```

## 3. Module Description
```
│File Name                  │Description
│---------------------------│----------------------------------
│extract_frames.py          │Extract video frames                
│visualize_verify.py        │Detect and visualize instrument keypoints
│XXXXX.py
│
```

## 4. Project Highlights

- **YOLOv11**: surgical instrument keypoint detection
- **Kalman Filter**: denoising and smoothing the observed tool trajectories
- **Metrics Calculation**: measures smoothness, efficiency and XXXX of surgical movements 
- **Visualization**: outputs trajectory plots and radar chars for skill analysis
- **Teaching Agent**: generates feedback automatically by comparing user's performance with expert demonstrations

### * License

This project is for academic and research use only

## 6. XXXXX
### Others