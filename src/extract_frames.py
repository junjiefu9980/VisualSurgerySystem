import json
import argparse
import keyword
import subprocess
import yaml
import csv
import random
from pathlib import Path
from typing import Optional, Dict, List

import os
import cv2


# ===============================================
# Find the root path in Project   -> Reproducible if structure and environment are same
# ================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "dataset"
TEMP_ROOT = REPO_ROOT / "data" / "temp" / "frames"

# 1. Find all regular mp4 files in local path
# ==============================================
def find_videos(root_dir):
    print("\n", "="*20, f"Step 1: Preparing dataset", "="*20)
    
    if not root_dir.exists():
        print (f"[Warning] {root_dir} does not exist")
        return None

    # Filter mp4 only in "regular" for each case
    all_mp4s = list(root_dir.glob("**/regular/*.mp4"))
    cases = {p.parents[1].name for p in all_mp4s}
    print(f"[DONE] Found {len(all_mp4s)} mp4 videos, {len(cases)} case folders.")

    # Check the path of a random video
    check_path = random.choice(all_mp4s)
    print("(test) Check the path of a random video:")
    print(f"       Case name: {check_path.parts[-3]}")
    print(f"       Video name: {check_path.name}")
    print(f"       Local path: {check_path}")

    return all_mp4s


# 1.2 Tools: frame to list
# ===========================================
def to_list(x):
    if isinstance(x, list):
        return x
    
    if isinstance(x, dict):
        keys = list(x.keys())
        
        # str key
        if keys and all(isinstance(k, str) and k.isdigit() for k in keys):
            return [x[k] for k in sorted(keys, key=int)]
        
        # int key
        if keys and all(isinstance(k, int) for k in keys):
            return [x[k] for k in sorted(keys)]
        
        # try values
        for v in x.values():
            try:
                result = to_list(v)
                if isinstance(result, list) and len(result) > 0:
                    return result
            except TypeError:
                continue
    
    raise TypeError(f"[WARNING] Can't convert type {type(x)}")


# 2. Create output directory if not exist
# ==============================================
def audit_check(videos, out_csv_path):
    """
    :param videos: all video_data 
    :param out_csv_path: path to output csv file
    """
    print("\n","="*20, "Step 2: Auditing dataset", "="*20)

    if os.path.exists(out_csv_path):
        print(f"[INFO] Audit csv file already exists at {out_csv_path}.")
        return videos
    os.makedirs(Path(out_csv_path).parent, exist_ok=True)
    rows = []
    ok_videos = []

    for vp in videos:
        
        # Case folder path
        case_id = vp.parts[-3]
        name = vp.stem
        case_dir = vp.parents[1]

        # Open video file
        cap = cv2.VideoCapture(str(vp))
        opened = cap.isOpened()

        if opened:
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            fps = -1
            video_frames = -1
        
        cap.release()

        # Choose the corresponding bbox and keypoints files
        if "left" in name.lower():
            bbox_path = case_dir / "bbox_left.json"
            kpt_path = case_dir / "keypoints_left.yaml"
        else:
            bbox_path = case_dir / "bbox_right.json"
            kpt_path = case_dir / "keypoints_right.yaml"
        
        # Load bbox and keypoints info to rows
        if bbox_path.exists():
            with open(bbox_path, 'r', encoding='utf-8') as f:
                bbox_data = to_list(json.load(f))
            bbox_len = len(bbox_data)
        else:
            bbox_len = 0
        
        if kpt_path.exists():
            with open(kpt_path, 'r', encoding='utf-8') as f:
                kpt_data = to_list(yaml.safe_load(f))
            kpt_len = len(kpt_data)
        else:
            kpt_len = 0
        
        rows.append([case_id, name, fps, video_frames, bbox_len, kpt_len, str(vp)])

        aligned = (abs(video_frames - bbox_len) <= 2) and (abs(video_frames - kpt_len) <= 2)
        if opened and aligned:
            ok_videos.append(vp)

    # Save to csv (usually not)
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["case_id", "video", "fps", "video_frames", "bbox_len", "kpt_len", "video_path"])
        w.writerows(rows)
    
    print(f"[DONE] Audit check finished. Results saved to {out_csv_path}.")

    # 
    mismatch = 0
    for r in rows:
        case_id, name, fps, video_frames, bbox_len, kpt_len, vpath = r
        ok1 = abs(video_frames - bbox_len) <= 1
        ok2 = abs(video_frames - kpt_len) <= 1
        status = "OK" if (ok1 and ok2) else "MISMATCH"
        if status == "MISMATCH":
            mismatch += 1
            print(f"[INFO] MISMATCH in {case_id}/{name}: video={video_frames}, bbox={bbox_len}, kpt={kpt_len} -> {status}")
    print(f"[INFO] Total mismatch: {mismatch}")
    return ok_videos


# 3. Extract frames from video files
# ==============================
def extract_frames(video_path, out_dir, target_fps=30, save_frames=False):
    print("\n","="*20, "Step 3: Extracting frames", "="*20)
    """
    :param video_path: path to mp4 video file
    :param out_dir: path to output directory
    :param target_fps: Align with original dataset .json record
    :return: path to saved frames
    """
    done = 0
    failed = []

    frame_idx = 0
    save_dir = Path(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Loop through all video files
    for vp in video_path:

        # Create case folder, and avoid error if has existed
        case_id = vp.parts[-3]
        name = vp.stem
        save_idr = Path(out_dir) / case_id / name
        os.makedirs(str(save_idr), exist_ok=True)   

        # Skip if already extracted
        existing = list(save_idr.glob("*.jpg"))
        if len(existing) > 0:
            done += 1
            continue   
    
        # Open video file and record failures with reasons
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            failed.append((str(vp), " * Reason: Read 0 frames"))
            continue

        # Extract frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break   # Break if no more frames

            # Save frame as jpg
            if save_frames:
                img_path = save_idr / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(img_path), frame)
            frame_idx += 1
    
        cap.release()

        # Check extraction result
        if frame_idx == 0:
            failed.append((str(vp), f"[INFO] Read 0 frames  from {vp}"))
            continue
        done += 1
        
    print(f"[DONE] Extraction Finished. Total videos: {len(video_path)}; Done: {done}; Failed: {len(failed)}.")

    if len(failed) > 0:
        for fail in failed:
            print("[INFO] Failed videos:", "/".join(Path(fail[0]).parts[-3:]), fail[1])
    
    return save_idr, frame_idx, "done"

# 4. Build frame table for each video, containing frame path, bbox and keypoints
# ==========================================
def export_frame_table(videos, frames_dir, out_dir):
    print("\n","="*20, "Step 4: Building frame table", "="*20)
    """
    :param videos: all video_data 
    :param frames_dir: path to extracted frames
    :param out_dir: path to output directory
    """

    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = out_dir / "frame_table.csv"

    # If output already exists, skip
    if out_csv.exists():
        print(f"[INFO] frame_table.csv already exists at {out_dir}.")
        return str(out_dir)
    
    rows = []
    skipped_videos = 0
    total_raws = 0

    for vp in videos:
        
        # Folder path
        case_id = vp.parts[-3]
        name = vp.stem
        case_dir = vp.parents[1]

        # Ensure l and r of video and corresponding files
        side = "left" if "left" in name.lower() else "right"

        if side == "left":
            bbox_path = case_dir / "bbox_left.json"
            kpt_path = case_dir / "keypoints_left.yaml"
        else:
            bbox_path = case_dir / "bbox_right.json"
            kpt_path = case_dir / "keypoints_right.yaml"
        
        video_frames_dir = frames_dir / case_id / name


        # Check the existence of frames, bbox and keypoints files

        if not video_frames_dir.exists():
            print(f"[WARNING] Frames directory does not exist: {video_frames_dir}.")
            skipped_videos += 1
            continue

        if not bbox_path.exists() or not kpt_path.exists():
            print(f"[WARNING] Missing bbox or keypoints file for {case_id}/{name}.")
            skipped_videos += 1
            continue

        if not kpt_path.exists():
            print(f"[WARNING] Missing keypoints file for {case_id}/{name}.")
            skipped_videos += 1
            continue
        
        # Load bbox and keypoints data
        with open(bbox_path, 'r', encoding='utf-8') as f:
            bbox_data = to_list(json.load(f))
        with open(kpt_path, 'r', encoding='utf-8') as f:
            kpt_data = to_list(yaml.safe_load(f))

        # Loop through each frame
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        p = Path(video_frames_dir)
        jpg_count = sum(
            1 for x in p.iterdir()
            if x.is_file() and x.suffix.lower() == ".jpg"
        )

        # Test 
        # print(len(bbox_data), len(kpt_data), jpg_count)

        # Check alignment
        n = min(len(bbox_data), len(kpt_data), jpg_count)

        if n <= 0:
            print(f"[WARNING] No valid frames/bbox/keypoints for {case_id}/{name}.")
            skipped_videos += 1
            continue
        

        for frame_inx in range(n):

            img_path = video_frames_dir / f"{frame_inx:06d}.jpg"

            rows.append([
                case_id,
                name,
                side,
                frame_inx,
                str(Path(img_path).relative_to(REPO_ROOT)),
                json.dumps(bbox_data[frame_inx], ensure_ascii=False),
                json.dumps(kpt_data[frame_inx], ensure_ascii=False),
                fps
            ])
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["case_id", "name", "side", "frame_idx", "frame_path", "boundingbox_raw", "keypoints_raw", "fps"])
        w.writerows(rows)
    
    print(f"[INFO] Total frames rows: {len(rows)}.")
    print(f"[DONE] frame_table saved to {out_csv}.")
    return str(out_csv)


# ==========================================
# ===========================================

def main():

    # Step 1: Find all video files path
    video_data = find_videos(DATA_ROOT)

    # Step 2: Audit check and save to csv
    audit_csv_path = REPO_ROOT / "output" / "audit" / "audit_frames.csv"
    ok_videos = audit_check(video_data, audit_csv_path)

    # Step 3: Extract frames from each video
    extract_frames(ok_videos, TEMP_ROOT, target_fps=30, save_frames=False)

    # Step 4: Build frame table for each video
    out_dir = REPO_ROOT / "output" / "frames"
    export_frame_table(ok_videos, TEMP_ROOT, out_dir) 


if __name__ == "__main__":
    main()