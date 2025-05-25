import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import math 

# 1) Init MediaPipe Face Mesh once
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# 3D model points & landmark indices
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye left corner
    (225.0, 170.0, -135.0), # Right eye right corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])
landmark_indices = [1, 152, 33, 263, 61, 291]

def get_head_pose(img):
    h, w = img.shape[:2]
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    # build 2D points
    pts_2d = []
    for idx in landmark_indices:
        lm = res.multi_face_landmarks[0].landmark[idx]
        pts_2d.append((lm.x * w, lm.y * h))
    pts_2d = np.array(pts_2d, dtype='double')
    # camera intrinsics
    cam_mat = np.array([[w, 0, w/2],
                        [0, w, h/2],
                        [0, 0,   1]], dtype='double')
    dist = np.zeros((4,1))
    # PnP -> rotation matrix
    _, rvec, _ = cv2.solvePnP(model_points, pts_2d, cam_mat, dist)
    rot_mat, _ = cv2.Rodrigues(rvec)
    angles = cv2.decomposeProjectionMatrix(
        np.hstack((rot_mat, np.zeros((3,1))))
    )[6].flatten()
    return angles

def prepare_factor_ranges_dynamic(cap=15, deg_per_fact=6.0, inputs_folder="inputs"):
    factor_ranges = []
    if not os.path.isdir(inputs_folder):
        print(f"No such folder: {inputs_folder}")
        return factor_ranges

    for fname in sorted(os.listdir(inputs_folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(inputs_folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"{fname}: could not read image")
            continue

        hp = get_head_pose(img)
        if hp is None:
            print(f"{fname}: no face detected")
            continue

        _, yaw, _ = hp
        f_min = math.floor((-30.0 - yaw) / deg_per_fact)
        f_max = math.ceil((+30.0 - yaw) / deg_per_fact) + 1 

        f_min, f_max = max(f_min, -cap), min(f_max, cap)
        factor_ranges.append((f_min, f_max))

    print(f"Dynamic factor ranges for {inputs_folder}: {factor_ranges}")
    return factor_ranges

def get_closest_yaw(editing_results_dir):
    eps = 0.05
    top_folder = Path(editing_results_dir)
    if not top_folder.exists():
        print(f"No such folder: {top_folder}")
        return

    targets = {"-30°": -30.0, "  0°": 0.0, "+30°": 30.0}

    for root, _, files in os.walk(top_folder):
        root_path = Path(root)
        # collect (fname, yaw)
        results = []
        for fname in sorted(files):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = root_path / fname
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            _, yaw, _ = get_head_pose(img)
            if abs(yaw) < eps:
                img_path.unlink()
                print(f"Deleted (close to zero yaw): {fname}")
            else:
                results.append((fname, yaw))

        if not results:
            continue

        subfolder = root_path.name
        print(f"\nSubfolder: {subfolder}")

        kept = set()
        for f, y in results:
            print(f"  {f}: Yaw = {y:.8f}°")
            
        for label, (tgt, suffix) in targets.items():
            best_fname, best_yaw = min(results, key=lambda x: abs(x[1] - tgt))

            new_name = f"{subfolder}_{suffix}.png"
            src = root_path / best_fname
            dst = root_path / new_name
            try:
                src.rename(dst)
                print(f"Kept for {label}: {new_name} (Yaw: {best_yaw:.1f}°)")
            except Exception as e:
                print(f"Could not rename {best_fname} → {new_name}: {e}")
            kept.add(new_name)

        # delete everything else in that folder
        for f in root_path.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            if f.name not in kept_new_names:
                try:
                    f.unlink()
                    print(f"Deleted: {f.name}")
                except Exception as e:
                    print(f"Could not delete {f.name}: {e}")