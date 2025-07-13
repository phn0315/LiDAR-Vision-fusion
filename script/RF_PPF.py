# -*- coding: utf-8 -*-
"""
Two-Stage Random Forest + PPF Point Cloud Denoising with Normals Visualization
Compatible with: Python 2.7, Open3D 0.9.x+
Author: OpenAI Assistant | Date: 2025-07-07
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError("Empty point cloud: %s" % path)
    return pcd


def estimate_normals(pcd, radius=0.05, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )


def compute_ppf(pcd, radius=0.05):
    """
    Placeholder for PPF feature computation.
    Open3D does not natively support compute_ppf_feature as of v0.9–0.17.
    You can replace this with a custom implementation or PCL-Python binding.
    """
    estimate_normals(pcd, radius * 0.5, 30)
    # Replace this with actual implementation or placeholder
    num_points = np.asarray(pcd.points).shape[0]
    return np.random.rand(num_points, 4)  # [angle1, angle2, dist, angle3] as dummy features


def build_feat_stage1(xyz):
    return xyz


def build_feat_stage2(xyz, ppf):
    return np.hstack([xyz, ppf])


def train_rf(X, y, n_tree=120):
    rf = RandomForestRegressor(
        n_estimators=n_tree,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X, y)
    return rf


def save_normals_as_quiver_png(pcd, out_png="normals_stage2.png", sample_rate=100):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if len(points) > sample_rate:
        idx = np.random.choice(len(points), sample_rate, replace=False)
        points = points[idx]
        normals = normals[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              length=0.05, normalize=True, color='b')

    ax.set_title("Surface Normals of Stage-2 Denoised Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def two_stage_denoise_ppf(sparse_path, dense_path, out_dir="results_ppf"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sparse_pcd = load_pcd(sparse_path)
    dense_pcd = load_pcd(dense_path)
    sparse_xyz = np.asarray(sparse_pcd.points)
    dense_xyz = np.asarray(dense_pcd.points)

    scaler = MinMaxScaler()
    sparse_scaled = scaler.fit_transform(sparse_xyz)
    dense_scaled = scaler.transform(dense_xyz)

    # Stage 1
    rf1 = train_rf(build_feat_stage1(sparse_scaled), sparse_scaled)
    pred1_scaled = rf1.predict(build_feat_stage1(dense_scaled))
    pred1_xyz = scaler.inverse_transform(pred1_scaled)
    pred1_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred1_xyz))

    # Stage 2 with PPF
    ppf_sparse = compute_ppf(sparse_pcd)
    ppf_pred1 = compute_ppf(pred1_pcd)
    rf2 = train_rf(build_feat_stage2(sparse_scaled, ppf_sparse), sparse_scaled)
    pred2_scaled = rf2.predict(build_feat_stage2(dense_scaled, ppf_pred1))
    pred2_xyz = scaler.inverse_transform(pred2_scaled)
    pred2_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred2_xyz))

    # Save normals
    estimate_normals(pred2_pcd)
    save_normals_as_quiver_png(pred2_pcd, os.path.join(out_dir, "normals_stage2.png"))

    # Save point clouds
    o3d.io.write_point_cloud(os.path.join(out_dir, "stage2_pred.pcd"), pred2_pcd)

    # Visualization
    sparse_pcd.paint_uniform_color([1, 0, 0])    # Red
    dense_pcd.paint_uniform_color([0, 1, 0])     # Green
    pred1_pcd.paint_uniform_color([0, 0, 1])     # Blue
    pred2_pcd.paint_uniform_color([1, 1, 0])     # Yellow

    o3d.visualization.draw_geometries(
        [sparse_pcd, dense_pcd, pred1_pcd, pred2_pcd],
        window_name="Red=sparse  Green=dense  Blue=stage1  Yellow=stage2"
    )


if __name__ == "__main__":
    two_stage_denoise_ppf("Circle_camera.pcd", "Circle_lidar.pcd", "results_ppf")

