# -*- coding: utf-8 -*-
"""
Two-Stage Point Cloud Denoising using Random Forest and FPFH Features
Author: [Your Name]
Date: 2025-07-13
Description:
    This script implements a two-stage denoising pipeline for depth camera point clouds
    using sparse LiDAR data as supervision. The first stage performs geometry-only regression,
    while the second stage incorporates local FPFH descriptors for refinement.
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


# === Stage 1: Data Loading and Preprocessing ===
def load_point_cloud(file_path):
    """Load a PCD file as an Open3D PointCloud object."""
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError("Point cloud is empty: {}".format(file_path))
    return pcd


def normalize_coordinates(xyz):
    """Apply MinMax normalization to point cloud coordinates."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(xyz)
    return scaled, scaler


# === Stage 2: Normal and FPFH Computation ===
def estimate_normals(pcd, radius=0.05, max_nn=30):
    """Estimate surface normals for the point cloud."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )


def compute_fpfh(pcd, radius=0.1):
    """Compute Fast Point Feature Histogram (FPFH) descriptors."""
    estimate_normals(pcd, radius=radius * 0.5)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamRadius(radius=radius)
    )
    return np.asarray(fpfh.data).T  # Shape: (N, 33)


# === Stage 3: Feature Construction ===
def build_features_stage1(xyz):
    """Feature construction for stage-1: spatial coordinates only."""
    return xyz


def build_features_stage2(xyz, fpfh):
    """Feature construction for stage-2: coordinates + FPFH."""
    return np.hstack([xyz, fpfh])


# === Stage 4: Model Training ===
def train_random_forest(X, y, n_estimators=120):
    """Train a Random Forest regressor with predefined hyperparameters."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X, y)
    return rf


# === Stage 5: Visualization ===
def visualize_normals(pcd, output_path="normals.png", sample_size=100):
    """Visualize and export 3D surface normals as a quiver plot."""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        points = points[indices]
        normals = normals[indices]

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
    plt.savefig(output_path, dpi=300)
    plt.close()


# === Stage 6: Main Denoising Pipeline ===
def two_stage_denoise(sparse_path, dense_path, output_dir="results_fpfh"):
    """Main pipeline: two-stage RF denoising of depth point clouds guided by LiDAR."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load point clouds
    sparse_pcd = load_point_cloud(sparse_path)
    dense_pcd = load_point_cloud(dense_path)
    sparse_xyz = np.asarray(sparse_pcd.points)
    dense_xyz = np.asarray(dense_pcd.points)

    # Normalize
    sparse_scaled, scaler = normalize_coordinates(sparse_xyz)
    dense_scaled = scaler.transform(dense_xyz)

    # Stage 1 Regression
    rf_stage1 = train_random_forest(build_features_stage1(sparse_scaled), sparse_scaled)
    pred1_scaled = rf_stage1.predict(build_features_stage1(dense_scaled))
    pred1_xyz = scaler.inverse_transform(pred1_scaled)
    pred1_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred1_xyz))

    # Stage 2 Regression with FPFH
    print("[INFO] Computing FPFH descriptors...")
    fpfh_sparse = compute_fpfh(sparse_pcd)
    fpfh_pred1 = compute_fpfh(pred1_pcd)

    rf_stage2 = train_random_forest(
        build_features_stage2(sparse_scaled, fpfh_sparse),
        sparse_scaled
    )
    pred2_scaled = rf_stage2.predict(build_features_stage2(dense_scaled, fpfh_pred1))
    pred2_xyz = scaler.inverse_transform(pred2_scaled)
    pred2_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred2_xyz))

    # Estimate normals
    estimate_normals(pred2_pcd, radius=0.05)
    visualize_normals(pred2_pcd, os.path.join(output_dir, "normals_stage2.png"))

    # Save output point clouds
    o3d.io.write_point_cloud(os.path.join(output_dir, "stage2_pred.pcd"), pred2_pcd)

    # Optional color coding and visualization
    sparse_pcd.paint_uniform_color([1.0, 0.0, 0.0])   # Red
    dense_pcd.paint_uniform_color([0.0, 1.0, 0.0])    # Green
    pred1_pcd.paint_uniform_color([0.0, 0.0, 1.0])    # Blue
    pred2_pcd.paint_uniform_color([1.0, 1.0, 0.0])    # Yellow

    o3d.visualization.draw_geometries(
        [sparse_pcd, dense_pcd, pred1_pcd, pred2_pcd],
        window_name="Red=Sparse | Green=Dense | Blue=Stage-1 | Yellow=Stage-2"
    )


# === Entry Point ===
if __name__ == "__main__":
    # Replace with actual input point cloud paths
    sparse_input = "622.pcd"
    dense_input = "4.pcd"
    output_folder = "results_fpfh"
    two_stage_denoise(sparse_input, dense_input, output_folder)
