# -*- coding: utf-8 -*-
"""
Two-Stage Random Forest + CVFH-like Global Feature Denoising
Author: OpenAI Assistant | 2025-07-07
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


def compute_cvfh_like_feature(pcd, bins=33):
    """
    Simulated CVFH descriptor using histogram of surface normals (spherical coords).
    Returns a (2*bins,) vector.
    """
    estimate_normals(pcd)
    normals = np.asarray(pcd.normals)

    theta = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))  # elevation
    phi = np.arctan2(normals[:, 1], normals[:, 0])        # azimuth

    hist_theta, _ = np.histogram(theta, bins=bins, range=(0, np.pi), density=True)
    hist_phi, _ = np.histogram(phi, bins=bins, range=(-np.pi, np.pi), density=True)

    return np.hstack([hist_theta, hist_phi])  # (2*bins,)


def build_feat_stage1(xyz):
    return xyz


def build_feat_stage2(xyz, cvfh_global):
    repeated = np.tile(cvfh_global, (xyz.shape[0], 1))
    return np.hstack([xyz, repeated])


def train_rf(X, y, n_tree=100):
    rf = RandomForestRegressor(
        n_estimators=n_tree, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return rf


def save_normals_as_quiver_png(pcd, out_png="normals_cvfh.png", sample_rate=100):
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

    ax.set_title("Surface Normals after CVFH-based Denoising")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def two_stage_denoise_cvfh(sparse_path, dense_path, out_dir="results_cvfh"):
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
    pred1_scaled = rf1.predict(dense_scaled)
    pred1_xyz = scaler.inverse_transform(pred1_scaled)
    pred1_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred1_xyz))

    # Stage 2: Global CVFH-like features
    cvfh_sparse = compute_cvfh_like_feature(sparse_pcd)
    cvfh_pred1 = compute_cvfh_like_feature(pred1_pcd)

    rf2 = train_rf(build_feat_stage2(sparse_scaled, cvfh_sparse), sparse_scaled)
    pred2_scaled = rf2.predict(build_feat_stage2(dense_scaled, cvfh_pred1))
    pred2_xyz = scaler.inverse_transform(pred2_scaled)
    pred2_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred2_xyz))

    estimate_normals(pred2_pcd)
    save_normals_as_quiver_png(pred2_pcd, os.path.join(out_dir, "normals_stage2.png"))

    # Save results
    o3d.io.write_point_cloud(os.path.join(out_dir, "stage2_denoised.pcd"), pred2_pcd)

    # Visualization
    sparse_pcd.paint_uniform_color([1, 0, 0])   # Red
    dense_pcd.paint_uniform_color([0, 1, 0])    # Green
    pred1_pcd.paint_uniform_color([0, 0, 1])    # Blue
    pred2_pcd.paint_uniform_color([1, 1, 0])    # Yellow

    o3d.visualization.draw_geometries(
        [sparse_pcd, dense_pcd, pred1_pcd, pred2_pcd],
        window_name="Red=sparse  Green=dense  Yellow=stage2"
    )

    print("Finished. Results saved in folder:", out_dir)


if __name__ == "__main__":
    two_stage_denoise_cvfh("4.pcd", "622.pcd", "results_cvfh")

