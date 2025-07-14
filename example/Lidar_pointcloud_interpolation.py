import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

def align_to_z_axis(points):
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    aligned_points = np.dot(points, sorted_eigenvectors)
    return aligned_points, sorted_eigenvectors

def draw_bounding_box(pcd):
    points = np.asarray(pcd.points)
    aligned_points, eigenvectors = align_to_z_axis(points)

    min_bound = np.min(aligned_points, axis=0)
    max_bound = np.max(aligned_points, axis=0)

    bounding_box_points = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
    ])

    bounding_box_points = np.dot(bounding_box_points, eigenvectors.T)

    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bounding_box_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def compute_bounding_box(points):
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounding_box_edges = max_bound - min_bound
    z_axis_index = np.argmax(bounding_box_edges)
    return min_bound, max_bound, z_axis_index

def slice_point_cloud_along_z(points, min_bound, max_bound, z_axis_index, num_slices=10):
    z_min = min_bound[z_axis_index]
    z_max = max_bound[z_axis_index]
    slice_height = (z_max - z_min) / num_slices
    slices = []

    for i in range(num_slices):
        z_slice_min = z_min + i * slice_height
        z_slice_max = z_slice_min + slice_height
        slice_indices = np.where((points[:, z_axis_index] >= z_slice_min) & (points[:, z_axis_index] < z_slice_max))
        slice_points = points[slice_indices]
        slices.append(slice_points)
    
    return slices

def find_nearest_neighbors(source_points, target_points):
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(target_points)
    distances, indices = neighbors.kneighbors(source_points)
    return target_points[indices[:, 0]], distances[:, 0]

def interpolate_points(source_points, target_points, num_points=10):
    interpolated_points = []
    for source, target in zip(source_points, target_points):
        for t in np.linspace(0, 1, num_points):
            interpolated_point = (1 - t) * source + t * target
            interpolated_points.append(interpolated_point)
    return np.array(interpolated_points)

def segment_point_cloud(slices):
    segments = []
    for i in range(len(slices) - 1):
        source_points = slices[i]
        target_points = slices[i + 1]
        if source_points.size == 0 or target_points.size == 0:
            continue
        nearest_neighbors, distances = find_nearest_neighbors(source_points, target_points)
        interpolated_points = interpolate_points(source_points, nearest_neighbors)
        segments.append(interpolated_points)
    return segments

def visualize_and_save_segments(segments, output_path="interpolated_segments.ply"):
    geometries = []
    combined_points = []

    for segment in segments:
        segment_pcd = o3d.geometry.PointCloud()
        segment_pcd.points = o3d.utility.Vector3dVector(segment)
        
        combined_points.extend(segment)

        color = [random.random(), random.random(), random.random()]
        colors = np.tile(color, (segment.shape[0], 1))
        segment_pcd.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(segment_pcd)

    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(np.array(combined_points))
    o3d.io.write_point_cloud(output_path, combined_pcd)
    
    o3d.visualization.draw_geometries(geometries, window_name="Segmented Point Cloud with Interpolated Points")

point_cloud_file_path = 'Circle_original_lidar.pcd'

pcd = load_point_cloud(point_cloud_file_path)
points = np.asarray(pcd.points)

aligned_points, eigenvectors = align_to_z_axis(points)
pcd.points = o3d.utility.Vector3dVector(aligned_points)

bounding_box = draw_bounding_box(pcd)
o3d.visualization.draw_geometries([pcd, bounding_box])

min_bound, max_bound, z_axis_index = compute_bounding_box(aligned_points)
num_slices = 50
slices = slice_point_cloud_along_z(aligned_points, min_bound, max_bound, z_axis_index, num_slices=num_slices)

cluster_pcds = []
for cluster in slices:
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
    cluster_color = [random.random(), random.random(), random.random()]
    cluster_pcd.paint_uniform_color(cluster_color)
    cluster_pcds.append(cluster_pcd)
o3d.visualization.draw_geometries(cluster_pcds, window_name="Clustered Point Clouds")

segments = segment_point_cloud(slices)
visualize_and_save_segments(segments, output_path="221.pcd")

