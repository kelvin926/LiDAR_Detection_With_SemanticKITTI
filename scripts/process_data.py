import os
import numpy as np

def load_bin_file(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    return scan.reshape((-1, 4))

def load_label_file(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels.reshape((-1, 1))
    return labels

def save_processed_data(pointcloud, label, output_path):
    np.savez(output_path, point_cloud=pointcloud, label=label)

def process_data(data_dir, output_dir, target_label=11):
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith('.bin'):
                point_file = os.path.join(root, file_name)
                label_file = os.path.join(root, file_name.replace('.bin', '.label'))
                
                points = load_bin_file(point_file)
                labels = load_label_file(label_file)
                
                # Filter points with the target label
                person_points = points[labels[:, 0] == target_label]
                if person_points.size > 0:
                    output_subdir = os.path.join(output_dir, "person")
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, file_name.replace('.bin', '.npz'))
                    save_processed_data(person_points, 1, output_path)  # Label '1' for person
                else:
                    output_subdir = os.path.join(output_dir, "non_person")
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, file_name.replace('.bin', '.npz'))
                    save_processed_data(points, 0, output_path)  # Label '0' for non_person

if __name__ == "__main__":
    # Define relative paths
    data_dir = os.path.join(os.path.dirname(__file__), '../data/sequences/00/velodyne')
    output_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    
    process_data(data_dir, output_dir)
