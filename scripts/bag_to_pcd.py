import os
import sys
import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d

def extract_pcd_from_bag(bag_file, output_dir):
    bag = rosbag.Bag(bag_file, 'r')
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        pc = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points = list(pc)
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([point[:3] for point in points])
        pcd.colors = o3d.utility.Vector3dVector([[point[3], point[3], point[3]] for point in points])
        
        # Save to PCD file
        filename = os.path.join(output_dir, f"{t.to_nsec()}.pcd")
        o3d.io.write_point_cloud(filename, pcd)
    
    bag.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python bag_to_pcd.py <bag_file> <output_dir>")
        sys.exit(1)

    bag_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_pcd_from_bag(bag_file, output_dir)
