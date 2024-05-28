#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import torch
from model import PointNet2Classification

class PointNetHumanDetectionNode(Node):
    def __init__(self):
        super().__init__('pointnet_human_detection_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(MarkerArray, '/detected_humans', 10)
        self.subscription  # prevent unused variable warning
        
        # Load pre-trained PointNet++ model
        self.model = PointNet2Classification(num_classes=2)
        self.model.load_state_dict(torch.load('pointnet2_human_detection.pth'))
        self.model.eval()

    def pointcloud_callback(self, msg):
        self.get_logger().info(f'Received PointCloud2 data with {msg.width * msg.height} points')

        points = []
        for i, data in enumerate(pc2.read_points(msg, skip_nans=True)):
            points.append([data[0], data[1], data[2]])

        points = np.array(points)
        if points.shape[0] == 0:
            return
        
        # Prepare data for PointNet++
        points = torch.tensor(points, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        
        with torch.no_grad():
            outputs = self.model(points)
            pred = torch.argmax(outputs, dim=1).item()

        markers = MarkerArray()
        if pred == 1:  # If human detected
            center_x = np.mean(points[0, 0, :].numpy())
            center_y = np.mean(points[0, 1, :].numpy())
            center_z = np.mean(points[0, 2, :].numpy())

            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "human_detection"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 1.7
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            markers.markers.append(marker)

        self.publisher.publish(markers)
        self.get_logger().info(f'Published {len(markers.markers)} human markers.')

def main(args=None):
    rclpy.init(args=args)
    node = PointNetHumanDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
