import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import tf_transformations as tf
import numpy as np


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_cb,
            1)

        # self.goal_sub = self.create_subscription(
        #     PoseStamped,
        #     "/goal_pose",
        #     self.goal_cb,
        #     10
        # )

        # self.traj_pub = self.create_publisher(
        #     PoseArray,
        #     "/trajectory/current",
        #     10
        # )

        # self.pose_sub = self.create_subscription(
        #     PoseWithCovarianceStamped,
        #     self.initial_pose_topic,
        #     self.pose_cb,
        #     10
        # )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, map_msg):
        occupancy_data = np.array(map_msg.data)
        binary_occupancy_grid = np.where((occupancy_data >= 0) & (occupancy_data <= 50), 0, 1)
        binary_occupancy_grid = binary_occupancy_grid.reshape((map_msg.info.height, map_msg.info.width))
        self.binary_occupancy_grid = binary_occupancy_grid

        # Invert and dilate the binary occupancy grid
        self.dilated_occupancy_grid = binary_dilation(self.binary_occupancy_grid, iterations=10)
        self.dilated_occupancy_grid = ~self.dilated_occupancy_grid 

        self.get_logger().info("Binary occupancy grid created and dilated.")
        
        # Save the grid as an image
        self.save_grid_image()

        self.resolution = map_msg.info.resolution
        self.origin_x = map_msg.info.origin.position.x
        self.origin_y = map_msg.info.origin.position.y

        # orientation of the map's origin in terms of rotation around the Z-axis
        self.origin_theta = tf.euler_from_quaternion((
            map_msg.info.origin.orientation.x,
            map_msg.info.origin.orientation.y,
            map_msg.info.origin.orientation.z,
            map_msg.info.origin.orientation.w))[2]

        self.inv_rotation_matrix = np.array([
            [np.cos(-self.origin_theta), -np.sin(-self.origin_theta)],
            [np.sin(-self.origin_theta), np.cos(-self.origin_theta)]
        ])

        self.rotation_matrix = np.array([
            [np.cos(self.origin_theta), -np.sin(self.origin_theta)],
            [np.sin(self.origin_theta), np.cos(self.origin_theta)]
        ])

        self.get_logger().info("Defined transformations")

    def pixel_to_map(self, pixel_x, pixel_y):
        # Convert pixel coordinates to map coordinates
        pixel_coords = np.array([[pixel_x], [pixel_y]])
        map_coords = np.dot(self.inv_rotation_matrix, pixel_coords) * self.resolution
        map_x = map_coords[0] + self.origin_x
        map_y = map_coords[1] + self.origin_y
        return map_x, map_y

    def map_to_pixel(self, map_x, map_y):
        # Convert map coordinates to pixel coordinates
        map_coords = np.array([[map_x - self.origin_x], [map_y - self.origin_y]])
        pixel_coords = np.dot(self.rotation_matrix, map_coords) / self.resolution
        pixel_x = pixel_coords[0]
        pixel_y = pixel_coords[1]
        return pixel_x, pixel_y

    def pose_cb(self, pose):
        clicked_x = pose.pose.position.x
        clicked_y = pose.pose.position.y

        raise NotImplementedError

    def goal_cb(self, msg):
        raise NotImplementedError

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def save_grid_image(self, filename="src/path_planning/binary_occupancy_grid.png", dpi=1000):
        np.save("src/path_planning/binary_occupancy_grid.npy", self.binary_occupancy_grid)
        np.save("src/path_planning/dilated_occupancy_grid.npy", self.dilated_occupancy_grid)

        plt.imshow(self.binary_occupancy_grid, cmap='gray', origin='lower')

        masked_dilated_grid = np.ma.masked_where(self.dilated_occupancy_grid == 0, self.dilated_occupancy_grid)

        plt.imshow(masked_dilated_grid, cmap='Blues', origin='lower', alpha=1)

        plt.title("Binary Occupancy Grid with Dilation")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.colorbar(label='Occupancy')
        plt.savefig(filename, dpi=dpi)
        plt.close()
        self.get_logger().info(f"Binary occupancy grid saved as {filename} with DPI {dpi}")


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
