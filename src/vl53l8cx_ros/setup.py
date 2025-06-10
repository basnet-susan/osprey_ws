from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vl53l8cx_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.24',
        'scipy>=1.8.0'
    ],
    zip_safe=True,
    maintainer='susan',
    maintainer_email='susan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "vl53l8cx_pointcloud = vl53l8cx_ros.vl53l8cx_pointcloud:main",
            "sensor_each_pointcloud = vl53l8cx_ros.sensor_each_pointcloud:main",
            "depth_point_each = vl53l8cx_ros.depth_point_each:main",
            "ray_visualizer = vl53l8cx_ros.ray_visualizer:main",
            "px4_fused_sensors = vl53l8cx_ros.px4_fused_sensors:main",
            
            "offboard_control = vl53l8cx_ros.offboard_control:main",
            "octomap_builder_node = vl53l8cx_ros.octomap_builder_node:main",
            "plotjuggler_visualizer = vl53l8cx_ros.plotjuggler_visualizer:main",
            
            "nav_odom_tf_broadcaster = vl53l8cx_ros.nav_odom_tf_broadcaster:main",
            "potential_field_computer = vl53l8cx_ros.potential_field_computer:main",
            "vfh_planner_node = vl53l8cx_ros.vfh_planner_node:main",
        ],
    },
)

