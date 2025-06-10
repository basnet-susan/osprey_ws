from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path

def generate_launch_description():
    return LaunchDescription([
    
	
        
        
        Node(
	    package='tf2_ros',
	    executable='static_transform_publisher',
	    name='map_to_odom',
	    arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
	),
	
	# Dynamic: odom → base_link
        Node(
            package='vl53l8cx_ros',
            executable='nav_odom_tf_broadcaster',
            name='odom_to_base_link_tf_broadcaster'
        ),
	
	
        Node(
	    package='tf2_ros',
	    executable='static_transform_publisher',
	    name='static_tf_fused',
	    arguments=['1', '1', '0', '0', '0', '0', 'base_link', 'fused_lidar']
	),
    
    
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_1',
            arguments=['0.036', '0.16', '0.025', '-3.8515', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_1/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_2',
            arguments=['0.15', '0.16', '0.03', '0.8392', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_2/lidar']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_3',
            arguments=['0.149', '0.050', '0.03', '-0.80281', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_3/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_4',
            arguments=['0.17', '-0.11', '0.03', '0', '0', '0.0',
                       'base_link', 'osprey_0/sensor_circ_4/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_5',
            arguments=['0.097', '-0.184', '0.03', '-1.5359', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_5/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_6',
            arguments=['0.0348', '-0.162', '0.025', '-2.3744', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_6/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_7',
            arguments=['0.061', '-0.012', '0.055', '0', '-1.5708', '0',
                       'base_link', 'osprey_0/sensor_circ_7/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_8',
            arguments=['-0.125', '0.183', '0.03', '1.5708', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_8/lidar']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_9',
            arguments=['-0.197', '0.110', '0.03', '-3.2055', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_9/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_10',
            arguments=['-0.197', '-0.106', '0.03', '-3.1416', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_10/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_11',
            arguments=['-0.119', '-0.185', '0.03', '-1.5708', '0', '0',
                       'base_link', 'osprey_0/sensor_circ_11/lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_sensor_12',
            arguments=['0.050', '0.012', '-0.055', '0', '1.5708', '-3.1416',
                       'base_link', 'osprey_0/sensor_circ_12/lidar']
        ),
        
        Node(
	    package='rviz2',
	    executable='rviz2',
	    name='rviz2',
	    arguments=['-d', '/home/susan/Downloads/sensor_location.rviz'],
	    output='screen',
	    parameters=[{'use_sim_time': True}]
	),

        

    ])
