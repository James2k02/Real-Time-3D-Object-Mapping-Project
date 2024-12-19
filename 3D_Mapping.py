import pyrealsense2 as rs   # allows access to camera streams (RGB, Depth, etc.) and sensor configurations
import numpy as np          # numerical operations
import open3d as o3d        # working with 3D data such as point clouds and meshes; used here to generate and visualiz the 3D point cloud
import cv2                  # computer vision tasks; used here to process and visualize RGB and depth images

'''Initialize and Configure RealSense Camera'''

pipeline = rs.pipeline() # creases a pipeline object which acts as the interface to the RealSense camera for managing streams
config = rs.config()     # creates a configuration object to specify what data (streams) to enable and their settings (resolution, format)

'''Configure the Streams (RGB and Depth)'''

# enable_stream enables the specific camera streams at a certain resolution, format, and framerate
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

'''Start the Pipeline'''

# starts the pipeline with the given configuration, enabling the camera to begin streaming data
pipeline.start(config)

# align depth to color
align = rs.align(rs.stream.color)

# spratial and temporal filters for noise reductions
spatial = rs.spatial_filter()   # reduces depth noise
temporal = rs.temporal_filter() # reduces jitter

'''Define the Point Cloud Generation Function'''

# this function will take the RGB frame, depth fram, and camera intrinsics as input and generates a 3D point cloud
def create_point_cloud(color_frame, depth_frame, intrinsics, depth_scale):
    # get_data retrieves the data as a 2D array and then asanyarray will convert the frame data into a NumPy array for processing
    depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale
    color_image = np.asanyarray(color_frame.get_data())

    # getting the camera intrinsics
    fx, fy = intrinsics.fx, intrinsics.fy    # focal lengths; define how strongly the camera focuses light onto the image sensor
    cx, cy = intrinsics.ppx, intrinsics.ppy  # principal point; point in image where the optical axis intersects the image plane
    
    # creating pixel grid
    h, w = depth_image.shape                       # gets height and width of the depth image
    x, y = np.meshgrid(np.arange(w), np.arange(h)) # creates a grid of pixel coordinaes for the entire image 
    
    # converting depth to meters and calculate the 3D points
    valid = (depth_image > 0) & (depth_image < 5) # adjusting max range
    z = depth_image[valid] # converts the depth data from millimeters to meters
    x = (x[valid] - cx) * z / fx    # calculates the 3D coordinates using the camera model
    y = (y[valid] - cy) * z / fy    # " "
    
    # combining the coordinates
    points = np.stack((x, y, z), axis = -1).reshape(-1, 3) # combines x, y, z into a single array of 3D point then flattens the arrays
                                                           # into a list of points (3D coordinates) and their corresponding colors
    colors = color_image[valid] / 255.0            # normalizes the color values from 0 - 255 to 0 - 1 for visualization
    
    # creating an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

'''Stream and Process Frames'''

try:
    print("Press 'q' in the Open3D window to quit.")
    while True:
        # capturing frames
        frames = pipeline.wait_for_frames()    # blocks until a new frame is available
        aligned_frames = align.process(frames) # aligning frames
        depth_frame = aligned_frames.get_depth_frame() # extracts the depth frame
        color_frame = aligned_frames.get_color_frame() # extracts the RGB frame
        
        if not depth_frame or not color_frame:
            continue
        
        # applying filters to depth frame
        filtered_depth = spatial.process(depth_frame)
        filtered_depth = temporal.process(filtered_depth)
        
        # getting camera intrinsics
        profile = pipeline.get_active_profile()                                     # Get the active pipeline profile
        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()      
        depth_stream = profile.get_stream(rs.stream.depth)                          # Access the depth stream
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()        # Get intrinsics

        
        # generate the point cloud
        pcd = create_point_cloud(color_frame, filtered_depth, intrinsics, depth_scale)
            
        vis = o3d.visualization.Visualizer()
        vis.create_window("Real-Time 3D Point Cloud")
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        # # visualizing point cloud
        # o3d.visualization.draw_geometries([pcd], window_name = "Real-Time 3D Point Cloud") # opens interactive viewer to see point cloud
        
       
finally:
    pipeline.stop()
    print("Streaming stopped.")
        
    
    
    
    
    
    
    
    
    
    
    

