import pyrealsense2 as rs
import numpy as np
def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z]) 

class t265:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.pose)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        #config.enable_stream(rs.stream.fisheye, 1)
        #config.enable_stream(rs.stream.fisheye, 2) 
        # Start streaming
        self.pipeline.start(config)
    
    def get_pose(self):
         
        frames = self.pipeline.wait_for_frames()
        pose = frames.get_pose_frame()
        color_frame = frames.get_color_frame()
        accel = frames[1].as_motion_frame().get_motion_data()
        gyro = frames[0].as_motion_frame().get_motion_data()
        #print(gyro.x)
        print('accel',accel.z)
        #f1 = frames[0].get_fisheye_frame(1).as_video_frame_profile()
        #f2 = frames[0].get_fisheye_frame(2).as_video_frame_profile()
        #left_data = np.asanyarray(f1.get_data())
        #right_data = np.asanyarray(f2.get_data())
        ts = frames.get_timestamp()
        data = pose.get_pose_data()
        
        position = data.translation
        velocity = data.velocity
        orientation  = data.rotation
        acceleration = data.acceleration
         
        return True, ts, position, velocity, acceleration, orientation, accel, gyro

    def release(self):
        self.pipeline.stop()