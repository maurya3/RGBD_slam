#from realsense_depth.realsense_depth import DepthCamera
import cv2
import numpy as np
import rospy
from numpy.core.fromnumeric import shape
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RT
from rospy.core import is_shutdown
from std_msgs.msg import Float32MultiArray 
import time
import math
import csv

### Realsense Depth camera calling

rtdata = Float32MultiArray()

def callback(data):
    global imu_data
    imu_data = data.data

def visual_odom(data):
    global c_pos
    global c_ori
    global c_vel
    c_pos = data.pose.pose.position
    c_ori = data.pose.pose.orientation
    c_vel = data.twist.twist.linear


def kalman_filter_estimate(ax,ay,az,phi,theta,psi,gx,gy,gz,XXp,PPp,dt=0.033):

    conv_mat = np.array([[1 , 0, -math.sin(theta)],[0, math.cos(phi),math.sin(phi)*math.cos(theta)],[0, -math.sin(phi), math.cos(phi)*math.cos(theta)]])
    inv_mat = np.linalg.inv(conv_mat)
    p,q,r = inv_mat.dot(np.array([[gx],[gy],[gz]]))
    
    ### Motion Model
    ## X_t1 = A @ X_t + B @ T_t + w
    A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, dt, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    B = np.array([[0.5*(dt**2), 0, 0, 0, 0, 0],
                    [0, 0.5*(dt**2), 0, 0, 0, 0],
                         [0, 0, 0.5*(dt**2), 0, 0, 0],
                            [dt, 0, 0, 0, 0, 0],
                                [0, dt, 0, 0, 0, 0],
                                    [0, 0, dt, 0, 0, 0],
                                        [0,0, 0, dt, 0, 0],
                                            [0, 0, 0, 0, dt, 0],
                                                [0, 0, 0, 0, 0, dt]])

    u = np.array([[ax],[ay],[az],
                            p,q,r])

    Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    
    ######
    X = A @ XXp + B @ u    
    P = A @ PPp @ A.T + Q    
    return X, P

def kalman_filter_update(X,P, mx,my,mz, c_phi, c_theta, c_psi):

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    R = np.diag([0.3,0.53,0.16,5,5,5])
    Z = H @ X       #+ np.array([[0.01],[0.01],[0.01],[0.01],[0.01],[0.01]])
    KG = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    MZ = np.array([[mx],[my],[mz],[c_phi],[c_theta],[c_psi]])

    X = X + KG @ (MZ - Z)
    P = P - KG @ H @ P  

    return X, P

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians
# camera parameters
cx, cy = 322.399, 243.355
fx ,fy = 603.917, 603.917
pp = (cx,cy)
focal = (fx, fy)
#Initial rotation and translation vectors
curp_R = np.ones(3,dtype=None)
curp_t = np.array([[0],[0],[0]])
depth_arr = []
pax = 0
pay = 0
paz = 0
pre_time = time.time()
#print(pre_time)
Xp=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
Pp=np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

while True:
    rospy.Subscriber('/rtabmap/odom',Odometry, visual_odom)
    rospy.init_node('Imu_odom',anonymous=True)
    pub = rospy.Publisher('/my_odom', Float32MultiArray, queue_size=10 )
    rospy.Subscriber('/custom_imu',Float32MultiArray, callback)
    rate = rospy.Rate(30)
    rate.sleep()

    cur_time = time.time()
   
    ax  =   imu_data[0]
    ay  =   imu_data[1]
    az  =   imu_data[2]
    gx  =   imu_data[3]
    gy  =   imu_data[4]
    gz  =   imu_data[5]
    roll =  imu_data[6]
    pitch = imu_data[7]
    yaw =   imu_data[8]
    Itheta = (roll,pitch,yaw)

    R = eulerAnglesToRotationMatrix(Itheta)
    ax,ay,az = R @ np.array([ax,ay,az])
    ##### frame transfornation from imu to camera

    rxi = RT.from_quat([c_ori.x, c_ori.y, c_ori.z, c_ori.w])
    c_roll,c_pitch,c_yaw = rxi.as_euler('xyz',degrees = True)

    # rx = RT.from_euler('zyx', [0, 0, 90], degrees=True)
    # r = rx.as_matrix()
    # RR = r @ r1 
    # ri = RT.from_matrix(RR)
    # roll, pitch, yaw = ri.as_euler('zyx',degrees = True)
    ### scale recovery
    # s_x = (ax-pax)*dt*dt
    # s_y = (ay-pay)*dt*dt
    # s_z = (az-paz)*dt*dt   
    # scale = np.array([np.sqrt(s_x**2)+np.sqrt(s_y**2)+np.sqrt(s_z**2)])
    # #print(scale)
    ## capture the current frame   
    #rot = rotation.as_euler('xyz',degrees=True) - pre_rotation.as_euler('xyz',degrees=True)
    
    XXp = Xp
    PPp = Pp

    X, P = kalman_filter_estimate(ax,ay,az,roll,pitch,yaw,gx,gy,gz,XXp,PPp,(1/30))
    x1 = X[0][0]
    x2 = X[1][0]
    x3 = X[2][0]

    X,P = kalman_filter_update(X,P,c_pos.x,c_pos.y,c_pos.z,c_roll,c_pitch,c_yaw)
    print(np.array(X))
    Xp = X
    Pp = P

    with open('openst_imu_kf.csv','a') as csv_file:    
        pass

    with open('openst_imu_kf.csv', 'a') as csvfile:
        fieldnames = ['tx', 'ty', 'tz','rx', 'ry', 'rz','x1','x2','x3']
        my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
        #my_writer.writeheader()
        my_writer.writerow({'tx' :X[0][0] , 'ty':X[1][0] , 'tz' :X[2][0], 'rx' : c_pos.x, 'ry':c_pos.y, 'rz' : c_pos.z, 'x1' :x1, 'x2': x2, 'x3':x3})
    print(shape(X))
    
    # curp_R = cur_R
    # curp_t = cur_t
 
    # r11 = cur_R[0]
    # r12 = cur_R[1]
    # r13 = cur_R[2]
    # t11 = cur_t[0]
    # t12 = cur_t[1]
    # t13 = cur_t[2]

    rtdata.data = [100*X[0][0], 100*X[1][0], 100*X[2][0], 100*c_pos.x, 100*c_pos.y, 100*c_pos.z,100*X[3][0],100*X[4][0],100*X[5][0],100*c_vel.x,100*c_vel.y,100*c_vel.z]
    
    pub.publish(rtdata)
    #rospy.spin()     
    pre_time  = cur_time
    pax = ax
    pay = ay
    paz = az
    key = cv2.waitKey(100)
    if key==27:
        break


cv2.destroyAllWindows()
