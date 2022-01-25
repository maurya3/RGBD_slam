#!/usr/bin/python

import rospy
from sensor_msgs import msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visual_estimation.msg import inertial_msg
from visual_estimation.msg import vel_msg
import cv2 as cv
import numpy as np
import time
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.6,
                       minDistance = 50,
                       blockSize = 3 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
#color = np.random.randint(0,255,(100,3))

#extra variable
check = 0
f = 0.8088662
#f = 0.8088680
#f = 0.8088662
fx = 0.8093608 # in m
fy = 0.8083715
Or = 328.3222
Oc = 237.0372
Z = 1.0
global T1p
T1p = 0
#i = 0
cam_vel = np.zeros(6)
data = vel_msg()

#global dt
#dt = 0
#J_u = np.array([x_vel*y_vel/f, -(f + (x_vel*x_vel/f)), y_vel])
#J_v = np.array([f + (y_vel*y_vel/f), -(x_vel*y_vel/f), -(x_vel)])
#W = np.array([[[Wx], [Wy], [Wz]]])

#initiate old frame variable
prev_frame = np.zeros((480, 640), dtype = int)
p0 = np.zeros((1,1,2), dtype = int)


#image call_back function
def Image_callback(raw_image): 
    #dt = T1-T1p
    global T1p
    T1 = time.time()
    dt = T1-T1p
    T1p = T1
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(raw_image, desired_encoding='mono8')
    #print(np.shape(frame))
    global prev_frame
    global p0
    global check 
    global x_vel
    global y_vel
    global cam_vel 
    #global i
    #i += 1
    

    if(check == 0):
        # Take first frame and find corners in it
        #old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(frame, mask = None, **feature_params)
        print(np.shape(p0))
        prev_frame = frame
        
    
    else:
        # Create a mask image for drawing purposes
        mask = np.zeros_like(prev_frame)

    
        #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # check for minimum key features 
        if(np.size(p0, 0) < 20):
            p0_ = cv.goodFeaturesToTrack(prev_frame, mask = None, **feature_params)
            #print(np.shape(p0_), np.shape(p0))
            if(p0_ is not None):
                p0 = np.concatenate((p0, p0_), axis = 0)
            

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk_params)

        # print optical flow vectors
        #print(p0-p1)
    

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new, good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        #     frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        # img = cv.add(frame,mask)
        #cv.imshow('frame',img)
        #k = cv.waitKey(30) & 0xff
        
        # Now update the previous frame and previous points
        prev_frame = frame.copy()
        p0 = good_old.reshape(-1,1,2)
        p1 = good_new.reshape(-1,1,2)
        #print(p1-p0)

        ################
        #Extracting pixel velocities
        P1 = np.reshape(p1, [-1,2])
        P0 = np.reshape(p0, [-1,2])
        #print(P0)
        if(np.size(P0, axis = 0) > np.size(P1, axis = 0)):
            min = np.size(P1, axis = 0)
        else:
            min = np.size(P0, axis = 0)

        #print(min)
        #s = slice(0, min)
        #vel = [P1[s, :]-P0[s, :]]

        #vel = np.reshape(vel, [-1, 2])
        #print(vel)
        '''u_ = vel[0:None][0][0]
        v_ = vel[0:None][0][1]
        u = np.array(u_)
        v = np.array(v_)'''
        #print(u_)
        u_ = P1[0:min, 0]-P0[0:min, 0]
        v_ = P1[0:min, 1]-P0[0:min, 1]
        u = np.reshape(u_, [-1, 1])
        v = np.reshape(v_, [-1, 1])
        #print(np.size(u))
        u = u.flatten()
        v = v.flatten()
        U = []
        V = []
        for element in u:
            element = element*dt
            U.append(element)
        
        for element2 in v:
            element2 = element2*dt
            V.append(element2)
            
        #print(U)
        '''x_vel = np.average(u)
        y_vel = np.average(v)
        V_x = Z*x_vel/fx
        V_y = Z*y_vel/fy
        print(V_x, V_y)'''
        ########################
        #Interaction matrix framework
        for j in range(min):
            L_ = np.array([[-f/Z, 0, P0[j, 0]/Z, P0[j, 0]*P0[j, 1]/f, -(f + P0[j, 0]*P0[j, 0]/f), P0[j, 1]], [0, -f/Z, P0[j, 1]/Z, f+(P0[j, 1]*P0[j, 1]/f), -P0[j, 0]*P0[j, 1]/f, -P0[j, 0]]])
            cam_vel = np.empty([1, 6], dtype = float)
            pixel_vel_ = np.vstack((U[j], V[j]))
            if(j==0):
                global L
                global pixel_vel
                L = L_
                pixel_vel = pixel_vel_
            else:
                L = np.concatenate((L, L_), axis=0)
                pixel_vel = np.concatenate((pixel_vel, pixel_vel_))
        
        L_sq = (np.transpose(L)) @ L
        cam_vel = np.linalg.inv(L_sq) @ np.transpose(L) @ pixel_vel
        #print(cam_vel)
        print(cam_vel[0], cam_vel[1], cam_vel[2])
        #print(U, cam_vel[0])
        

        #U, D, V = np.linalg.svd(L, full_matrices=False)
        #cam_vel = np.transpose(V) @ np.linalg.inv(np.diag(D)) @ np.transpose(U) @ pixel_vel
        #print(cam_vel)
        #print(cam_vel[3]*53, cam_vel[4]*53, cam_vel[5]*53)
        #print(cam_vel[0], cam_vel[1], cam_vel[2])

        p0 = good_new.reshape(-1, 1, 2)
    check = check + 0.001
    for i in range(3):
        if(cam_vel[i] >= 5):
            cam_vel[i] = 5
    
    data.v_x = cam_vel[0]
    data.v_y = cam_vel[1]
    data.v_z = cam_vel[2]
    data.w_x = cam_vel[3]
    data.w_y = cam_vel[4]
    data.w_z = cam_vel[5]

    pub.publish(data)
    #plt.plot(check, cam_vel[0])
    #T2 = time.time()
    #dt = T1-T1p
    #print(1/dt)
    #T1p = T1

    



def inertial_msg_cb(inertial_data):
    global Wx, Wy, Wz
    Wx = inertial_data.wx
    Wy = inertial_data.wy
    Wz = inertial_data.wz
    global Z
    Z = inertial_data.h/100
    #Vx = [Z*(J_u*W - x_vel)]/f
    #Vy = [Z*(J_v*W - y_vel)]/f
    #print(Vx)
    #print(Vy)



if __name__ == '__main__':
    #Raw image Subscriber
    #check = 0
    rospy.init_node('velocity_estimator', anonymous = False)
    raw_image_topic = '/usb_cam/image_raw'
    rospy.Subscriber(raw_image_topic, Image, Image_callback)
    pub = rospy.Publisher('vel_topic', vel_msg, queue_size=10)
    rospy.Subscriber('/inertial_data_topic', inertial_msg, inertial_msg_cb)
    rospy.spin()
    #plt.plot(begin, cam_vel[0])
