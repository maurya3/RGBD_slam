#from realsense_depth.realsense_depth import DepthCamera
from signal import pause
from typing import Sequence
import cv2
import numpy as np
from numpy.lib.type_check import imag
import rospy
from numpy.core.fromnumeric import shape
from realsense_depth import DepthCamera
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RT
from rospy.core import is_shutdown
from std_msgs.msg import Float32MultiArray 
import time
import math
import csv
import pandas as pd
from matplotlib.animation import FuncAnimation
x = 0
y = 0
z = 0
Rx = 0
Ry = 0
Rz = 0
tx = 0
ty = 0
tz = 0
Tr = 0
Tp = 0
Ty = 0
img_id =0
sequence = '08'
ground_truth = '/media/deepak/Thanos/data_odometry_gray/dataset/poses/{}.txt'.format(sequence)
with open(ground_truth) as f:
    ground_truth = f.readlines()

def getAbsoluteScale(frame_id):  #specialized for KITTI odometry dataset
    ss = ground_truth[frame_id-1].strip().split()
    x_prev = float(ss[3])
    y_prev = float(ss[7])
    z_prev = float(ss[11])
    ss = ground_truth[frame_id].strip().split()
    tx = float(ss[3])
    ty = float(ss[7])
    tz = float(ss[11])
    trueX, trueY, trueZ = tx, ty, tz
    R11 = float(ss[0])
    R12 = float(ss[1])
    R13 = float(ss[2])
    R21 = float(ss[4])
    R22 = float(ss[5])
    R23 = float(ss[6])
    R31 = float(ss[8])
    R32 = float(ss[9])
    R33 = float(ss[10])
    R_1 = np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])
    return np.sqrt((tx - x_prev)*(tx - x_prev) + (ty - y_prev)*(ty - y_prev) + (tz - z_prev)*(tz - z_prev)),tx, ty, tz, R_1

imgr = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/{}/image_0/000000.png'.format(sequence), 0)
#imgr = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/camera_data/1.jpg', 0)
#cv2.imshow("immgr",imgr)
### Realsense Depth camera calling
kMinNumFeature = 400
rtdata = Float32MultiArray()
 
# def callback(data):
#     global imu_data
#     imu_data = data.data

# dc = DepthCamera()
# ret, ref_depth_frame, ref_color_frame = dc.get_frame()

## Detector Creat
#detector = cv2.SIFT_create()
#detector = cv2.ORB_create(edgeThreshold=15,patchSize=20,nlevels=10,fastThreshold=20,scaleFactor=2,WTA_K=4,scoreType=cv2.ORB_HARRIS_SCORE,firstLevel=2,nfeatures=1500)
detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
## Lucas Kanade Parameters
lk_params = dict(winSize  = (21, 21), maxLevel = 10,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# convert the image to trhe gray scale
#grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
#grey_r = cv2.GaussianBlur(grey_r,(5,5),0)
## detect the feature points in the first frame  || draw the key points
px_ref = detector.detect(imgr)
img_kp_ref = cv2.drawKeypoints(imgr,px_ref,None,(0,np.random.randint(0,255,dtype=int),0),cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)#cv2.imshow('First_Refrence_frame',img_kp_ref)
#cv2.imshow("ref_kp",img_kp_ref)
# convert the feature points to the float32 numbers
px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
xp = 0
yp = 0
zp = 0

calib = pd.read_csv('/home/deepak/data_odometry_gray/dataset/poses/{}/calib.txt'.format(sequence), delimiter=' ', header=None, index_col=0)
P0 = np.array(calib.loc['P0:']).reshape((3,4))
k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
t = (t / t[3])[:3]
cx = k[0, 2]
cy = k[1, 2]
fx = k[0, 0]
fy = k[1, 1]
thetax = 90
thetay = 90
R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(thetax), -math.sin(thetax) ],
                    [0,         math.sin(thetax), math.cos(thetax)  ]
                    ])

R_y = np.array([[math.cos(thetay),    0,      math.sin(thetay)  ],
                [0,                     1,      0                   ],
                [-math.sin(thetay),   0,      math.cos(thetay)  ]
                ])

def visualize_matches(image1, kp1, image2, kp2, match):
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    return image_matches

# camera parameters
#cx, cy = 322.399, 243.355
#fx ,fy = 603.917, 603.917
#cx, cy = 607.1928, 185.2157
#fx ,fy = 718.8560, 718.8560
# cx, cy = 607.1928, 185.2157
# fx ,fy = 1358.812, 1358.812

pp = (cx,cy)
focal = (fx, fy)
#Initial rotation and translation vectors
cur_R =  np.identity(3,dtype=None)     # r #@ R_y @ R_x #
cur_t = np.array([[0],[0],[0]])        #t # 
pre_time = time.time()

def draw_match_2_side(img1, kp1, img2, kp2, N):
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1,
                                                     N, dtype=np.int)

    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(N)]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    return out_img 


# def point2dto3d(points,fx,fy,cx,cy):
#     z =(points.z)/scale
#     x = (points.x-cx)*z/fx
#     y = (points.y-cy)*z/fy
#     return x,y,z
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha*180/np.pi, beta*180/np.pi, gamma*180/np.pi))

with open('{}datawscale.csv'.format(sequence),'w') as csvfile:
    fieldnames = ['tx', 'ty', 'tz','rx', 'ry', 'rz','trx', 'try', 'trz', 'dx','dy','dz','Tr','Tp','Ty']
    my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
    my_writer.writeheader()    
           

#while True:
for img_id in range(2000):

    px_ref = detector.detect(imgr)
    px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    print(px_ref)
    #rospy.init_node('Imu_odom',anonymous=True)
    #pub = rospy.Publisher('/visual_odom', Float32MultiArray, queue_size=10 )
    # rospy.Subscriber('/vec3',Float32MultiArray, callback)
    #rate = rospy.Rate(30)
    #rate.sleep()
    ########################################################################
    ########################################################################

    imgc = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/{}/image_0/'.format(sequence)+str(img_id).zfill(6)+'.png', 0)
    #print('imgc',imgc)
    #imgc = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/{}/image_0/000000.png'.format(sequence)+'.jpg', 0)
    # grey_r = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # grey_r = cv2.GaussianBlur(grey_r,(5,5),0)
    # ## detect the feature points in the first frame  || draw the key points
    # px_ref = detector.detect(grey_r)
    # #print(shape(px_ref))
    # img_kp_ref = cv2.drawKeypoints(ref_color_frame,px_ref,None,(20,0,250),cv2.DrawMatchesFlags_DEFAULT)
    
    # #cv2.imshow('First_Refrence_frame',img_kp_ref)
    # cv2.imshow("Feature Points",img_kp_ref)
    # # convert the feature points to the float32 numbers
    # px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    
    # ########################################################################
    # ########################################################################
    # ##################### Capture Current frame ############################
    
    # ret, cur_depth_frame,cur_color_frame = dc.get_frame()
    #convert the frames in the grey scale
    #grey_c = cv2.cvtColor(imgc,cv2.COLOR_BGR2GRAY)
    #grey_c = cv2.GaussianBlur(imgc,(5,5),0)

    #px_ref = detector.detect(grey_r)
    #img_kp_ref = cv2.drawKeypoints(grey_r,px_ref,None,(0,220,0),cv2.DrawMatchesFlags_DEFAULT)
    #cv2.putText(img_kp_ref,'Reference Image',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    #cv2.imshow('Previous_reference_frame',img_kp_ref)
    #px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    
    px_cur = detector.detect(imgc)
    img_kp_cur = cv2.drawKeypoints(imgc,px_cur,None,(255,0,0),cv2.DrawMatchesFlags_DEFAULT)

    #Windowd_img = np.concatenate((img_kp_ref,img_kp_cur),axis=1)
    #cv2.imshow('Images',Windowd_img)
    px_cur = np.array([x.pt for x in px_cur], dtype=np.float32) 

    if(len(px_ref) < 1500):
        px_cur = detector.detect(imgc)
        px_cur = np.array([x.pt for x in px_cur], dtype=np.float32)
    px_ref = px_cur 

    ##### calcOpticalFlowPyrLK doesn't respect window size? [bug ?]
    kp2, st, err = cv2.calcOpticalFlowPyrLK(imgc,imgr, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
    px_ref = px_ref[st==1]
    kp2 = kp2[st==1]

    #out_img = cv2.drawMatches(imgc, px_cur,imgr, px_ref, matches1to2=good_matches, outImg=out_img)
    out_image = draw_match_2_side(imgr,px_ref,imgc,kp2,103)
    imS = cv2. resize(out_image, (1080, 320)) 
    cv2.imshow("Feature tracking",imS)
    camMatrix = np.array([[fx,0, cx],[0, fy, cy],[0, 0, 1]])

    #fi_0 = np.concatenate((kp2,px_ref), axis=1)
    print(img_id)
    #convert the points u, v in 3D X, Y, Z
    
    #print(max(kp2[0]),max(kp2[1]))
    # for cur_point in fi_0:
    #     if ((cur_point[0] > 0) and (cur_point[0] < 479)): 
    #         if ((cur_point[1] > 0) and (cur_point[1] < 639)):
    #             depth = cur_depth_frame[round(cur_point[0]),round(cur_point[1])]
    #             ref0 = cur_point[2]
    #             ref1 = cur_point[3]
    #             Z = depth/scale
    #             X = (cur_point[0] - cx)*Z/fx
    #             Y = (cur_point[1] - cy)*Z/fy
    #         point3D = [X,Y,Z]
    #         refr = [ref0,ref1]
    #         kp21.append([cur_point[0],cur_point[1]])
    #         eq.append(point3D)
    #         ref.append(refr)
    #     else:
    #         pass
    # q = np.array(eq,dtype=float)
    # refo = np.array(ref,dtype=float)
    # kp210 = np.array(kp21,dtype=float) 

    #out_image = draw_match_2_side(ref_color_frame,refo,cur_color_frame,kp210,len(kp210))
    #cv2.imshow("out_image",out_image)

    ###############################################################################
    ###############################################################################
    ######################## Essential Matrix #####################################
    
    E, mask = cv2.findEssentialMat(kp2, px_ref,cameraMatrix=camMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0) 
    s, R, t, mask = cv2.recoverPose(E, kp2, px_ref, cameraMatrix=camMatrix, mask=mask)
    #s, cur_R, cur_t, mask = cv2.recoverPose(E, kp2, px_ref, cameraMatrix=camMatrix, mask=mask)

    R = R.T   ####3X3
    t = -R.dot(np.matrix(t))
    scale,tx,ty,tz,R_1 = getAbsoluteScale(frame_id=img_id)

    print(scale)
    #scale = 1
    if(scale > 0.1):
        # self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
        # self.cur_R = R.dot(self.cur_R)
        cur_t = cur_t + scale*cur_R.dot(t)
        cur_R = R.dot(cur_R)
    # x = float(cur_t[0])
    # y = float(cur_t[1])
    # z = float(cur_t[2])
    #scale,tx,ty,tz = getAbsoluteScale(frame_id=img_id)
    # x = 0.95*xp + 0.05*x
    # y = 0.95*yp + 0.05*y
    # z = 0.95*xp + 0.05*z
    Tr,Tp,Ty = rot2eul(R_1)
    Er,Ep,Ey = rot2eul(cur_R)
    # roll_camera1, pitch_camera1, yaw_camera1 = rotationMatrixToEulerAngles(R)
    # with open('data.csv','a') as csv_file:    
    #         pass

    # with open('data.csv', 'a') as csvfile:
    #     fieldnames = ['tx', 'ty', 'tz']
    #     my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
    #     #my_writer.writeheader()
    #     my_writer.writerow({'tx' :x , 'ty':y , 'tz' :z})

    # obj_point = np.array(q,dtype=np.float32)
    # img_point = np.array(kp210,dtype=np.float32)
   
    #dist_coeffs = np.zeros((5,1))
    # dist_coeffs = np.array([0.06547512, -0.22747884, -0.00049986, -0.01098885,  0.29589387],dtype=np.float32)
    # #rvec = np.zeros((3,1))
    # #tvec = np.zeros((3,1)) 
                                                                                   
    # #retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_point,img_point,camMatrix,dist_coeffs)#,rvec,tvec)
    # #print(shape(rvec),shape(tvec))

    # ### convert rotation vector to rotation matrix 
    # Rot = np.matrix(cv2.Rodrigues(R)[0])
   
    # ###### Inverse Rotation
    # Rot = Rot.T   ####3X3
    # check = isRotationMatrix(Rot)
    # print(check)
    # roll_camera, pitch_pitch, yaw_camera = rot2eul(Rot)
    # pos_camera = -Rot.dot(np.matrix(tvec))
    # tvec = -Rot.dot(tvec)
    
    # cur_t = curp_t + scale * Rot.dot(t)
    # cur_R = Rot.dot(curp_R)

    ##### 05    + 213.88017969549108   - 213.8801796954911  + 213.8801796954911
    x = float(cur_t[0]) + 213.88017969549108  
    y = float(cur_t[1]) - 213.8801796954911   
    z = float(cur_t[2]) + 213.8801796954911  



    ##### 08   -180.1066326627873 180.10663266278738 -180.1066326627873
    # x = -(float(cur_t[2]) + 180.10663266278726) 
    # y = -(float(cur_t[1]) - 180.10663266278735) 
    # z = float(cur_t[0]) + 180.10663266278726 
    
    # x = 0.95*xp + 0.05*x
    # y = 0.95*yp + 0.05*y
    # z = 0.95*xp + 0.05*z

    pnpR = RT.from_matrix(cur_R)
    Rx,Ry,Rz = pnpR.as_euler('xyz', degrees=True)

    TpR = RT.from_matrix(R_1)
    Tr,Tp,Ty = TpR.as_euler('xyz', degrees=True)

    # with open('{}datawscale.csv'.format(sequence),'a') as csv_file:    
    #         pass

    with open('{}datawscale.csv'.format(sequence), 'a') as csvfile:
        fieldnames = ['tx', 'ty', 'tz','rx', 'ry', 'rz','trx', 'try', 'trz', 'dx','dy','dz','Tr','Tp','Ty']
        my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
        #my_writer.writeheader()
        my_writer.writerow({'tx' :x , 'ty':y , 'tz' :z, 'rx' : Rx, 'ry': Ry, 'rz' : Rz,'trx' : tx, 'try':ty, 'trz':tz, 'dx' : tx -x,'dy' : ty - y,'dz' : tz - z,'Tr' :Tr,'Tp' : Tp,'Ty': Ty})

    # rotation = RT.from_rotvec(cur_R)
    # pre_rotation = RT.from_rotvec(curp_R)
    # Roll, Pitch, Yaw = rotation.as_euler('xyz',degrees=True)
    # pRoll, pPitch, pYaw = pre_rotation.as_euler('xyz',degrees=True)

    # rot = rotation.as_euler('xyz',degrees=True) - pre_rotation.as_euler('xyz',degrees=True)
    # def animate(i):
    #     data = pd.read_csv('datawscale.csv')
    #     x = data['tx']
    #     z = data['tz']

    #     plt.cla()
    #     plt.plot(x,z)
    #     plt.tight_layout()

    # ani = FuncAnimation(plt.gcf(),animate,interval=1000)
    # plt.show()
    imgr = imgc
    px_ref = px_cur
    img_id +=1
 
    curp_R = cur_R
    curp_t = cur_t
    # #px_ref = kp210

    # xp = x
    # yp = y
    # zp = z 


    key = cv2.waitKey(10)
    if key==27:
        break
#rospy.spin()
#dc.release()
cv2.destroyAllWindows()
