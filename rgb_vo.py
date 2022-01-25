#from realsense_depth.realsense_depth import DepthCamera
from signal import pause
import cv2
import numpy as np
import rospy
from numpy.core.fromnumeric import shape
from realsense_depth import DepthCamera
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RT
from rospy.core import is_shutdown
from std_msgs.msg import Float32MultiArray 
import time

### Realsense Depth camera calling
kMinNumFeature = 400
rtdata = Float32MultiArray()

# def callback(data):
#     global imu_data
#     imu_data = data.data

dc = DepthCamera()
ret, ref_depth_frame, ref_color_frame = dc.get_frame()

## Detector Creat
#detector = cv2.ORB_create(edgeThreshold=15,patchSize=20,nlevels=10,fastThreshold=20,scaleFactor=1.2,WTA_K=2,scoreType=cv2.ORB_FAST_SCORE,firstLevel=0,nfeatures=500)
detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
## Lucas Kanade Parameters
lk_params = dict(winSize  = (10, 10), maxLevel = 8,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# convert the image to trhe gray scale
grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
## detect the feature points in the first frame  || draw the key points
px_ref = detector.detect(grey_r)
img_kp_ref = cv2.drawKeypoints(grey_r,px_ref,None,(0,np.random.randint(0,255,dtype=int),0),cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
#cv2.imshow('First_Refrence_frame',img_kp_ref)
cv2.imshow("ref_kp",img_kp_ref)
# convert the feature points to the float32 numbers
px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)

# camera parameters
cx, cy = 322.399, 243.355
fx ,fy = 603.917, 603.917
pp = (cx,cy)
focal = (fx, fy)
#Initial rotation and translation vectors
curp_R = np.identity(3,dtype=None)
curp_t = np.array([[0],[0],[0]])
depth_arr = []
pax = 0
pay = 0
paz = 0
pre_time = time.time()
scale = 1
#print(pre_time)
def draw_match_2_side(img1, kp1, img2, kp2, N):

    """Draw matches on 2 sides
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                                            dtype=np.int
                                            )

    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(N)]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    return out_img 


def point2dto3d(points,fx,fy,cx,cy):
    
    z =(points.z)/scale
    x = (points.x-cx)*z/fx
    y = (points.y-cy)*z/fy

    return x,y,z

dep = []

while True:
    rospy.init_node('Imu_odom',anonymous=True)
    pub = rospy.Publisher('/visual_odom/rt', Float32MultiArray, queue_size=10 )
    # rospy.Subscriber('/vec3',Float32MultiArray, callback)
    rate = rospy.Rate(30)
    rate.sleep()

    # ## capture the current frame

    ret, cur_depth_frame,cur_color_frame = dc.get_frame()

    #convert the frames in the grey scale
    #grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
    grey_c = cv2.cvtColor(cur_color_frame,cv2.COLOR_BGR2GRAY)
    
    #px_ref = detector.detect(grey_r)
    #img_kp_ref = cv2.drawKeypoints(grey_r,px_ref,None,(0,220,0),cv2.DrawMatchesFlags_DEFAULT)
    #cv2.putText(img_kp_ref,'Reference Image',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    #cv2.imshow('Previous_reference_frame',img_kp_ref)
    #px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    
    px_cur = detector.detect(grey_c)
    img_kp_cur = cv2.drawKeypoints(cur_color_frame,px_cur,None,(255,0,0),cv2.DrawMatchesFlags_DEFAULT)
    #cv2.putText(img_kp_cur,'Current Image',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
    Windowd_img = np.concatenate((img_kp_ref,img_kp_cur),axis=1)
    cv2.imshow('Images',Windowd_img)
    px_cur = np.array([x.pt for x in px_cur], dtype=np.float32) 

    if(len(px_ref) < len(px_cur)):
        px_cur = detector.detect(grey_c)
        px_cur = np.array([x.pt for x in px_cur], dtype=np.float32)
    px_ref = px_cur 

    ##### calcOpticalFlowPyrLK doesn't respect window size? [bug ?]
    kp2, st, err = cv2.calcOpticalFlowPyrLK(grey_r,grey_c, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
   
    #out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)
    out_image = draw_match_2_side(ref_color_frame,px_ref,cur_color_frame,kp2,len(kp2))
    cv2.imshow("out_image",out_image)
    camMatrix = np.array([[fx,0, cx],[0, fy, cy],[0, 0, 1]])

    #convert the points u, v in 3D X, Y, Z
    eq = []
    print(max(kp2[0]),max(kp2[1]))
    for cur_point in kp2:
        # if (cur_point[0] > 479):
        #     cur_point[0] = 479
        # if (cur_point[1]> 639):
        #     cur_point[1] = 639
        depth = cur_depth_frame[round(cur_point[1]),round(cur_point[0])]
        multi_point_c = cur_point 
        duc = multi_point_c[0]
        dvc = multi_point_c[1]
        Z = depth/scale
        X = (duc - cx)*Z/fx
        Y = (dvc - cy)*Z/fy
        dudvd_c = [X,Y,Z]
        eq.append(dudvd_c)
    
    q = np.array(eq)
 

    E, mask = cv2.findEssentialMat(kp2, px_ref,cameraMatrix=camMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
     
    s, R, t, mask = cv2.recoverPose(E, kp2, px_ref, cameraMatrix=camMatrix, mask=mask)
    scale = 1000
    #cur_t = curp_t + scale*curp_R.dot(t)
    #cur_R = R.dot(curp_R)

    obj_point = np.array(q,dtype=np.float32)
    img_point = np.fliplr(np.array(kp2,dtype=np.float32))
   
    dist_coeffs = np.zeros((5,1))
    rvec = np.zeros((3,1))
    tvec = np.zeros((3,1)) 
                                                                                    ######'Too many values to unpack' with solvePnPRansac() - Pose Estimation https://stackoverflow.com/questions/28440461/too-many-values-to-unpack-with-solvepnpransac-pose-estimation
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_point,img_point,camMatrix,dist_coeffs,rvec,tvec)
    #print(shape(rvec),shape(tvec))

    ### convert rotation vector to rotation matrix 
    Rot, _ = cv2.Rodrigues(rvec)
   
    ###### Inverse Rotation
    Rot = Rot.T   ####3X3
  
    tvec = -Rot.dot(tvec)


    cur_t = curp_t + scale * Rot.dot(t)
    cur_R = Rot.dot(curp_R)

    pnpR = RT.from_matrix(Rot)
    Rx,Ry,Rz = pnpR.as_euler('xyz', degrees=True)
    print("rot ",Rx, Ry, Rz)
    print("tvec ",tvec)

    rotation = RT.from_rotvec(cur_R)
    pre_rotation = RT.from_rotvec(curp_R)
    Roll, Pitch, Yaw = rotation.as_euler('xyz',degrees=True)
    pRoll, pPitch, pYaw = pre_rotation.as_euler('xyz',degrees=True)

    rot = rotation.as_euler('xyz',degrees=True) - pre_rotation.as_euler('xyz',degrees=True)

    ref_color_frame  = cur_color_frame
    ref_depth_frame = cur_depth_frame
    img_kp_ref= img_kp_cur


    curp_R = cur_R
    curp_t = cur_t
    px_ref = kp2

    r11 = cur_R[0]
    r12 = cur_R[1]
    r13 = cur_R[2]
    t11 = cur_t[0]
    t12 = cur_t[1]
    t13 = cur_t[2]

    rtdata.data = [Rx, Ry, Rz, t11, t12, t13]
    pub.publish(rtdata)
     
    key = cv2.waitKey(100)
    if key==27:
        break
rospy.spin()
dc.release()
cv2.destroyAllWindows()
