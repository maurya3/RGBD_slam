#from realsense_depth.realsense_depth import DepthCamera
import cv2
import numpy as np
import rospy
from numpy.core.fromnumeric import shape
from realsense_depth import DepthCamera
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RT
from std_msgs.msg import Float32MultiArray 
import time
import math
import csv

### Realsense Depth camera calling
kMinNumFeature = 400
rtdata = Float32MultiArray()
# camera parameters
cx, cy = 322.399, 243.355
fx ,fy = 603.917, 603.917
pp = (cx,cy)
focal = (fx, fy)
camMatrix = np.array([[fx,0, cx],[0, fy, cy],[0, 0, 1]])
dist_coeffs = np.array([0.06547512, -0.22747884, -0.00049986, -0.01098885,  0.29589387],dtype=np.float32)         

# def callback(data):
#     global imu_data
#     imu_data = data.data


## Detector Creat
detector = cv2.ORB_create(edgeThreshold=15,patchSize=20,
                                                nlevels=10,fastThreshold=20,scaleFactor=2,
                                                WTA_K=4,scoreType=cv2.ORB_HARRIS_SCORE,firstLevel=2,nfeatures=1500)
#detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

## Lucas Kanade Parameters
lk_params = dict(winSize  = (21, 21), maxLevel = 10,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Take first image and convert it to  gray scale
dc = DepthCamera()
ret, ref_depth_frame, ref_color_frame = dc.get_frame()
grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
grey_r = cv2.GaussianBlur(grey_r,(5,5),0)
## detect the feature points in the first frame  || draw the key points
px_ref = detector.detect(grey_r)
# convert the feature points to the float32 numbers
px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)

#Initial rotation and translation vectors
curp_R = np.identity(3,dtype=None)
curp_t = np.array([[0],[0],[0]])
cur_R = np.identity(3,dtype=None)
cur_t = np.array([[0],[0],[0]])
x = 0
y = 0
z = 0
pre_time = time.time()


def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]
	return kp1, kp2


def draw_match_2_side(img1, kp1, img2, kp2, N):

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


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x*180.0/3.141592653589793, y*180.0/3.141592653589793, z*180.0/3.141592653589793])

dep = []

while True:
    #rospy.init_node('Imu_odom',anonymous=True)
    #pub = rospy.Publisher('/visual_odom', Float32MultiArray, queue_size=10 )
    # rospy.Subscriber('/vec3',Float32MultiArray, callback)
    #rate = rospy.Rate(30)
    #rate.sleep()
    ########################################################################
    ########################################################################
    grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
    grey_r = cv2.GaussianBlur(grey_r,(5,5),0)
    ## detect the feature points in the first frame  
    px_ref = detector.detect(grey_r)
    px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    print(shape(px_ref))
    
    ########################################################################
    ########################################################################
    ##################### Capture Current frame ############################
    
    ret, cur_depth_frame,cur_color_frame = dc.get_frame()
    #convert the frames in the grey scale
    grey_c = cv2.cvtColor(cur_color_frame,cv2.COLOR_BGR2GRAY)
    grey_c = cv2.GaussianBlur(grey_c,(5,5),0)
    px_cur = detector.detect(grey_c)
    #img_kp_cur = cv2.drawKeypoints(cur_color_frame,px_cur,None,(255,0,0),cv2.DrawMatchesFlags_DEFAULT)
    px_cur = np.array([x.pt for x in px_cur], dtype=np.float32) 

    if(len(px_ref) < 1500):
        px_cur = detector.detect(grey_c)
        px_cur = np.array([x.pt for x in px_cur], dtype=np.float32)
    px_ref = px_cur 

    ##### calcOpticalFlowPyrLK doesn't respect window size? [bug ?]
    kp2, st, err = cv2.calcOpticalFlowPyrLK(grey_r,grey_c, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
    px_ref = px_ref[st==1]
    kp2 = kp2[st==1]
    
    fi_0 = np.concatenate((kp2,px_ref), axis=1)
    print(shape(fi_0))
    q=[]
    #convert the points u, v in 3D X, Y, Z
    eq = []
    ref = []
    kp21 = []
    #print(cur_depth_frame)

    for cur_point in fi_0:
        if ((cur_point[0] > 0) and (cur_point[0] < 479)): 
            if ((cur_point[1] > 0) and (cur_point[1] < 639)):
                depth = cur_depth_frame[round(cur_point[0]),round(cur_point[1])]
                ref0 = cur_point[2]
                ref1 = cur_point[3]  
                Z = depth #/scale
                X = (cur_point[0] - cx)*Z/fx
                Y = (cur_point[1] - cy)*Z/fy
                
            point3D = [X,Y,Z]
            refr = [ref0,ref1]
            kp21.append([cur_point[0],cur_point[1]])
            eq.append(point3D)
            ref.append(refr)
        else:
            pass
    q = np.array(eq,dtype=float)        #####3D points
    refo = np.array(ref,dtype=float)
    kp210 = np.array(kp21,dtype=float) 

    out_image = draw_match_2_side(ref_color_frame,refo,cur_color_frame,kp210,len(kp210))
    cv2.imshow("out_image",out_image)
    #cv2.imwrite("Feature",out_image)

    ###############################################################################
    ###############################################################################
    ######################## Computation using  Essential Matrix #####################################
    
    E, mask = cv2.findEssentialMat(kp2, px_ref,cameraMatrix=camMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0) 
    s, R, t, mask = cv2.recoverPose(E, kp2, px_ref, cameraMatrix=camMatrix, mask=mask)
    scale = 1
    ## frame Transformation from camera to world 
    R = R.T   ####3X3
    t = -R.dot(np.matrix(t))
    cur_t = cur_t + scale*curp_R.dot(t)
    cur_R = R.dot(cur_R)
    x = float(cur_t[0])
    y = float(cur_t[1])
    z = float(cur_t[2])
    

    ###############################################################################
    ###############################################################################
    ######################## Computation using  PnPRansac #####################################

    obj_point = np.array(q,dtype=np.float32)
    img_point = np.array(kp210,dtype=np.float32)
    print(shape(obj_point))
    print(shape(img_point))
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_point,img_point,camMatrix,dist_coeffs)#,rvec,tvec)
     
    ### convert rotation vector to rotation matrix 
    Rot = np.matrix(cv2.Rodrigues(rvec)[0])   ## rvec 3x1 ###Rot 3x3

    ###### Inverse Rotation
    Rot = Rot.T                 #3X3
    tvec = -Rot.dot(tvec)       #3x1     
    check = isRotationMatrix(Rot)
    #print(check)
    roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(Rot)
    
    cur_t = cur_t + scale*curp_R.dot(t)
    cur_R = R.dot(cur_R)

    curp_t = curp_t + scale * Rot.dot(t)
    curp_R = Rot.dot(curp_R)
    px = float(curp_t[0])
    py = float(curp_t[1])
    pz = float(curp_t[2])
    
    #x = 0.95*x + 0.05*x
    #y = 0.95*y + 0.05*y
    #z = 0.95*x + 0.05*z

    #pnpR = RT.from_matrix(cur_R)
    #Rx,Ry,Rz = pnpR.as_euler('xyz', degrees=True)
    #print("rot ",Rx, Ry, Rz)
    #print("tvec ",cur_t[0],cur_t[1],cur_t[2])

    with open('data415.csv','a') as csv_file:    
            pass

    with open('data415.csv', 'a') as csvfile:
        fieldnames = ['ex', 'ey', 'ez','px', 'py', 'pz']
        my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
        #my_writer.writeheader()
        my_writer.writerow({'ex' :x , 'ey':y , 'ez' :z, 'px' : px, 'py': py, 'pz' : pz})


    ref_color_frame  = cur_color_frame
    ref_depth_frame = cur_depth_frame
    px_ref = kp210                         ####2D image point after filtering

    key = cv2.waitKey(1000)
    if key==27:
        break
#rospy.spin()
dc.release()
cv2.destroyAllWindows()
