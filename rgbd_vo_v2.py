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
import datetime
import pandas as pd
import csv
### Realsense Depth camera calling
kMinNumFeature = 400
rtdata = Float32MultiArray()
fourcc = cv2.VideoWriter_fourcc(*'XVID') #cv2.cv.CV_FOURCC(*'XVID')
#out1 = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,320))
#out1 = cv2.VideoWriter('depth.avi',fourcc, 30.0, (640,320))
#out2 = cv2.VideoWriter('matching.avi',fourcc, 30.0, (1280,320))
#out2 = cv2.VideoWriter('outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,320))


calib = pd.read_csv('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/05/calib.txt', delimiter=' ', header=None, index_col=0)
poses_dir = '/media/deepak/Thanos/data_odometry_gray/dataset/poses/05.txt'
poses = pd.read_csv(poses_dir, delimiter=' ', header=None)
gt = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3, 4))  
     
P0 = np.array(calib.loc['P0:']).reshape((3,4))
P1 = np.array(calib.loc['P1:']).reshape((3,4))
P2 = np.array(calib.loc['P2:']).reshape((3,4))


# camera parameters
cx, cy = 322.399, 243.355
fx ,fy = 603.917, 603.917

#Initial rotation and translation vectors
curp_R = np.identity(3,dtype=None)
curp_t = np.array([[0],[0],[0]])
camMatrix = np.array([[fx,0, cx],[0, fy, cy],[0, 0, 1]])

def extract_features(image, detector='sift', mask=None):
    
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
    elif detector == 'surf':
        det = cv2.xfeatures2d.SURF_create()
        
    kp, des = det.detectAndCompute(image, mask)
    
    return kp, des


def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=k)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=k)
    
    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)

    return matches


def filter_matches_distance(matches, dist_threshold):
   
    filtered_match = []
    for m, n in matches:
        if m.distance <= dist_threshold*n.distance:
            filtered_match.append(m)

    return filtered_match

def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):
   
    # Feel free to read OpenCV documentation and tweak these values. These work well
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                     )
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 3 * sad_window ** 2,
                                        P2 = 32 * 3 * sad_window ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                       )
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}:', end-start)
    
    return disp_left


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

def visualize_matches(image1, kp1, image2, kp2, match):
   
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])

    if depth1 is not None:
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for i, (u, v) in enumerate(image1_points):
            z = depth1[int(v), int(u)]
            # If the depth at the position of our matched feature is above 3000, then we
            # ignore this feature because we don't actually know the depth and it will throw
            # our calculations off. We add its index to a list of coordinates to delete from our
            # keypoint lists, and continue the loop. After the loop, we remove these indices
            if z > max_depth:
                delete.append(i)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        #print(object_points)
        
        # Use PnP algorithm with RANSAC for robustness to outliers
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
        #print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))
        
        # Above function returns axis angle rotation representation rvec, use Rodrigues formula
        # to convert this to our desired format of a 3x3 rotation matrix
        rmat = cv2.Rodrigues(rvec)[0]
    
    else:
        # With no depth provided, use essential matrix decomposition instead. This is not really
        # very useful, since you will get a 3D motion tracking but the scale will be ambiguous
        image1_points_hom = np.hstack([image1_points, np.ones(len(image1_points)).reshape(-1,1)])
        image2_points_hom = np.hstack([image2_points, np.ones(len(image2_points)).reshape(-1,1)])
        E = cv2.findEssentialMat(image1_points, image2_points, k)[0]
        _, rmat, tvec, mask = cv2.recoverPose(E, image1_points, image2_points, k)
    
    return rmat, tvec, image1_points, image2_points

def decompose_projection_matrix(p):   
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]   
    return k, r, t

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):   
    # Get focal length of x axis for left camera
    f = k_left[0][0]   
    # Calculate baseline of stereo pair
    if rectified:
        b = t_right[0] - t_left[0] 
    else:
        b = t_left[0] - t_right[0]        
    # Avoid instability and division by zero
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1    
    # Make empty depth map then fill with depth
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left    
    return depth_map

# Let's make an all-inclusive function to get the depth from an incoming set of stereo images
def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False, 
                   rectified=True):
    # Compute disparity map
    disp = compute_left_disparity_map(img_left, 
                                      img_right, 
                                      matcher=matcher, 
                                      rgb=rgb, 
                                      verbose=verbose)
    # Decompose projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)
    # Calculate depth map for left camera
    depth = calc_depth_map(disp, k_left, t_left, t_right)   
    return depth

def extract_features(image, detector='sift', mask=None):
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
    elif detector == 'fast':
        detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    elif detector == 'surf':
        det = cv2.xfeatures2d.SURF_create()
        
    kp, des = det.detectAndCompute(image, mask)
    #kp = np.array(kp,dtype=np.float32)
    
    return kp, des

def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        
        matches = matcher.knnMatch(des1, des2, k=k)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=k)
    
    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)

    return matches  

def filter_matches_distance(matches, dist_threshold):
    filtered_match = []
    for m, n in matches:
        if m.distance <= dist_threshold*n.distance:
            filtered_match.append(m)

    return filtered_match   

def visualize_matches(image1, kp1, image2, kp2, match):
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    return image_matches


T_tot = np.eye(4)
img_id = 0

k_left, r_left, t_left = decompose_projection_matrix(P0)
k_right, r_right, t_right = decompose_projection_matrix(P1)

imgl = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/05/image_0/'+str(img_id).zfill(6)+'.png', 0)
mask = np.zeros(imgl.shape[:2], dtype=np.uint8)
ymax = imgl.shape[0]
xmax =imgl.shape[1]
x = cv2.rectangle(mask, (96,0), (xmax,ymax), (255), thickness = -1)
cur_R = np.identity(3,dtype=None)
cur_t = np.array([[0],[0],[0]])

while True:
    det = cv2.SIFT_create()
    #det = cv2.ORB_create()
    #det = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    #det = cv2.xfeatures2d.SURF_create()
    start = datetime.datetime.now()
    # ## capture the current frame
    imgl = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/05/image_0/'+str(img_id).zfill(6)+'.png', 0)
    imgr = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/05/image_1/'+str(img_id).zfill(6)+'.png', 0)

    imgl1 = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/05/image_0/'+str(img_id+1).zfill(6)+'.png', 0)
    #imgr = cv2.imread('/media/deepak/Thanos/data_odometry_gray/dataset/sequences/08/image_1/'+str(img_id).zfill(6)+'.png', 0)
    disp_left = compute_left_disparity_map(imgl,imgr,matcher='sgbm',rgb=False,verbose=False)    
    #cv2.imshow('disaparity',disp_left)
    #cv2.imwrite("Disparity.jpg",disp_left)
    depth = stereo_2_depth(imgl,imgr,P0,P1)
    dept2 = (depth).astype(np.uint8)
    heatmap = cv2.applyColorMap(dept2, cv2.COLORMAP_RAINBOW)
    #cv2.imwrite('heat map.jpg',heatmap)
    hm = cv2.resize(heatmap, (640, 320))
    cv2.imshow('Depth Map',hm)
    #out1.write(hm)
    kp0, des0 = det.detectAndCompute(imgl, mask=mask)
    kp1, des1 = det.detectAndCompute(imgl1, mask=mask)
    matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(des0, des1, k=2)
    filter_matches = filter_matches_distance(matches,0.5)
    rmat, tvec, img1_points, img2_points = estimate_motion(filter_matches, kp0, kp1, k_left, depth)
    print(img_id)
    #cv2.imshow('Previous_reference_frame',disp_left)    
    matched_image = visualize_matches(imgl,kp0,imgl1,kp1,filter_matches)
    #I1 = np.concatenate((matched_image,heatmap),axis=1)   
    imS = cv2.resize(matched_image, (640, 320))
    #cv2.imshow('Matching',imS)
    #cv2.imwrite('matchedpair.jpg',matched_image)
    #out2.write(imS)
    Tmat = np.eye(4)
    Tmat[:3, :3] = rmat
    Tmat[:3, 3] = tvec.T
    T_tot = T_tot.dot(np.linalg.inv(Tmat))

    R = rmat.T   ####3X3
    t = -R.dot(np.matrix(tvec))
 
    cur_t = cur_t + cur_R.dot(t)
    cur_R = R.dot(cur_R)
    # # Create blank homogeneous transformation matrix
    # Tmat = np.eye(4)
    # # Place resulting rotation matrix  and translation vector in their proper locations
    # # in homogeneous T matrix
    # Tmat[:3, :3] = rmat
    # Tmat[:3, 3] = tvec.T
    # T_tot = T_tot.dot(np.linalg.inv(Tmat))
    # xs = float(T_tot[0][3])
    # ys = float(T_tot[1][3])
    # zs = float(T_tot[2][3])
    xs = float(cur_t[0])
    ys = float(cur_t[1])
    zs = float(cur_t[2])
    
    gx = gt[img_id,0,3]
    gy = gt[img_id,1,3]
    gz = gt[img_id,2,3]

    # with open('datasetdepth.csv','a') as csv_file:  
    #     pass
    #pause(10)
    with open('datasetdepth.csv', 'a') as csvfile:
        fieldnames = ['tx', 'ty', 'tz','gx', 'gy', 'gz','dx','dy','dz']
        my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
        #my_writer.writeheader()
        my_writer.writerow({'tx' :xs , 'ty':ys , 'tz' :zs, 'gx' : gx, 'gy': gy, 'gz' : gz,'dx' : gx -xs,'dy' : gy-ys,'dz' : gz-zs})

    plt.close()
    #out1.release()
    #out2.release()
    img_id+=1
    key = cv2.waitKey(10)
    if key==27:
        break
rospy.spin()
cv2.destroyAllWindows()
