#from realsense_depth.realsense_depth import DepthCamera
import cv2
import csv
import numpy as np
from realsense_t265 import t265
from scipy.spatial.transform import Rotation as RT
import numpy as np
import matplotlib.pyplot as plt
import re

px = 0
py = 0
pz = 0
pvx = 0
pvy = 0
pvz = 0
pax = 0
pay = 0
paz = 0
pox = 0
poy = 0
poz = 0
pow = 0

fig = plt.figure()
ax = plt.axes(projection ='3d')
### Realsense Depth camera calling
realsense_camera = t265()
def lpfilter(pdata,data,cons=0.85):
    x = cons*pdata + (1-cons)*data
    
    return x

while True:
   
    ret, ts, position, velocity,acceleration, orientation, accel, gyro = realsense_camera.get_pose()
    
    x = lpfilter(px,position.x)
    y = lpfilter(py,position.y)
    z = lpfilter(pz,position.z)
    vx = lpfilter(pvx,velocity.x)
    vy = lpfilter(pvy,velocity.y)
    vz = lpfilter(pvz,velocity.z)


    with open('data.csv','a') as csv_file:
        pass
    
    with open('data.csv', 'a') as csvfile: 
        fieldnames = ['ts','x','y','z','vx','vy','vz','ax','ay','az','ox','oy','oz','ow','accel','gyro']
        my_writer = csv.DictWriter(csvfile, delimiter = ' ',fieldnames= fieldnames)
        #my_writer.writeheader()
        my_writer.writerow({'ts' : ts,'x' :position.x, 'y' : position.y, 'z' :position.z,'vx' :velocity.x,'vy' :velocity.y ,'vz' :velocity.z,'ax' :acceleration.x,'ay' :acceleration.y,'az' :acceleration.z, 'ox' :orientation.x,'oy' :orientation.y,'oz' :orientation.z,'ow' :orientation.w, 'accel': accel, 'gyro': gyro})

    px = x
    py = y
    pz = z
    pvx = vx
    pvy = vy
    pvz = vz


    key = cv2.waitKey(100)
    if key==27:
         break

realsense_camera.release()
cv2.destroyAllWindows()
