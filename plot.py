import matplotlib.pyplot as plt
import csv
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import plotly.graph_objs as go
from mpl_toolkits import mplot3d
from scipy.signal import butter,filtfilt

# Filter requirements.
T = 2.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 25      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hznyq = 0.5 * fs  # Nyquist Frequencyorder = 2       # sin wave can be approx represented as quadratic
nyq = int(T * fs) # total number of samples
order = 3       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Filter the data, and plot both the original and filtered signals.
#KX1 = butter_lowpass_filter(KX, cutoff, fs, order)
#KY1 = butter_lowpass_filter(KY, cutoff, fs, order)
#KZ1 = butter_lowpass_filter(KZ, cutoff, fs, order)


plt.close("all")
data = pd.read_csv("08datawscale.csv",delimiter=' ')
CX = data['tx'] #+ 214
#CX = butter_lowpass_filter(CX, cutoff, fs, order)
CY = data['ty'] #+434.32
#CY = butter_lowpass_filter(CY, cutoff, fs, order)
CZ = data['tz'] #+214
#CZ = butter_lowpass_filter(CZ, cutoff, fs, order) 
range = np.array(np.linspace(0,1999,2000))
print(range)

rx = data['rx']
ry = data['ry']
rz = data['rz']
trx = data['Tr']
trry = data['Tp']
trz = data['Ty']

#08
CX1 = CX #+ (-0.00351078)*range
CY1 = CY + (-0.06822958)*range
CZ1 = CZ #+ (-8.62887335e-03)*range

#05
# CX1 = CX + ( -0.00693899)*range
# CY1 = CY + (- 0.00693899)*range
# CZ1 = CZ + (0.00760315)*range

#print(CX,CY,CZ)

CtX = data['trx']
CtY = data['try']
CtZ = data['trz']

dy = data['dy']
dz = data['dz']

dx = CX - CtX
dy = CY - CtY
dz = CZ - CtZ

dx1 = CX1 - CtX
dy1 = CY1 - CtY
dz1 = CZ1 - CtZ

ddr = rx - trx
ddp = ry - trry
ddy = rz - trz

range = np.linspace(0,1999,2000)
m1,b  = np.polyfit(np.array(range),np.array(dx),1,cov=True)
m2,b  = np.polyfit(np.array(range),np.array(dy),1,cov=True)
m3,b  = np.polyfit(np.array(range),np.array(dz),1,cov=True)
# plt.plot(range,dx,'o')
# plt.plot(range,m*range + b)
 
print(m1,b)
print(m2,b)
print(m3,b)
# data = pd.read_csv("data415.csv",delimiter=' ')
# CX = data['ex']
# CY = data['ey']
# CZ = data['ez'] 
# CtX = data['px']
# CtY = data['py']
# CtZ = data['pz']

plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(CX, CY, CZ,'red',label='Estimated')
ax.legend()
ax.plot3D(CtX, CtY, CtZ, 'blue', label='Ground Truth')
ax.legend()
ax.set_xlabel("X (in m)")
ax.set_ylabel("Y (in m)")
ax.set_zlabel("Z (in m)")
plt.savefig("3D plot wf.png")

plt.figure()
plt.plot(CX,CZ,'red')
plt.plot(CtX,CtZ,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('X (in m)')
plt.ylabel('Z (in m)')
plt.title("Estimated and Ground Truth XZ position (without filter)")
plt.savefig("/home/deepak/Desktop/GRAPHS/05/without/xzplotwf.png")

plt.figure()
plt.plot(CY,CZ,'red')
plt.plot(CtY,CtZ,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('Y (in m)')
plt.ylabel('Z (in m)')
plt.title("Estimated and Ground Truth YZ position (without filter)")
plt.savefig("/home/deepak/Desktop/GRAPHS/05/without/yzzplotwf.png")


plt.figure()
plt.plot(CX,CY,'red')
plt.plot(CtX,CtY,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('X (in m)')
plt.ylabel('Y (in m)')
plt.title("Estimated and Ground Truth XY position (without filter)")
plt.savefig("/home/deepak/Desktop/GRAPHS/05/without/xyplotwf.png")


plt.figure()
plt.plot(dx,'red')
x1, y1 = [0, 2000], [0, 135]
x2, y2 = [1, 10], [3, 2]
plt.plot(x1, y1,'-')
plt.plot(dy,'green')
#plt.plot(0, 0, 2000, 135, marker = '--')
plt.plot(dz,'blue')
plt.xlabel('Samples')
plt.ylabel('Error (in m)')
plt.legend(["Error in X","Error in Y","Error in Z"])
plt.savefig("/home/deepak/Desktop/GRAPHS/05/without/Errorplotwf.png")

###########################################################

plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(CX1, CY1, CZ1,'red',label='Estimated')
ax.legend()
ax.plot3D(CtX, CtY, CtZ, 'blue', label='Ground Truth')
ax.legend()
ax.set_xlabel("X (in m)")
ax.set_ylabel("Y (in m)")
ax.set_zlabel("Z (in m)")
plt.savefig('/home/deepak/Desktop/GRAPHS/05/with/3dwithfil.png')


plt.figure()
plt.plot(CX1,CZ1,'red')
plt.plot(CtX,CtZ,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('X (in m)')
plt.ylabel('Z (in m)')
plt.title("Estimated and Ground Truth XZ position (after filter)")
plt.savefig('/home/deepak/Desktop/GRAPHS/05/with/XZwithfil.png')


plt.figure()
plt.plot(CY1,CZ1,'red')
plt.plot(CtY,CtZ,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('Y (in m)')
plt.ylabel('Z (in m)')
plt.title("Estimated and Ground Truth YZ position (after filter)")
plt.savefig('/home/deepak/Desktop/GRAPHS/05/with/yzwithfil.png')


plt.figure()
plt.plot(CX1,CY1,'red')
plt.plot(CtX,CtY,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('X (in m)')
plt.ylabel('Y (in m)')
plt.title("Estimated and Ground Truth XY position (after filter)")
plt.savefig('/home/deepak/Desktop/GRAPHS/05/with/xywithfil.png')


plt.figure()
plt.plot(dx1,'red')
plt.plot(dy1,'green')
plt.plot(dz1,'blue')
plt.legend(["Estimated (Without Filter)","Ground Truth",])
plt.xlabel('Samples')
plt.ylabel('Error (in m)')
plt.legend(["Error in X","Error in Y","Error in Z"])
plt.title("Position Error (after filter)")
plt.savefig('/home/deepak/Desktop/GRAPHS/05/with/errorwithfil.png')


plt.figure()
plt.plot(ddr,'red')
plt.plot(ddp,'green')
plt.plot(ddy,'blue')
plt.legend(["Error in roll","Error in pitch","Error in yaw"])
plt.title("Orientation Error")
plt.savefig('/home/deepak/Desktop/GRAPHS/05/with/angleerrorwithfil.png')


plt.show()
