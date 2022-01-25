import matplotlib.pyplot as plt
import csv
import pandas as pd
from mpl_toolkits import mplot3d
import statistics

#mpl.rcParams['legend.fontsize'] = 10

def variance(data):
    n = len(data)
    mean = sum(data)/n
    deviation = [(x - mean)**2 for x in data]
    variance = sum(deviation) /n
    return variance


plt.close("all")
data = pd.read_csv("openst_imu_kf.csv",delimiter=' ')

CX = data['tx']
CY = data['ty']
CZ = data['tz']
KX = data['cx']
KY = data['cy']
KZ = data['cz'] 
pX = data['px']
pY = data['py']
pZ = data['pz'] 

var_cx = variance(CX)
var_cy = variance(CY)
var_cz = variance(CZ)
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from mpl_toolkits import mplot3d
from scipy.signal import butter,filtfilt


#plt.close("all")
#data = pd.read_excel("/home/deepak/Desktop/random.ods",engine='odf')

#CX = data['CX']
#CY = data['CY']
#CZ = data['CZ'] 
#TX = data['TX']
#TY = data['TY'] 
#TZ = data['TZ'] 


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
KX1 = butter_lowpass_filter(KX, cutoff, fs, order)
KY1 = butter_lowpass_filter(KY, cutoff, fs, order)
KZ1 = butter_lowpass_filter(KZ, cutoff, fs, order)




print(var_cx,var_cy,var_cz)

plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(CX, CY, CZ, 'red',label='After filtering')
ax.plot3D(KX1, KY1, KZ1, 'green',label='Low pass filtering')
ax.scatter3D(KX, KY, KZ, 'blue',label='Raw D415 camera data')
ax.scatter3D(pX, pY, pZ, 'cyan',label='kalman filter predicted data')
ax.legend()
#plt.plot(CX)
#plt.plot(CY)
#plt.plot(CZ)
plt.show()