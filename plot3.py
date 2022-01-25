import matplotlib.pyplot as plt
import csv
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np

plt.close("all")
data = pd.read_csv("datasetdepth.csv",delimiter=' ')
CX = data['tx']
CY = data['ty']
CZ = data['tz'] 

CtX = data['gx']
CtY = data['gy']
CtZ = data['gz']

dx= data['dx']
dy = data['dy']
dz = data['dz']


plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(CX, CY, CZ, 'red')
ax.plot3D(CtX, CtY, CtZ, 'blue')
plt.figure()
plt.plot(CX,CZ,'green')
plt.plot(CtX,CtZ)

plt.figure()
plt.plot(dx)
plt.plot(dy)
plt.plot(dz)

plt.show()
