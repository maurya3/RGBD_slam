from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib.pyplot as plt 
# import matplotlib.animation as manimation

# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Movie support!')
# writer = FFMpegWriter(fps=15, metadata=metadata)

def animate(i):
    data = pd.read_csv('05datawscale.csv',delimiter=" ")
    x = data["tx"] #- 476 #(-0.00351078)*i - 476
    z = data["tz"] #- 232 #(-18.62887335e-03)*i - 232
    trx = data['trx']
    trz = data['trz']
    plt.cla()
    plt.plot(x,z,label = "Estimated_trajectory")
    plt.plot(trx,trz,label = "Ground_truth_trajectory")
    plt.legend(["Estimated ","Ground Truth",])
    plt.xlabel("X (in m)")
    plt.ylabel("Z (in m)")
    plt.title("Estimated Vs Ground truth trajectory")
#    plt.tight_layout()

ani = FuncAnimation(plt.gcf(),animate,interval=1000)
#plt.tight_layout()
plt.show()