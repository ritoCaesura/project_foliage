import random as rand
import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

print ('Bivariate mixture of Gaussians')

sampleNum = 1000
x_margins=[1,15]
y_margins=[1,15]
mean=[[3,3],[7,7],[13,13]]
covar=[[1,0],[0,1]]

x=[]
y=[]
for i in range(0,len(mean)):
	for j in range(0,sampleNum):
		a,b=np.random.multivariate_normal(mean[i],covar)
		x.append(a)
		y.append(b)

bins=15
gridx = np.linspace(x_margins[0],x_margins[1],bins)
gridy = np.linspace(y_margins[0],y_margins[1],bins)

#normalize the histogram (divide by the number of samples)
weights = np.ones_like(x)/float(len(x))
	
H, xedges, yedges = np.histogram2d(x,y,bins=(gridx,gridy),weights=weights)

fig = plt.figure()
ax0=fig.add_subplot(121)
pc=ax0.pcolor(xedges, yedges, H)
plt.colorbar(pc)
plt.xlabel("x")
plt.ylabel("y")
plt.axis([x_margins[0], x_margins[1], y_margins[0], y_margins[1]])

ax = fig.add_subplot(122, projection='3d')

# Construct arrays for the anchor positions of the 16 bars.
# Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
# ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
# with indexing='ij'.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = H.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

plt.show()
