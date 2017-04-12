import random as rand
import numpy as np
import matplotlib.pyplot as plt
import math

def gaussian2D(mean, sigmaMat,x):
	diff=x-mean
	sigmaNorm=1
	return (1.0/(2*math.pi)*(2*math.pi)*sigmaNorm)*math.exp((-1.0/2)*np.dot(diff,np.dot(sigmaMat,diff)))

result=[]
xaxis=[]
yaxis=[]

mean=[np.array([3,3]),np.array([7,7]),np.array([13,13])]
sigma=[np.array([[0.6,0],[0,0.6]]),np.array([[0.5,0],[0,0.5]]),np.array([[1.5,0],[0,1.5]])]	

k=0.2
a=[1,2,5]

for x in np.arange(15,0,-0.1):
	xaxis.append(x)
	yVec=[]
	for y in np.arange(0,15,0.1):
		yaxis.append(y)
		vec=[x,y]
		r=0
		for i in range(0,len(mean)):
			r+=a[i]*(math.sin(k*(math.pow(x-mean[i][0],2)+math.pow(y-mean[i][1],2))))*gaussian2D(mean[i],sigma[i],vec)
		yVec.append(r)
	result.append(yVec)

#plotting
plt.imshow(result, extent=[xaxis[-1],xaxis[0],yaxis[0],yaxis[-1]])
plt.colorbar()
plt.ylabel("y")
plt.xlabel("x")
plt.show()

	
