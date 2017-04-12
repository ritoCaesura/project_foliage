import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
import sys
import numpy as np
import math
from PIL import Image

#semantic labelling: leaves blue (0,0,255), ground red (255,0,0)
#instead of computing N histograms and averaging over them, I concatenate the N image inputs and the histogram over it

N=3
colors=[]
for idx in range(1,N):

	im = Image.open('./im'+str(idx)+'0001.png')  
	w, h = im.size 
	labels=Image.open('./im_anno'+str(idx)+'0001.png')

	im_load=im.load()
	labels_load=labels.load()
	
	for x in range(w):
		for y in range(h):
			if labels_load[x,y][0] < 254: #if it is not ground
			#if labels_load[x,y][2] < 254: #if it is not leaf
				im_load[x,y]=(0,0,0,255)	

	#see only the chosen object on the ground
	#plt.imshow(im)
	#plt.show() 
	colors.extend(im.getcolors(w*h)) # number of pixels that used a color and color (r,g,b,a)

#compute
num=255
redFreq,b=np.histogram([c[1][0] for c in colors],bins=num)
greenFreq,b=np.histogram([c[1][1] for c in colors],bins=num)
blueFreq,b=np.histogram([c[1][2] for c in colors],bins=num)

#plot
#plt.hist([c[1][2] for c in colors],bins=num,histtype='step',linestyle='-',color=(0,0,1))
plt.plot(redFreq,color=(1,0,0))
plt.plot(blueFreq,color=(0,0,1))
plt.plot(greenFreq,color=(0,1,0))
plt.xlabel('colors from 0 to 255')
plt.ylabel('frequency')	
plt.xlim([0,255])
plt.show()
