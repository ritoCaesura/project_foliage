#pylint: skip-file

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import sys

data_location = 'c:/scripts/output/init/'

def computeAndSave():
    ## Step 1: Warp Perspective to match source image
    im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))

    ## Step 2: Subtract warped image and source iamge
    im_sub = cv2.subtract(im_out, orig_image) 

    ## Step 3: Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(im_sub,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # ## Step 3: Use different filters to turn gray scale image into binary image
    # img = im_sub
    # kernel = np.ones((5,5),np.uint8)
    # # median blur -> erosion -> binary threshold -> dilation -> edge detection
    # blur = cv2.medianBlur(img,5)
    # erosion = cv2.erode(blur,kernel,iterations = 1)
    # ret,threshold = cv2.threshold(erosion,50,255,cv2.THRESH_BINARY)
    # dilation = cv2.dilate(threshold,kernel,iterations = 1)
    # edges = cv2.Canny(dilation,100,200)

    # # create subplots of the different steps of using the filters
    # titles = ['Original Image->','MedianBlur(5)->','Erosion(5)->','Binary Threshold(50)->','Dilation(5)->','Canny Edge Detection']
    # images = [img, blur, erosion, threshold, dilation, edges]
    # for i in xrange(6):
    #     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])

    # orig_image_in_color = cv2.imread(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)0001.png')
    # final = np.bitwise_or(orig_image_in_color, edges[:,:,np.newaxis])

    # ##### START: MOG TEST
    # skewed_image_in_c = cv2.imread(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(1)0001.png')
    # testtt = cv2.warpPerspective(skewed_image_in_c, np.linalg.inv(M), (orig_image_in_color.shape[1], orig_image_in_color.shape[0]))

    # fgbg = cv2.BackgroundSubtractorMOG()
    # fgbg.apply(testtt, learningRate=0.5)
    # fgmask = fgbg.apply(orig_image_in_color, learningRate=0)

    # cv2.imwrite(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_MOGTEST.png', fgmask)
    # ##### END: MOG TEST

    # write all the images to disc
    cv2.imwrite(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_step1_warped.png', im_out)
    cv2.imwrite(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_step2_subtracted.png', im_sub)
    cv2.imwrite(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_step3_otsus_threshold(T='+ str(ret) +').png', th)
    # plt.savefig(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_step3.1_filters.png')
    # cv2.imwrite(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_step3.2_filters.png', edges)
    # cv2.imwrite(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)_step4_detected_not_ground.png', final)
    


orig_image = cv2.imread(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(2)0001.png',0)
skewed_image = cv2.imread(data_location + '/im' + sys.argv[len(sys.argv)-1] + '(1)0001.png',0)
print('/im' + sys.argv[len(sys.argv)-1] + '(2)0001.png')

surf = cv2.SURF(400)

kp1, des1 = surf.detectAndCompute(orig_image, None)
kp2, des2 = surf.detectAndCompute(skewed_image, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))

    computeAndSave()

else:
    print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None