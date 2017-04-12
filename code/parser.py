#pylint: skip-file

import numpy as np

data_location = 'c:/scripts/output/init/310117 SAMPLES texture_sand.jpg/'

fpr = []
fnr = []
acc = []
# res = []
gdYES = 0
gdNO = 0

text = open(data_location + "performance_measurement.txt", "r")

for line in open(data_location + "performance_measurement.txt", "r"):
    if 'EVERY' in line:
        t = line.split()
        acc.append(float(t[len(t)-1]))
    if 'positive' in line:
        t = line.split()
        fpr.append(float(t[len(t)-1]))
    if 'negative' in line:
        t = line.split()
        fnr.append(float(t[len(t)-1]))
    # if 'classifying' in line:
    #     t = line.split()
    #     res.append(float(t[len(t)-1]))
    if ' yes' in line:
        gdYES += 1
    if ' no' in line:
        gdNO += 1


print "Mean values of sample set of size " + str(gdNO + gdYES)
print "----------------------------------------"
print "Accuracy of predicting EVERY pixel: " + str(round(np.mean(acc),4))
print "False positive rate: " + str(round(np.mean(fpr),4))
print "False negative rate: " + str(round(np.mean(fnr),4))
print "Accuracy of correctly classifying as a leaf: " + str(round(1-np.mean(fnr),4))
print "# of times not enough matches for GD found: " + str(gdNO) + "/" + str(gdNO + gdYES)