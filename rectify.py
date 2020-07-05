import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
imgl = cv2.imread('left/left01.jpg', 0)
imgr = cv2.imread('right/right01.jpg', 0)
h, w = imgl.shape
# help(cv2.stereoRectify)
cml = np.load('cameraMatrix1.npy')
dcl = np.load('distCoeffs1.npy')
cmr = np.load('cameraMatrix2.npy')
dcr = np.load('distCoeffs2.npy')
R = np.load('R.npy')
T = np.load('T.npy')

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cml, dcl, cmr, dcr, (w, h), R, T, 1, (0,0))
print(P1)
print(P2)
Left_Stereo_Map = cv2.initUndistortRectifyMap(cml, dcl, R1, P1, (w, h), cv2.CV_16SC2)
Right_Stereo_Map= cv2.initUndistortRectifyMap(cmr, dcr, R2, P2, (w, h), cv2.CV_16SC2)

Left_rectified= cv2.remap(imgl, Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
Right_rectified= cv2.remap(imgr, Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
# cv2.imwrite('output/left_rectified.png', Left_rectified)
# cv2.imwrite('output/right_rectified.png', Right_rectified)
im_L=Image.fromarray(Left_rectified)
im_R=Image.fromarray(Right_rectified)

img_compare = Image.new('RGBA',(w * 2, h))
img_compare.paste(im_L, box=(0,0))
img_compare.paste(im_R, box=(640,0))

for i in range(1,20):
    len=480/20
    plt.axhline(y=i*len, color='r', linestyle='-')
plt.imshow(img_compare)
plt.savefig('output/rectify.png')