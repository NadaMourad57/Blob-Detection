
# Standard imports
import cv2
import numpy as np

# Read image
im = cv2.imread("drillbit_images/no_color_variation.jpeg", cv2.IMREAD_GRAYSCALE)
resize_ratio = 2
new_width = int(im.shape[1] * resize_ratio)
new_height = int(im.shape[0] * resize_ratio)
print((new_width, new_height))
im = cv2.resize(im, (new_width, new_height))
 
# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 20
params.maxThreshold = 50


# Filter by Area.
params.filterByArea = True
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.') 
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector.create(params)


# Detect blobs.
keypoints = detector.detect(im)
print(f"num of keypoints {len(keypoints)}")


im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs

cv2.imshow("Image", im_with_keypoints)
cv2.waitKey(0)

