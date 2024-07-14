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

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(im, (3, 3), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 20,50 )

# Apply morphological operations to enhance edges
kernel = np.ones((4,4), np.uint8)
morph_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.7

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.') 
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector.create(params)

# Detect blobs.
keypoints = detector.detect(morph_edges)
print(f"Number of keypoints: {len(keypoints)}")

# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
