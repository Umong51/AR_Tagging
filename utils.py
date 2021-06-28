import numpy as np
import cv2
import matplotlib.pyplot as plt
from sys import argv


def main():
    if len(argv) != 3:
        print("Usage: python", argv[0], "<marker> <scenery>")
        print("Example: python {} markers/marker1.jpg assets/scene1.jpg".format(argv[0]))
        return
    
    _, marker, img = argv

    img1 = cv2.imread(marker)

    img2 = cv2.imread(img)
    img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)

    found, dst = detect_feature(img1, img2)
    print("Found" if found else "Not Found")

    if not found:
        return

    img2 = cv2.polylines(img2, [dst], True, (255, 0, 255), 3)

    plt.imshow(img2)
    plt.show()



def detect_feature(img1, img2):
    hT, wT, _ = img1.shape

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = {
        "algorithm": FLANN_INDEX_KDTREE, 
        "trees": 5,
    }
    search_params = {"checks": 50}

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    good = []

    for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
    
    found = len(good) > 4

    dst = None

    if found:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
            
        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, matrix))

    return found, dst


if __name__ == '__main__':
    main()

