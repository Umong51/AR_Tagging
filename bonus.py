from sys import argv

import numpy as np
from utils import detect_feature
import matplotlib.pyplot as plt
import cv2

def main():
    if len(argv) < 3:
        print("Usage: python", argv[0], "<marker1> <marker2> <scenery>")
        print("Example: python {} markers/marker1.jpg"
            "markers/marker2.jpg assets/bonus1.jpg".format(argv[0]))
        return
    _, *markers, img = argv

    dsts = []
    img2 = cv2.imread(img)
    img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)

    for marker in markers:
        img1 = cv2.imread(marker)

        _, dst = detect_feature(img1, img2)
        dsts.append(dst)

    centers = [np.mean(dst, axis=0)[0] for dst in dsts]
    (x1, y1), (x2, y2) = centers
    x, y = int((x1 + x2)/2), int((y1 + y2)/2)

    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'X'

    cv2.putText(img2, text, (x, y), font, 1, (255, 0, 0), 2)
    
    cv2.polylines(img2, dsts, True, (255, 0, 255), 3)

    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    main()