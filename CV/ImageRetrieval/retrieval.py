import cv2
import numpy as np
from sklearn.cluster import MeanShift


def predict_image(img, query):
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(query,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    
#     X = np.array(list(map(lambda x : kp1[x[0].queryIdx].pt, good)))
#     clustering = MeanShift(bandwidth = 5e+1).fit(X)
#     img_height, img_width = img.shape[:2]
#     height, width = query.shape[:2]
    
#     bboxes = [((center[0] - width /2 ) / img_width, (center[1] - height / 2) / img_width, width / img_width, height / img_height) for center in clustering.cluster_centers_]
    height, width = query.shape[:2]
    cells = []
    for match in good:
        x, y = kp1[match[0].queryIdx].pt
        cells.append([ x // width, y // height])
    cells, counts = np.unique(cells, axis=0, return_counts=True)
    cells = cells.astype(int)
    
    good_cells = cells[np.where(counts > len(des2) / 20)]
    good_cells
    img_height, img_width = img.shape[:2]
    bboxes = [(width * cell[0] / img_width , height * cell[1] / img_height, width / img_width, height / img_height) for cell in good_cells]
    return bboxes