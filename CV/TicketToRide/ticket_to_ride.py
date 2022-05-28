from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
import json
from skimage.measure import label


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}

true_area = 750
true_perimeter = 150


def filter_img(img, th_perimeter, th_area):
    new_img = img.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return new_img
    for contour, h in zip(contours, hierarchy[0]):
        if (cv2.arcLength(contour, True) < th_perimeter or cv2.contourArea(contour) < th_area) and  h[3] == -1:
            cv2.drawContours(new_img, contour, -1, (0, 0, 0), cv2.FILLED)
            cv2.drawContours(new_img, contour, -1, (0, 0, 0), 5)
    return new_img

def color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def return_boxes(img, template):
    w, h = template.shape[::-1]
    res_img = np.int16(img.copy())

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    match = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
    loc = np.where(match >= 0.63)
    lbl, n = label(match >= 0.63, connectivity=2, return_num=True)
    
    lbl = np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])
    
    centers = [[pt[0]+w//2, pt[1]+h//2] for pt in lbl]
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(res_img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    return res_img, centers


def get_yellow_mask(img_rgb, centers):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
    yellow_mask1 = cv2.inRange(hsv, np.array((22, 185, 160)), np.array((45, 255, 255)))
    yellow_mask2 = cv2.inRange(hsv, np.array((22, 150, 212)), np.array((45, 255, 255)))
    yellow_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)
    
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    yellow_mask = filter_img(yellow_mask, 150 * 0.2, true_area * 0.2)
    
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, kernel, iterations = 8)
    
    for center in centers:
        cv2.circle(yellow_mask, center[::-1], 15, (0 , 0 ,0), -1)
    return yellow_mask

# def get_red_mask(img_rgb, mode='beginner'):
#     hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

#     red_mask = cv2.inRange(hsv, np.array((170, 200, 160)), np.array((180, 255, 255)))
    
#     kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,5))
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations = 2)
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
#     red_mask = cv2.medianBlur(red_mask, 5)
#     return red_mask


def get_red_mask(img_rgb, centers):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    red_mask = cv2.inRange(hsv, np.array((170, 170, 155)), np.array((180, 255, 255)))
    
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations = 1)
    red_mask = cv2.medianBlur(red_mask, 9)
    red_mask = filter_img(red_mask, 150 * 0.2, true_area * 0.2)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations = 8)
#     kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2,2))

#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    for center in centers:
        cv2.circle(red_mask, center[::-1], 15, (0 , 0 ,0), -1)
    return red_mask


def get_blue_mask(img_rgb, centers=None):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    blue_mask = cv2.inRange(hsv, np.array([98, 180, 120]), np.array([102, 255, 160]))
        
    blue_mask = filter_img(blue_mask, 150 * 0.2, true_area * 0.2)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_DILATE, kernel, iterations = 10)
    
    
    for center in centers:
        cv2.circle(blue_mask, center[::-1], 15, (0 , 0 ,0), -1)
    
    return blue_mask

def get_black_mask(img_rgb, centers = None):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 35]))
    
    
#     black_mask = filter_img(black_mask, 150 * 0.2, true_area * 0.2)
#     kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
#     black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_DILATE, kernel, iterations = 10)
    
    
#     for center in centers:
#         cv2.circle(black_mask, center[::-1], 15, (0 , 0 ,0), -1)
    
    
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_ERODE, kernel, iterations = 1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations = 4)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations = 3)

    black_mask = cv2.medianBlur(black_mask, 5)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations = 5)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_DILATE, kernel, iterations = 6)
    
    for center in centers:
        cv2.circle(black_mask, center[::-1], 15, (0 , 0 ,0), -1)
    
    return black_mask

def get_green_mask(img_rgb, centers=None):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([70, 160, 70]), np.array([90, 255, 255]))
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2,2))
    
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    green_mask = cv2.medianBlur(green_mask, 5)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations = 7)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2,2))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel, iterations = 11)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel, iterations = 2)
    
    
    
    for center in centers:
        cv2.circle(green_mask, center[::-1], 25, (0 , 0 ,0), -1)
    
    return green_mask

def get_masks(img_rgb,centers):
    green_mask = get_green_mask(img_rgb, centers)
    blue_mask = get_blue_mask(img_rgb, centers)
    red_mask = get_red_mask(img_rgb, centers)
    black_mask = get_black_mask(img_rgb, centers)
    yellow_mask = get_yellow_mask(img_rgb, centers)
    return {'green': green_mask, 'blue': blue_mask, 'red':red_mask, 
            'black': black_mask, 'yellow': yellow_mask}


def return_n_trains(img_rgb, centers):
    red_mask = color(get_red_mask(img_rgb, centers))
    yellow_mask = color(get_yellow_mask(img_rgb, centers))
    green_mask = color(get_green_mask(img_rgb, centers))
    blue_mask = color(get_blue_mask(img_rgb, centers))
    black_mask = color(get_black_mask(img_rgb, centers))

    bold_true_area = 1500
    bold_true_perimeter = 180

    scores = {}
    num_trains = {}
    for mask, colors in zip((red_mask, yellow_mask, green_mask, blue_mask, black_mask), ('red', 'yellow', 'green', 'blue', 'black')):


        contours, hierarchy = cv2.findContours(gray(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        score = 0
        num_train = 0
        if hierarchy is not None:
            for contour in contours:


                ratio = cv2.arcLength(contour, True) / bold_true_perimeter
                ratio += 0.1 * ratio

                value = int(round(ratio, 0))
                if value == 0:
                    continue
                if value in TRAINS2SCORE.keys():
                    score += TRAINS2SCORE[value]
                    num_train += value
                else:
                    if value == 5:
                        score += TRAINS2SCORE[2] + TRAINS2SCORE[3] 
                        num_train += 5
                    elif value == 7:
                        score += TRAINS2SCORE[4] + TRAINS2SCORE[3] 
                        num_train += 7
                    elif value == 9:
                        score += 2 * TRAINS2SCORE[4] 
                        num_train += 8
                    elif value == 10:
                        score += TRAINS2SCORE[4] + TRAINS2SCORE[6] 
                        num_train += 10
                    elif value == 11:
                        score += TRAINS2SCORE[4] + TRAINS2SCORE[6] 
                        num_train += 10
                    elif value == 12:
                        score += TRAINS2SCORE[6] + TRAINS2SCORE[6] 
                        num_train += 12
                    
        if num_train < 8:
            num_train = 0
            score = 0
        scores[colors] = score
        num_trains[colors] = num_train

    return num_trains, scores

def predict_image(img):
    
    ###city detection

    lt, rb = (np.array([312, 101]), np.array([336, 125]))
    img_rgb = cv2.cvtColor(cv2.imread('/autograder/source/train/all.jpg'), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    img_rgb = cv2.resize(img_rgb, np.array(img_rgb.shape[:2])[::-1] // 2)
    img = cv2.resize(img, np.array(img.shape[:2])[::-1] // 2)
    
    template = img_rgb[lt[1] : rb[1], lt[0] : rb[0]]
    template = gray(template)
    
    res, centers = return_boxes( img, template)

    
    ###n_train estimate
    n_trains, scores = return_n_trains(img, centers)
    return list(map(lambda x : [x[0] * 2, x[1] * 2], centers)), n_trains, scores


# centers, n_trains, scores = predict_image(cv2.imread('train/all.jpg'))
# print(n_trains)
# print(scores)

# with open('train/all_scores.json', 'r') as f:
#     all_scores = json.load(f)
# with open('train/all_n_trains.json', 'r') as f:
#     all_n_trains = json.load(f)
    
# print(all_n_trains)
# print(all_scores)
