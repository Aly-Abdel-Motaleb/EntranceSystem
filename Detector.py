from commonfunctions import *
import cv2  as cv
import imutils
import numpy as np
import pandas as pd 
from joblib import load

from skimage.filters import threshold_local
from skimage import measure
from skimage.feature import hog
import os
# import requests




def calculate_contour_distance(contour1, contour2): # 
    x1, y1, w1, h1 = cv.boundingRect(contour1)
    c_x1 = x1 + w1
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv.boundingRect(contour2)
    c_x2 = x2 
    c_y2 = y2 + h2/2
    
    if c_x1 > c_x2:
        c_x1 = x1
        c_x2 = x2 + w2

    
    return (abs(c_x1 - c_x2) , abs(c_y1 - c_y2))

def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)

def calculate_distance_y(contour1, contour2):
    x1, y1, w1, h1 = cv.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1

    x2, y2, w2, h2 = cv.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2
    
    if c_y1 > c_y2:
        c_y1 = y1
        c_y2 = y2 + h2

    return (abs(c_x1 - c_x2) , abs(c_y1 - c_y2))

def agglomerative_cluster(contours, threshold_distance=20):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None
        
        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distancex,distancey = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None and distancey <= 25:
                    min_distance = distancex
                    min_coordinate = (x, y)
                elif  min_distance != None and distancex < min_distance and distancey <= 25:
                    min_distance = distancex
                    min_coordinate = (x, y)
        
        if  min_distance != None and min_distance < threshold_distance :
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else:
            break

    return current_contours

def agglomerative_cluster_y(contours, threshold_distance=5):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None
        
        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distancex,distancey = calculate_distance_y(current_contours[x], current_contours[y])
                if min_distance is None and distancex <= 10:
                    min_distance = distancey
                    min_coordinate = (x, y)
                elif  min_distance != None and distancey < min_distance and distancex <= 10:
                    min_distance = distancey
                    min_coordinate = (x, y)
        
        if min_distance != None and min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else:
            break

    return current_contours

def LPD(img):
    # Step 1: Edge Detection
    s= img.shape
    img = imutils.resize(img, width = 1000 , height= 1000 * s[0] // s[1])
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    gray = gray[gray.shape[0]*2//5:gray.shape[0],:]
    gray = cv.GaussianBlur(gray, (3,3),0)
    
    img = img[img.shape[0]*2//5:img.shape[0],:]
    
    # The Black Hat operation is the difference between the closing and input image 
    
    rectKern = cv.getStructuringElement(cv.MORPH_RECT, (7,5))
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKern)
    


    img_thresh = blackhat
    img_thresh[ img_thresh < 50 ] = 0
    img_thresh[ img_thresh >= 50 ] = 255
    
    num_ones = np.count_nonzero(img_thresh == 255)
    num_zeros = np.count_nonzero(img_thresh == 0)
    ratio = round(num_ones / (num_zeros+num_ones),4)
    
    
    sobel_x = cv.Sobel(img_thresh, cv.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_x = np.absolute(sobel_x)
    maxVal = np.max(sobel_x)
    sobel_x = 255 * ((sobel_x) / (maxVal))
    sobel_x = sobel_x.astype("uint8")
    
    restore_kern = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    restored = sobel_x
    if ratio < 0.0042:
        restored = cv.dilate(sobel_x, None, iterations = 2)
    
    closeKern = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    closed_image = cv.morphologyEx(restored, cv.MORPH_CLOSE, closeKern)

    
    
    
    eroded_1= cv.erode(closed_image, None, iterations = 2)
    
    dilated_1 = cv.dilate(eroded_1, None, iterations = 3)
    
    
    eroded_2= cv.erode(dilated_1, None, iterations = 2)
    dilated_2 = cv.dilate(eroded_2, None, iterations = 3)
    
    dilated_2[ dilated_2 < 140 ] = 0
    dilated_2[ dilated_2 >= 140 ] = 255
    
    eroded_3 = cv.erode(dilated_2, None, iterations = 2)
    # print(ratio)
    if ratio >=0.01:
        # print("*"*100)
        erodekern = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        eroded_3 = cv.erode(eroded_3,erodekern,iterations = 2)
        
    dilated_3 = cv.dilate(eroded_3, None, iterations=8)
    
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,1))
    final_img = cv.dilate(dilated_3, vertical_kernel, iterations = 3)
        
    img1_thresh = final_img
    
    contours, hierarchy = cv.findContours(img1_thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

    merged_contours = []
    copyContours = list(contours)
    merged_contours = agglomerative_cluster(copyContours) # merge contours which is close together
    
    cnts = sorted(merged_contours, key=cv.contourArea, reverse=True)
    
    
    img_cont = img.copy()
    cv.drawContours(img_cont,cnts,-1, (0, 255, 0), 2)
#     show_images([final_img])
    cropped_image = np.zeros_like(img)
    for cnt in cnts:

        area = cv.contourArea(cnt)
        x1, y1, w1, h1 = cv.boundingRect(cnt)
        c_x1 = x1 + w1/2
        c_y1 = y1 + h1/2    
        
        if w1 > 300:
            continue
        if h1 >= 150:
            continue
        if 2200 < area < 17500: # filter on the area of the contours 
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv.boundingRect(approx)
            ar = w / float(h)
            # print(f"area:{area}")
            if (ar>=1.5 and ar<=6):
                # print(f"Ar:{ar}")
                if ( y == 0):
                    cropped_image = img[y :y + h + 5, x:x + w]
                else:
                    cropped_image = img[y -6 :y + h + 5, x:x + w]
#                 cv.drawContours(img,cnt,-1,(0,255,0),2)
                
                # show_images([img,cropped_image])
                return cropped_image
    return cropped_image
    

def enhance_plate(plate_img):
    if (np.all(plate_img == 0)):
        return []
    image_value = cv.split(cv.cvtColor(plate_img, cv.COLOR_BGR2HSV))[2]
    
    inverted_image = cv.bitwise_not(image_value)
    plate_img = imutils.resize(plate_img, width = 200)
    inverted_image = imutils.resize(inverted_image, width = 200)
    
    inverted_image[ inverted_image < 120 ] = 0
    inverted_image[ inverted_image >= 120] = 255
    
    
    closeKern = cv.getStructuringElement(cv.MORPH_RECT, (1,4))
    inverted_image = cv.dilate(inverted_image,closeKern,iterations=1)
    
    labels = measure.label(inverted_image, background = 0)
    
 
    # show_images([inverted_image])
    
    # loop over the unique components
    black_image = np.zeros(inverted_image.shape, dtype ='uint8')
    white_image = np.zeros(inverted_image.shape, dtype ='uint8')
    for label in np.unique(labels):
    
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display
        # only connected components for the current label,
        # then find contours in the label mask
        labelMask = np.zeros(inverted_image.shape, dtype ='uint8')
        labelMask[labels == label] = 255     
        
        
        cnts = cv.findContours(labelMask,
                    cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE)
        
        cnts = cnts[1] if imutils.is_cv3() else cnts[0] # for cv2 and cv3
        
        #    and solidity < 0.80
        
        if len(cnts) > 0:
                c = max(cnts, key = cv.contourArea)
                (boxX, boxY, boxW, boxH) = cv.boundingRect(c)
                area = cv.contourArea(c)
                heightRatio = boxH / float(plate_img.shape[0])
                widthRation = boxW / float(plate_img.shape[1])
                aspectRatio = boxW / float(boxH)
                solidity = cv.contourArea(c) / float(boxW * boxH)
                white_image = cv.bitwise_or(white_image,labelMask)
                if(area >=15 and area<600 and heightRatio  <0.9 and widthRation< 0.2 and aspectRatio < 2 and solidity > 0.2 and solidity < 0.8 ):
                    black_image = cv.bitwise_or(black_image,labelMask)
                
                    
    # show_images([black_image,white_image])      
    morph_kern = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
    dilated = cv.dilate(black_image,morph_kern,iterations=2)
    return [dilated,plate_img]

    
    

# Function to check if two contours intersect
def do_contours_intersect(cnt1, cnt2, x_threshold=0):
    rect1 = cv.boundingRect(cnt1)
    rect2 = cv.boundingRect(cnt2)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Check if there is no intersection in the x-direction with the given threshold
    if x1 + w1 + x_threshold < x2 or x2 + w2 + x_threshold < x1:
        return False

    # Check for intersection in the y-direction
    return not (y1 + h1 < y2 or y2 + h2 < y1)

# Function to merge intersecting contours
def merge_intersecting_contours(contours):
    merged_contours = []
    used = set()

    for i, cnt1 in enumerate(contours):
        if i not in used:
            merged = cnt1.copy()

            for j, cnt2 in enumerate(contours):
                if i != j and j not in used and do_contours_intersect(cnt1, cnt2):
                    merged = np.concatenate((merged, cnt2))
                    used.add(j)

            merged_contours.append(merged)

    return merged_contours




def extractChars(img):  
        if (img == []):
            return img
        filteredCnts = []
        closeKern = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
        img[0] = cv.dilate(img[0], closeKern, iterations = 1)
        
        # show_images([img[0],img[1]])
        labels = measure.label(img[0], background = 0)
        white = img[1].copy()
        for idx,label in enumerate(np.unique(labels)):
            if label == 0:
                continue
            # otherwise, construct the label mask to display
            # only connected components for the current label,
            # then find contours in the label mask
            labelMask = np.zeros(img[0].shape, dtype ='uint8')
            labelMask[labels == label] = 255

            cnts = cv.findContours(labelMask,
                    cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE)

            cnts = cnts[1] if imutils.is_cv3() else cnts[0] # for cv2 and cv3
            # ensure at least one contour was found in the mask
            if len(cnts) > 0:

                # grab the largest contour which corresponds
                # to the component in the mask, then grab the
                # bounding box for the contour
                c = max(cnts, key = cv.contourArea)
                (boxX, boxY, boxW, boxH) = cv.boundingRect(c)

                # compute the aspect ratio, solodity, and
                # height ration for the component
                aspectRatio = boxW / float(boxH)
                solidity = cv.contourArea(c) / float(boxW * boxH)
                area = cv.contourArea(c)
                
                # determine if the aspect ratio, solidity,
                keepAspectRatio =  aspectRatio < 1

                areaRatio =  area > 70 and area < 1100
                solidity = cv.contourArea(c) / float(boxW * boxH)
                center_line=img[1].shape[0] // 2
                
                cv.rectangle( white, (boxX, boxY), (boxX + boxW, boxY + boxH), (0, 255, 0), 1)
                
                # keepAspectRatio  and  
                if areaRatio and solidity < 0.8:
                    if((center_line > boxY and center_line < boxY+boxH) or (center_line<=boxY) ) :
                        # cropped_image = img[1].copy()[boxY:boxY + boxH, boxX:boxX + boxW]
                        # removing pole 
                        filteredCnts.append(c)
                        # car_letters.append((cropped_image,boxX))
                        # cv.rectangle( img[1], (boxX, boxY), (boxX + boxW, boxY + boxH), (0, 255, 0), 1)
                        # cv.putText(img[1], f"Area: {area}", (boxX, boxY + boxH + 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        
        #  can be edited to get better results by merging only the contours which is intersecting in y axis
        
        filteredCnts = agglomerative_cluster_y(filteredCnts)  
        filteredCnts = merge_intersecting_contours(filteredCnts) 
        filteredCnts = agglomerative_cluster(filteredCnts, threshold_distance=2)
        for cnt in filteredCnts:
            (boxX, boxY, boxW, boxH) = cv.boundingRect(cnt)
            cropped_image = img[1].copy()[boxY:boxY + boxH, boxX:boxX + boxW]
            car_letters.append((cropped_image,boxX)) 
            cv.rectangle( img[1], (boxX, boxY), (boxX + boxW, boxY + boxH), (0, 255, 0), 1)
        
        # merge contoure which is close together in y axis
        
        flag = 0
        
        if (len(car_letters) >= 2 and len(car_letters) <=7):
            flag = 1 
        else:
            print(len(car_letters))
            flag = 0
        
        cv.line(img[1], (0,img[1].shape[0]//2), ((img[1].shape[1],img[1].shape[0]//2,)), (255,0,0), 1)
        cv.line(white, (0,white.shape[0]//2), ((white.shape[1],white.shape[0]//2,)), (255,0,0), 1)
        # show_images([img[1],white])
        return [img[1],flag]

# def download_image(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         img = cv.imdecode(np.frombuffer(response.content, np.uint8), cv.IMREAD_COLOR)
#         return img
#     else:
#         print(f"Failed to download image from {url}")
#         return None

# Replace 'your_image_url' with the actual URL of the image you want to process

def process_image(image):
    global car_letters
    car_letters = []
    car_image = np.array(image)
    car_plate , flag = extractChars(enhance_plate(LPD(car_image)))
    car_plate = np.array(car_plate)
    result = []
    if flag == 1:
        model = load("data/trained_model.pkl", mmap_mode="r")
        data = []
        car_letters = sorted(car_letters ,key=lambda x: x[1], reverse = True)
        for i in car_letters:
            car_image = cv.resize(i[0],(32,64))
            gray = cv.cvtColor(car_image, cv.COLOR_BGR2GRAY)
            describtor= hog(gray,orientations=9,pixels_per_cell=(8,8), cells_per_block=(1, 1))
            data.append((describtor).flatten())
        df_test = pd.DataFrame(data)
        df_test = df_test.dropna(axis=1)
        result = model.predict(df_test)
        print(result)
        flag = 0
        car_letters = []
    else: 
        result = []
    
    return [car_plate,result]


