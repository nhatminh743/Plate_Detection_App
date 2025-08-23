import cv2
import imutils
import numpy as np
from experiment.functions.Functions import utils


def detect_license_plate(image, imshow_mode = False):
    original_image = image.copy()
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #Resize
    image = imutils.resize(image, width=300)
    if imshow_mode:
        cv2.imshow("original image", image)

    #Gray imagecondacondaconda
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if imshow_mode:
        cv2.imshow("gray image", gray_image)

    #Smoothing
    bilateral_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
    if imshow_mode:
        cv2.imshow("smoothened image", bilateral_image)

    #Otsu's thresholding
    ret, thresh = cv2.threshold(bilateral_image,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if imshow_mode:
        cv2.imshow("thresh image", thresh)

    # Morph
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    if imshow_mode:
        cv2.imshow("morphed image", morph_image)

    #Find contours
    cnts,new = cv2.findContours(morph_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image1=image.copy()
    cv2.drawContours(image1,cnts,-1,(0,255,0),3)
    if imshow_mode:
        cv2.imshow("contours",image1)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    screenCnt = []

    image2 = image.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
    if imshow_mode:
        cv2.imshow("Top 5 contours",image2)

    count = 0
    for c in cnts:
        count += 1
        perimeter = cv2.arcLength(c, True)
        for epsilon_multiplier in np.linspace(0.01, 0.1, 50):
            approx = cv2.approxPolyDP(c, epsilon_multiplier * perimeter, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                angles = []

                for i in range(4):
                    ang = utils.angle(pts[i], pts[(i - 2) % 4], pts[(i - 1) % 4])
                    angles.append(ang)

                approx_square_angles = sum(80 <= a <= 100 for a in angles)

                if approx_square_angles >= 3:
                    screenCnt.append(approx)

    if screenCnt == []:
        detected = 0
        status = 0
        if imshow_mode:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return original_image, detected, status
    else:
        detected = 1

    if detected == 1:

        for screenCnt in screenCnt:
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)  # Bounding boxx

            mask = np.zeros(gray_image.shape, np.uint8)

            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)

            # Cropping
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = image[topx:bottomx, topy:bottomy]

            imgThresh = gray_image[topx:bottomx, topy:bottomy]

            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            if imshow_mode:
                cv2.imshow("roi", roi)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
            if imshow_mode:
                cv2.imshow("imgThresh", imgThresh)

            status = 1

    if imshow_mode:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return original_image, imgThresh, status
