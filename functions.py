################################################### functions ######################################################
import numpy as np
import cv2


roi_static = [[(260, 146), (638, 239), 'text', 'CIN'],
            [(260, 230), (700, 296), 'text', 'fname'],
            [(269, 296), (702, 342), 'text', 'name'],
            [(272, 333), (777, 386), 'text', 'tree'],
            [(350, 388), (644, 426), 'text', 'Bd'],
            [(359, 432), (719, 498), 'text', 'place']]

#################################  V2  #####################################
roi_diff = [[(0, 0), (0, 0), 'text', 'CIN'],
            [(120, 84), (109, 59), 'text', 'fname'],
            [(65, 141), (117, 98), 'text', 'name'],
            [(92, 182), (200, 143), 'text', 'tree'],
            [(90, 225), (45, 193), 'text', 'Bd'],
            [(99, 275), (127, 250), 'text', 'place']]
#################################  V1  #####################################
#roi_diff = [[(0, 0), (0, 0), 'text', 'CIN'],
#            [(160, 84), (66, 48), 'text', 'fname'],
#            [(65, 145), (71, 89), 'text', 'name'],
#            [(92, 187), (153, 137), 'text', 'tree'],
#            [(90, 231), (8, 187), 'text', 'Bd'],
#            [(99, 276), (81, 239), 'text', 'place']]


def roi_dynamic(Ps,Pe):
    import operator
    roi=[]
    for x,r in enumerate(roi_diff):
        roi.append([tuple(map(operator.add, r[0], Pe)),tuple(map(operator.add, r[1], Ps)),r[2],r[3]])
    return roi


def CIN_Template(image):
    imgo = cv2.imread('static\\template_cin\Original.png')
    h, w, c = imgo.shape
    imgo = cv2.resize(imgo, (w * 2, h * 2))
    h, w, c = imgo.shape
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgo, None)
    imgkp = cv2.drawKeypoints(imgo, kp1, None)
    # cv2.imshow("outpout",imgkp)
    ####################################################    Test Image     ######################
    per = 25
    img = cv2.imread(image)
    w_img,h_img,_=img.shape
    if h_img<w_img:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (w * 2, h * 2))
    # cv2.imshow(im, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des2, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgo, kp1, good[:100], None, flags=2)
    # cv2.imshow(im, imgMatch)
    srcpts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstpts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcpts, dstpts, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))  # just h and w (no resize)
    #cv2.imshow(im, imgScan)
    return imgScan
def Crop_Black_Edges(imgScan):
    imgTest = imgScan.copy()
    h,w,p=imgTest.shape
    grayT = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayT, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, we, he = cv2.boundingRect(cnt)
    crop = imgTest[y:y + he, x:x + we]
    crop = cv2.resize(crop, (w, h))
    #cv2.imshow(im, crop)
    return crop
def Denoise_Image(crop):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(crop, -1, sharpen_kernel)
    #converted_img = cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)
    noiseless_image = cv2.fastNlMeansDenoisingColored(sharpen, None, 20, 20, 7, 21)
    #cv2.imshow(im, noiseless_image)
    #bilateral = cv2.bilateralFilter(crop, 15, 75, 75)       #low accruacy
    #cv2.imshow(im, noiseless_image)
    return noiseless_image
def Scan_Image(imgCrop):
        image = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se, iterations=1)
        out_gray = cv2.divide(image, bg, scale=255)


        #noiseless_image = cv2.fastNlMeansDenoising(out_gray,None,20,7,21)

        #sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #sharpen = cv2.filter2D(noiseless_image, -1, sharpen_kernel)


        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] #| cv2.THRESH_BINARY_INV
        out_binary = cv2.fastNlMeansDenoising(out_binary, None, 20, 7, 21)
        out_binary= cv2.GaussianBlur(out_binary,(5,5),0)
        #ho, wo = out_binary.shape
        #out_binary = cv2.resize(out_binary, (wo-150, ho))
        #cv2.imshow(im + '/' + str(x), out_binary)
        return out_binary

