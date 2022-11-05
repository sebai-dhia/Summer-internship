from fileinput import filename
import numpy as np
import cv2
import easyocr
from functions import *
reader = easyocr.Reader(['ar'])

###################################################    Template Image      ################## 
def OCR(path, filename):  
    imgScan=CIN_Template(path)
    imago=cv2.imread(path)
    wa,ha,_=imago.shape
    #print(wa,ha)
    #cv2.imwrite(f'static\\predicted\\{filename}',cv2.resize(imgScan, (ha, wa)))
    
    crop=Crop_Black_Edges(imgScan)
    if len(crop.shape)==3:
         noiseless_image=Denoise_Image(crop)
    else:
         noiseless_image= crop.copy()   
    ########################################### Template Matching for CIN Reginon #####################
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('static\\template_cin\\tem2.png', 0)
    wt, ht = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    (_, max_val, _, max_loc) = cv2.minMaxLoc(res)
    #threshold = 0.39
    loc = np.where(res == max_val)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(noiseless_image, pt, (pt[0] + wt, pt[1] + ht), (0, 0, 255), 2)
        roi=roi_dynamic((pt[0] + wt, pt[1] + ht),pt)
        #print(f'################## Extract data from {im} #############')
        #print(f'[{pt},({pt[0] + wt},{pt[1] + ht})]')
        #cv2.imshow(im, crop)
        #########################################################################################################
        imgShow = noiseless_image.copy()
        imgMask = np.zeros_like(imgShow)
        result=[]
        #print(w_imgsh,h_imgsh)
        #print('################## Extract data from image #############')
        for x, r in enumerate(roi):
            cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (255,60, 0), 2)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
            #cv2.imshow(im+"2",imgShow)
            imgCrop = crop[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            #cv2.imshow('im'+'/'+str(x), imgCrop)
            
            
            if ha>wa:
                cv2.imwrite(f'static\\predicted\\{filename}',cv2.resize(imgShow, (ha, wa)))
            else:
                cv2.imwrite(f'static\\predicted\\{filename}',cv2.resize(imgShow, (wa, ha)))   
    
            ##################################  Scan Image ###################################################
            out_binary=Scan_Image(imgCrop)
            ###################################### Canny Edge ####################################
          
  
            
            if r[3] == 'CIN':
                cin=reader.readtext(out_binary ,detail = 0,paragraph=True,allowlist="0123456789")
                if cin != []:
                   result.append(cin[0]) 
                else :
                   result.append(" ")   
            elif r[3] == 'Bd':
                Bd=reader.readtext(out_binary, detail = 0,paragraph=True ,blocklist="٠١٢٣٤٥٦٧٨٩:;,؛“©!^<>$!?+-|[]/*\#)»{(}.")
                if Bd != []:
                   result.append(Bd[0]) 
                else :
                   result.append(" ") 
            elif r[3] == 'fname':
                out_binary = cv2.resize(out_binary, (0, 0), fx=1.2, fy=1.2)
                fname=reader.readtext(out_binary, detail = 0,paragraph=True,blocklist="0123456789٠١٢٣٤٥٦٧٨٩؛:;,“©!^<>$!?+-|[]/*\#){»}(.")
                if fname != []:
                   result.append(fname[0]) 
                else :
                   result.append(" ")  
            elif r[3] == 'name':
                #out_binary = cv2.resize(out_binary, (0, 0), fx=1.5, fy=1.5)
                name=reader.readtext(out_binary, detail = 0,paragraph=True,blocklist="0123456789٠١٢٣٤٥٦٧٨٩:;؛,“©!^<>$!?+-|[]/*\#)»(.",rotation_info=[0,10,-10])
                if name != []:
                   result.append(name[0]) 
                else :
                   result.append(" ")  
            elif r[3] == 'place':
                #out_binary = cv2.resize(out_binary, (0, 0), fx=0.5, fy=0.5)
                place=reader.readtext(out_binary, detail = 0,paragraph=True,blocklist="0123456789٠١٢٣٤٥٦٧٨٩:؛;,“©!^<>$+-|[]/*\#)»(.")
                if place != []:
                   result.append(place[0]) 
                else :
                   result.append(" ") 
            elif r[3] == 'tree':
                #out_binary = cv2.resize(out_binary, (0, 0), fx=1.5, fy=1.5)
                tree=reader.readtext(out_binary, detail = 0,paragraph=True,blocklist="0123456789٠١٢٣٤٥٦٧٨٩؛:;,“©!^<>$+-|[]/*\#?!)»(.",rotation_info=[0,10,-10])
                if tree != []:
                   result.append(tree[0]) 
                else :
                   result.append(" ")  
                
            #result="".join(result).split('\n')
            #result.pop()
            ##Data.append(pytesseract.image_to_string(imgCrop))
        
    # with open('Output.csv','a+') as f:
    #    for data in Data:
    #        f.write((str(im)+','+ str(data)))
    #strg=f'Numb:{result[0]}-----Fname:{result[1]}-----Name:{result[2]}-----tree:{result[3]}-----Bd:{result[3]}-----Place:{result[4]}'
    #cv2.waitKey(0)
    #print(result)
    return result
