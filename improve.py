import numpy as np
import pandas as pd
import pprint as pp
import pydicom as dicom
import os
from PIL import Image
import mritopng
import collections

from darkflow.net.build import TFNet
import cv2

# Specify the .dcm folder path
# r"D:\FYP\imageConvert\dicom_image\C3N-04611\02-11-2011-K Neck ContrastAdult-21611\7-Neck Venous  150  Br40  S4-46110"
folder_path = r"D:\FYP\imageConvert\dicom_image\C3N-04611\02-11-2011-K Neck ContrastAdult-21611\4-Neck Native  200  Br40  S3-09851"

convert_image_path = r"D:\FYP\imageConvert\dicom_image\C3N-04611\02-11-2011-K Neck ContrastAdult-21611\4-Neck Native  200  Br40  S3-09851\convert"

#Set working directory and locate the image
root = r"D:\FYP\imageConvert\dicom_image\C3N-04611\02-11-2011-K Neck ContrastAdult-21611\4-Neck Native  200  Br40  S3-09851"

#Generate list of filepath and filenames
filepath = []
filename = []
for path, subdirs, _files in os.walk(root):
    for name in _files:
        if '._' not in name:
            filename.append(name)
            filepath.append(os.path.join(path, name))
        else:
            pass

filelist = pd.DataFrame(filename, columns=['filename']) 
filelist['filepath'] = filepath #General file list & directories
fail_list = [] #List of files that failed to be anonymized
image_with_sequence = {}
image_tuple = {}
counter = 0
box_result = {}

options = {"model": "cfg/tiny-yolo-voc-custom.cfg", "load": -1, "threshold": 0.1}

tfnet = TFNet(options)

def calculate_IoU(gt, pt):
    # when the selected picture got no detection
    if gt == 0 or pt == 0:
        return 0

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], pt[0])
    yA = max(gt[1], pt[1])
    xB = min(gt[2], pt[2])
    yB = min(gt[3], pt[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pt[2] - pt[0] + 1) * (pt[3] - pt[1] + 1)

    # compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
    return iou

def boxing(imgcv, predictions):
    #newImage = np.copy(imgcv)
    boxing_result = []

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        #label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.5:
            boxing_result.append(top_x)
            boxing_result.append(top_y)
            boxing_result.append(btm_x)
            boxing_result.append(btm_y)

    return boxing_result

def black_boxing(imgcv, predictions):
    imgcv = cv2.rectangle(imgcv, (predictions[0], predictions[1]), (predictions[2], predictions[3]), (0,0,0), -1)

    return imgcv

def convertImg(dicomImage):
    #print("Image: " + dicomImage)
    outputImg = dicomImage.split('.')
    mritopng.convert_file(os.path.join(folder_path, dicomImage), os.path.join(folder_path, outputImg[0] + '.png'), auto_contrast=True)

    convertedImg = os.path.join(folder_path, outputImg[0] + '.png')
    return convertedImg

for i in range(0, len(filepath)):
    os.chdir('\\'.join(str(filepath[i]).split('\\')[:-1])) #Step 1 - Access the file location

    dataset = dicom.dcmread(filename[i])  #Step 2 - Read the DICOM file
    sequenceNo = dataset.InstanceNumber
    image_with_sequence[filename[i]] = sequenceNo

sorted_dict = sorted(image_with_sequence.items(), key=lambda kv: kv[1])

print("sorted and arrange data: ")
for item in sorted_dict:
    # Convert DICOM image to jpg
    getImg = convertImg(item[0])

    imgcv = cv2.imread(getImg)
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    result = tfnet.return_predict(imgcv)

    image_name = item[0]
    data = dicom.dcmread(image_name)
    image = data.pixel_array
    box = boxing(image, result)
    if len(box) == 0:
        new_list = list(item)
        new_list.append(False)
        new_list.append(0)
        image_tuple[counter] = new_list
    else:
        new_list = list(item)
        new_list.append(True)
        new_list.append(box)
        image_tuple[counter] = new_list
    
    counter+=1

first_index = 0
last_index = 0
biggest_area = 0
biggest_area_list = []

def getFirstIndex(number):
    print("Getting First Index: ")
    for i in range(number, len(image_tuple)):
        if image_tuple[i][2] == True:
            return i
    return 0

first_index = getFirstIndex(0)

print("Getting Last Index: ")
for u in range(0, len(image_tuple))[::-1]:
    if image_tuple[u][2] == True:
        last_index = u
        break

if last_index != 0:
    # compare iou to make sure first index is at the nose
    cmp_count = int((last_index - first_index)*0.1)
    not_first_index_count = 0
    first_index_confirm = False

    # check if first index is correct nose detection by comparing iou with 10% of the total detection images
    while first_index_confirm is False:
        for z in range(1, cmp_count):
            cmp_iou = calculate_IoU(image_tuple[first_index][3], image_tuple[first_index + z][3])
            if cmp_iou < 0.3:
                not_first_index_count += 1
        if not_first_index_count > int(cmp_count/2):
            first_index += 1
            not_first_index_count = 0
        else:
            first_index_confirm = True

    print("Getting biggest area list: ")
    for y in range(first_index, last_index):
        boxing = image_tuple[y][3]
        if boxing == 0:
            boxArea = 0
        else:
            boxArea = (boxing[2] - boxing[0]) * (boxing[3] - boxing[1])
        if (boxArea > biggest_area):
            biggest_area = boxArea
            biggest_area_list = image_tuple[y][3]

    print("Getting iou: ")
    iou = calculate_IoU(image_tuple[first_index][3], image_tuple[last_index][3])
    while iou < 0.3:
        last_index -= 1
        iou = calculate_IoU(image_tuple[first_index][3], image_tuple[last_index][3])

    print("start blacking out: ")
    print(first_index)
    print(last_index)
    for x in range(first_index, last_index + 1):
        try:
            latest_image_name = image_tuple[x][0]
            latest_data = dicom.dcmread(latest_image_name)
            image = latest_data.pixel_array
            if image_tuple[x][2] == True:
                modifiedImage = black_boxing(image, image_tuple[x][3])
            else:
                modifiedImage = black_boxing(image, biggest_area_list)
            latest_data.PixelData = modifiedImage.tobytes()
            #Save & Overwrite original record
            latest_data.save_as(latest_image_name)
            print('Successfully anonymized:',x,'of',' ', 'Filename:',latest_image_name)
        except:
            print("FAILED!")
else:
    print('None anonymized!')