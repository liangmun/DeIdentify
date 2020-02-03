import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint as pp
import matplotlib.pyplot as plt
import pydicom as dicom
import os
from PIL import Image
import mritopng
import csv

from darkflow.net.build import TFNet
import cv2

# Specify the .dcm folder path
# r"D:\FYP\imageConvert\dicom_image\C3N-04611\02-11-2011-K Neck ContrastAdult-21611\7-Neck Venous  150  Br40  S4-46110"
folder_path = r"D:\FYP\imageConvert\jpg_conver2\test_data"

output_path = r"D:\FYP\imageConvert\jpg_conver2\test_data_out"

#Set working directory and locate the image
root = r"D:\FYP\imageConvert\jpg_conver2\test_data"

counter = 0
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

with open('test_result2.csv', 'w', newline='') as test_result:
    writer = csv.writer(test_result)
    writer.writerow(["ID", "FileName", "Result", "Confidence"])

options = {"model": "cfg/tiny-yolo-voc-custom.cfg", "load": -1, "threshold": 0.1}

tfnet = TFNet(options)

def boxing(itemname, imgcv, predictions):
    #newImage = np.copy(imgcv)
    boxing_result = []

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        #label = result['label'] + " " + str(round(confidence, 3))
        boxing_result.append(top_x)
        boxing_result.append(top_y)
        boxing_result.append(btm_x)
        boxing_result.append(btm_y)

        if confidence > 0.5:
            with open(r"C:\Users\munhe\Dev\darkflow\test_result2.csv", 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow([counter, itemname, boxing_result, confidence])
            imgcv = cv2.rectangle(imgcv, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)

    return imgcv

# def convertImg(dicomImage):
#     print("Image: " + dicomImage)
#     outputImg = dicomImage.split('.')
#     mritopng.convert_file(os.path.join(folder_path, dicomImage), os.path.join(folder_path, outputImg[0] + '.jpg'), auto_contrast=True)

#     convertedImg = os.path.join(folder_path, outputImg[0] + '.jpg')
#     return convertedImg

for i in range(0, len(filepath)):
    os.chdir('\\'.join(str(filepath[i]).split('\\')[:-1])) #Step 1 - Access the file location

    print(filename[i])

    counter += 1
    imgcv = cv2.imread(filename[i])
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    result = tfnet.return_predict(imgcv)

    try:
        #dataset = dicom.dcmread(filename[i])  #Step 2 - Read the DICOM file
        # image = dataset.pixel_array
        modifiedImage = boxing(filename[i], imgcv, result)
        # dataset.PixelData = modifiedImage.tobytes()
        # #Save & Overwrite original record
        # dataset.save_as(str(output_path))
        cv2.imwrite(os.path.join(output_path, filename[i]), modifiedImage)
        print('Successfully anonymized:',i,'of',len(filename)-1,' ', 'Filename:',filename[i])
    except Exception as e:
        print(e)
