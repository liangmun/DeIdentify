import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint as pp
import matplotlib.pyplot as plt
import pydicom as dicom
import os
from PIL import Image
import mritopng

from darkflow.net.build import TFNet
import cv2

# Specify the .dcm folder path
# r"D:\FYP\imageConvert\dicom_image\C3N-04611\02-11-2011-K Neck ContrastAdult-21611\7-Neck Venous  150  Br40  S4-46110"
folder_path = r"D:\FYP\imageConvert\dicom_image\C3N-01944\09-06-2000-Szyja i krtan z kontrastem-98323\5-SZYJACM  1.0  I26s  3-70627"

convert_image_path = r"D:\FYP\imageConvert\dicom_image\C3N-01944\09-06-2000-Szyja i krtan z kontrastem-98323\5-SZYJACM  1.0  I26s  3-70627\convert"

#Set working directory and locate the image
root = r"D:\FYP\imageConvert\dicom_image\C3N-01944\09-06-2000-Szyja i krtan z kontrastem-98323\5-SZYJACM  1.0  I26s  3-70627"

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

options = {"model": "cfg/tiny-yolo-voc-custom.cfg", "load": -1, "threshold": 0.1}

tfnet = TFNet(options)

def boxing(imgcv, predictions):
    #newImage = np.copy(imgcv)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        #label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.5:
            imgcv = cv2.rectangle(imgcv, (top_x, top_y), (btm_x, btm_y), (0,0,0), -1)
            #newImage = cv2.putText(newImage, label, (btm_x, btm_y-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

    return imgcv

# def convertImg(dicomImage):
#     print("Image: " + dicomImage)
#     outputImg = dicomImage.split('.')
#     mritopng.convert_file(os.path.join(folder_path, dicomImage), os.path.join(folder_path, outputImg[0] + '.jpg'), auto_contrast=True)

#     convertedImg = os.path.join(folder_path, outputImg[0] + '.jpg')
#     return convertedImg

def convertImg(dicomImage):
    ds = dicom.dcmread(os.path.join(folder_path, dicomImage))
    pixel_array_numpy = ds.pixel_array

    image = dicomImage.split('.')
    image = image[0] + '.jpg'

    if not os.path.exists(convert_image_path):
        try:
            os.makedirs(convert_image_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    convert_image = os.path.join(convert_image_path, image)
    cv2.imwrite(convert_image, pixel_array_numpy)

    return convert_image

for i in range(0, len(filepath)):
    os.chdir('\\'.join(str(filepath[i]).split('\\')[:-1])) #Step 1 - Access the file location

    # Convert DICOM image to jpg
    getImg = convertImg(filename[i])

    imgcv = cv2.imread(getImg)
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    result = tfnet.return_predict(imgcv)
    #print(result)

    try:
        dataset = dicom.dcmread(filename[i])  #Step 2 - Read the DICOM file
        image = dataset.pixel_array
        modifiedImage = boxing(image, result)
        dataset.PixelData = modifiedImage.tobytes()
        #Save & Overwrite original record
        dataset.save_as(str(filepath[i].split("\\")[-1]))
        print('Successfully anonymized:',i,'of',len(filename)-1,' ', 'Filename:',filename[i])
    except:
        print("FAILED!")
