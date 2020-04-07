import os
import pandas as pd
import csv
import cv2
import xml.etree.ElementTree as ET
import xml.dom.minidom

annotation_folder = r"D:\FYP\imageConvert\jpg_conver2\test_data_annotations"

annotation_path = os.listdir(annotation_folder)

# get predicted data
predicted_list = pd.read_csv(r"C:\Users\munhe\Dev\darkflow\test_result.csv")
predicted_list_filter = predicted_list.filter(["FileName", "Result"])

with open('groundtruth.csv', 'w', newline='') as test_result:
    writer = csv.writer(test_result)
    writer.writerow(["FileName", "xmin", "ymin", "xmax", "ymax", "xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred"])

fileName_count = 0

# for n, annotation in enumerate(annotation_path):
#     pt = []
#     annotation_name = annotation.split('.')
#     doc = xml.dom.minidom.parse(annotation_folder + "\\" + annotation_name[0] + ".xml")
#     xmin = doc.getElementsByTagName("xmin")[0]
#     ymin = doc.getElementsByTagName("ymin")[0]
#     xmax = doc.getElementsByTagName("xmax")[0]
#     ymax = doc.getElementsByTagName("ymax")[0]

#     xmin = int(xmin.firstChild.data)
#     ymin = int(ymin.firstChild.data)
#     xmax = int(xmax.firstChild.data)
#     ymax = int(ymax.firstChild.data)


#     # predicted bounding box by model
#     for item in predicted_list_filter["FileName"]:
#         file_name = item.split('.')
#         if (annotation_name[0] == file_name[0]):
#             for bndbox in predicted_list_filter["Result"][fileName_count:]:
#                 bnbarray = eval(bndbox)
#                 for item in bnbarray:
#                     pt.append(item)
#                 break
#             fileName_count = 0
#             break
#         else:
#             fileName_count += 1

#     if len(pt) == 0:
#         break
#     else:
#         with open(r"C:\Users\munhe\Dev\darkflow\groundtruth.csv", 'a', newline='') as fd:
#             writer = csv.writer(fd)
#             writer.writerow([annotation_name[0] + ".xml", xmin, ymin, xmax, ymax, pt[0], pt[1], pt[2], pt[3]])
annotation_count = len(annotation_path)
for item in predicted_list_filter["FileName"]:
    file_name = item.split('.')
    count = 0
    for n, annotation in enumerate(annotation_path):
        pt = []
        annotation_name = annotation.split('.')
        doc = xml.dom.minidom.parse(annotation_folder + "\\" + annotation_name[0] + ".xml")
        xmin = doc.getElementsByTagName("xmin")[0]
        ymin = doc.getElementsByTagName("ymin")[0]
        xmax = doc.getElementsByTagName("xmax")[0]
        ymax = doc.getElementsByTagName("ymax")[0]

        xmin = int(xmin.firstChild.data)
        ymin = int(ymin.firstChild.data)
        xmax = int(xmax.firstChild.data)
        ymax = int(ymax.firstChild.data)
        count += 1
        if (annotation_name[0] == file_name[0]):
            # got nose, detect nose            
            for bndbox in predicted_list_filter["Result"][fileName_count:]:
                bnbarray = eval(bndbox)
                for bnb in bnbarray:
                    pt.append(bnb)
                break
            fileName_count += 1
            with open(r"C:\Users\munhe\Dev\darkflow\groundtruth.csv", 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow([annotation_name[0] + ".xml", xmin, ymin, xmax, ymax, pt[0], pt[1], pt[2], pt[3]])
            break
        if (count == annotation_count):
            # no nose, detect nose
            fileName_count += 1