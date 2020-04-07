import os
import pandas as pd
import csv
import cv2
import xml.etree.ElementTree as ET
import xml.dom.minidom

jpg_folder_path = r"D:\FYP\imageConvert\jpg_conver2\test_data"

annotation_folder = r"D:\FYP\imageConvert\jpg_conver2\test_data_annotations"

annotation_path = os.listdir(annotation_folder)

# get predicted data
predicted_list = pd.read_csv(r"C:\Users\munhe\Dev\darkflow\test_result2.csv")
predicted_list_filter = predicted_list.filter(["FileName", "Result"])

with open('bndbox_evaluation.csv', 'w', newline='') as test_result:
    writer = csv.writer(test_result)
    writer.writerow(["FileName", "Ground Truth", "Predicted Result", "IoU"])

# calculate IoU (Intersection Over Union)
def calculate_IoU(annotation_name, fileName_count):
    pt = []
    doc = xml.dom.minidom.parse(annotation_folder + "\\" + annotation_name + ".xml")
    xmin = doc.getElementsByTagName("xmin")[0]
    ymin = doc.getElementsByTagName("ymin")[0]
    xmax = doc.getElementsByTagName("xmax")[0]
    ymax = doc.getElementsByTagName("ymax")[0]
    
    # ground truth bounding box
    gt = [int(xmin.firstChild.data), int(ymin.firstChild.data), int(xmax.firstChild.data), int(ymax.firstChild.data)]

    # predicted bounding box by model
    for bndbox in predicted_list_filter["Result"][fileName_count:]:
        bnbarray = eval(bndbox)
        for item in bnbarray:
            pt.append(item)
        break
    
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
 
    with open(r"C:\Users\munhe\Dev\darkflow\bndbox_evaluation.csv", 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([annotation_name + ".xml", gt, pt, iou])
    
	# return the intersection over union value
    return iou

def evaluation_data():
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # total_image = len(os.listdir(jpg_folder_path))

    # find true positive
    annotation_count = len(annotation_path)
    fileName_count = 0
    for item in predicted_list_filter["FileName"]:
        file_name = item.split('.')
        count = 0
        for n, annotation in enumerate(annotation_path):
            annotation_name = annotation.split('.')
            count += 1
            if (annotation_name[0] == file_name[0]):
                # got nose, detect nose
                iou = calculate_IoU(annotation_name[0], fileName_count)
                fileName_count += 1
                if (iou > 0.5):
                    true_positive += 1
                break
            if (count == annotation_count):
                # no nose, detect nose
                false_positive += 1
                fileName_count += 1

    # got nose, detect no nose
    false_negative = annotation_count - true_positive

    print("True Positive: ", true_positive)
    print("False Positive", false_positive)
    print("False Negative", false_negative)

    recall = (true_positive)/(true_positive + false_negative)
    precision = (true_positive)/(true_positive + false_positive)
    fmeasure = (2*recall*precision)/(recall+precision)

    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1-Score: ", fmeasure)


evaluation_data()