import os
import pandas as pd
import csv
import cv2

jpg_folder_path = r"D:\FYP\imageConvert\jpg_convert\test_data2"

annotation_folder = r"D:\FYP\imageConvert\jpg_convert\test_data2_annotations"

annotation_path = os.listdir(annotation_folder)

def evaluation_data():
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    total_image = len(os.listdir(jpg_folder_path))

    # get predicted data
    predicted_list = pd.read_csv(r"C:\Users\munhe\Dev\darkflow\test_result.csv")
    predicted_list_filter = predicted_list.filter(["FileName"])

    # find true positive

    annotation_count = len(annotation_path)
    
    for item in predicted_list_filter["FileName"]:
        file_name = item.split('.')
        count = 0
        for n, annotation in enumerate(annotation_path):
            annotation_name = annotation.split('.')
            count += 1
            if (annotation_name[0] == file_name[0]):
                # got nose, detect nose
                true_positive += 1
                break
            if (count == annotation_count):
                # no nose, detect nose
                false_positive += 1

    # got nose, detect no nose
    false_negative = annotation_count - true_positive

    # no nose, detect no nose
    true_negative = total_image - annotation_count - false_positive

    print("True Positive: ", true_positive)
    print("False Positive", false_positive)
    print("False Negative", false_negative)
    print("True Negative", true_negative)

    accuracy = (true_positive + true_negative)/(true_positive + false_positive + false_negative + true_negative)
    recall = (true_positive)/(true_positive + false_negative)
    precision = (true_positive)/(true_positive + false_positive)
    fmeasure = (2*recall*precision)/(recall+precision)

    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1-Score: ", fmeasure)

evaluation_data()