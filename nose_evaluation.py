import os
import pandas as pd
import csv
import cv2

jpg_folder_path = r"D:\FYP\imageConvert\jpg_conver2\test_data"

annotation_folder = r"D:\FYP\imageConvert\jpg_conver2\test_data_annotations"

annotation_path = os.listdir(annotation_folder)

with open('false_negative.csv', 'w', newline='') as test_result:
    writer = csv.writer(test_result)
    writer.writerow(["FileName"])

def evaluation_data():
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    total_image = len(os.listdir(jpg_folder_path))

    # get predicted data
    predicted_list = pd.read_csv(r"C:\Users\munhe\Dev\darkflow\test_result2.csv")
    predicted_list_filter = predicted_list.filter(["FileName"])

    # find true positive

    annotation_count = len(annotation_path)
    predicted_count = len(predicted_list_filter["FileName"])
    
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

    # false negative filename
    for n, item in enumerate(annotation_path):
        name = item.split('.')
        count = 0
        for predicted in predicted_list_filter["FileName"]:
            predicted_name = predicted.split('.')
            count += 1
            if (predicted_name[0] == name[0]):
                break
            if (count == predicted_count):
                with open(r"C:\Users\munhe\Dev\darkflow\false_negative.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([name])


    # no nose, detect no nose
    true_negative = total_image - annotation_count - false_positive

    print("True Positive: ", true_positive)
    print("False Positive", false_positive)
    print("False Negative", false_negative)
    print("True Negative", true_negative)

    accuracy = (true_positive + true_negative)/(true_positive + false_positive + false_negative + true_negative)
    # sensitivity = recall
    recall = (true_positive)/(true_positive + false_negative)
    # specificity
    # specificity = 
    # positive predictive value
    precision = (true_positive)/(true_positive + false_positive)
    # negative predictive value
    # negative_predictive_value =
    # accuracy/kappa
    fmeasure = (2*recall*precision)/(recall+precision)

    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1-Score: ", fmeasure)

evaluation_data()