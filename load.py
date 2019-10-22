import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

from darkflow.net.build import TFNet
import cv2

options = {"model": "C:/Users/munhe/Dev/darkflow/cfg/tiny-yolo-voc-custom.cfg",
           "load": -1,
           "gpu": 1.0}
		   
tfnet2 = TFNet(options)

tfnet2.load_from_ckpt()

original_img = cv2.imread("willsmith.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)

def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 230, 0), 1, cv2.LINE_AA)

    return newImage

fig = plt.subplots(figsize=(20, 10))
plt.imshow(boxing(original_img, results))
plt.show()