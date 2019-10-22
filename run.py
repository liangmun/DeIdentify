import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/tiny-yolo-voc-custom.cfg", "load": -1, "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("61ovu1Es0eLSL1000.jpg")
imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(imgcv)
print(result)

def boxing(imgcv, predictions):
    newImage = np.copy(imgcv)

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
plt.imshow(boxing(imgcv, result))
plt.show()