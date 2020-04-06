## Dependencies

Keras-Applications==1.0.8,
Keras-Preprocessing==1.1.0,
labelImg==1.8.3,
lxml==4.4.1,
mritopng==2.2,
numpy==1.17.2,
opencv-python==4.1.1.26,
pandas==0.25.3,
pydicom==1.3.0,
scikit-learn==0.21.3,
sklearn==0.0,
tensorboard==1.14.0,
tensorflow==1.14.0,
tensorflow-estimator==1.14.0

## Getting started
Download darkflow from https://github.com/thtrieu/darkflow ,
Download labelImg from https://github.com/tzutalin/labelImg ,

You can choose _one_ of the following three ways to get started with darkflow.

1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

## Download Weights
There are two ways of downloading the pre-trained weights. First, you can download it from the official [YOLO project webpage](https://pjreddie.com/darknet/yolo/) . 
Second, you can download it from here which Darkflow author’s own trained version. But for this case we will use tiny-yolo-voc.weights instead. 
Beside download the weights file, cfg file need to download as well.

## Preparing the Dataset

First things to do is to prepare the dataset, you can get mri scan images from this [website](https://wiki.cancerimagingarchive.net/display/Public/CPTAC-HNSCC#)

Secondly, preprocessing the collected dataset images. As the image dataset downloaded is in DICOM (Digital Imaging and Communications in Medicine) format
In order to be able to train YOLO model, all the images need to convert to either JPG or PNG format. 
We are using mritopng library to convert images. 
Simply “pip install mritopng” to get the library. 
Open convert_image.py, change the folder path of your dataset and the destinate folder path.
```
line 4	folder_path = r"D:\FYP\imageConvert\dicom_image\C3N-01944\09-06-2000-Szyja i krtan z kontrastem-98323\5-SZYJACM  1.0  I26s  3-70627"
line 5	destinate_path = 'D:/FYP/imageConvert/dicom_image/C3N-01944/09-06-2000-Szyja i krtan z kontrastem-98323/5-SZYJACM  1.0  I26s  3-70627/PNG/'
```

## Indicate the annotations
Once all the images converted into JPG or PNG format, 
use this tool labelImg to indicate annotation where an object in the image is located along with its size like (x_top, y_top, x_bottom, y_bottom, width, height). 
After extract the file from github, to install labelimg you need to do this. 
Download labelImg from https://github.com/tzutalin/labelImg.

After install labelimg is done, open cmd and just run labelimg, labelimg program will appear. Following steps is to indicate annotation by drawing bounding box on the image:
1.	Build and launch using the instructions above.
2.	Click 'Change default saved annotation folder' in Menu/File
3.	Click 'Open Dir'
4.	Click 'Create RectBox'
5.	Click and release left mouse to select a region to annotate the rect box
6.	You can use right mouse to drag the rect box to copy or move it
The annotation will be saved to the folder you specify in xml format. You can refer to the below hotkeys to speed up your workflow.

## Predict the Object

In this step, before prediction, trained model should be prepared for inferences. In order to gain a trained model, here are what you should do.
1.	Find a pre-trained model on COCO or VOC dataset (In this case for darkflow, please download tiny-yolo-voc.cfg and tiny-yolo-voc.weights as pre-trained model and configuration)
2.	Create a copy of the configuration file tiny-yolo-voc.cfg and rename it according to your preference tiny-yolo-voc-custom.cfg
3.	Change configurations to fit the model into your own situation
4.	Build the model
5.	Train the model
there should be a file for specifying the model’s configuration. If not, you should change it in the code itself. For darkflow, there is a file (*.cfg) for each model. Ultimately, you should change some values in the last layer. For example, in my case, I wanted only a nose to be detected. So, I changed the value of “classes” to 1 (the number of classes is 20 and 80 for VOC and COCO dataset respectively). Since the number of units(classes) in the last layer, some associated number of parameters should also be changed appropriately. According to darkflow’s document, I needed to change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5) which is 30.

Why should I leave the original tiny-yolo-voc.cfg file unchanged?

When darkflow sees you are loading tiny-yolo-voc.weights it will look for tiny-yolo-voc.cfg in your cfg/ folder and compare that configuration file to the new one you have set with --model cfg/tiny-yolo-voc-custom.cfg. 
In this case, every layer will have the same exact number of weights except for the last two, so it will load the weights into all layers up to the last two because they now contain different number of weights.
```
[convolutional]
size=1
stride=1
pad=1
filters=30
activation=linear

[region]
anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
bias_match=1
classes=1
coords=4
num=5
```
Change the classes and filters values same as above. 

## Building the Model``
Before training the model, change labels.txt in darkflow to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in tiny-yolo-voc-custom.cfg file). 
Then you can enter this command to train the model.
```
python flow --model cfg/tiny-yolo-voc-custom.cfg --load bin/tiny-yolo-voc.weights 
--train --annotation D:/FYP/imageConvert/jpg_convert/nose2/annotations 
--dataset D:/FYP/imageConvert/jpg_convert/nose2 --batch 8 --gpu 0.8 --epoch 35
```
Once the command start running, it will print the architecture of the model with some associated changes which are different from the pre-trained model. 

## Detecting Objects and Draw Boxes on a Picture

To use the trained model and detect nose in DICOM. 
First open improve.py to change the folder path and root path. 
Once folder path and root path is changed. Simply run the file in cmd will auto used the trained model to detect nose and redact out the nose.

### Evaluation
Run the below program in sequence to extract out the data and calculated the evaluation results.
Before execute the program, change the folder path to your specific folder of the annotation

1. evaluation_data.py
2. nose_evaluation.py
3. bndbox_evaluation.oy
4. gruondtruth.py
5. mAP

all the will generate out to CSV file.
