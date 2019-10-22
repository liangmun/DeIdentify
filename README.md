# DeIdentify

# Run the command below for testing or training
** Train with Python flow (SoccerBall)**
python flow --model cfg/tiny-yolo-voc-custom.cfg --load bin/tiny-yolo-voc.weights --train --annotation soccer_ball_data/annotations/ --dataset soccer_ball_data/images/ --batch 6 --gpu 0.8 --epoch 150

** Train with Python flow (FacialFeature) **
python flow --model cfg/tiny-yolo-voc-custom.cfg --load bin/tiny-yolo-voc.weights --train --annotation facial_features/annotations/ --dataset facial_features/dataset/ --batch 4 --gpu 0.8 --epoch 250

** Run with Python flow with image dir **
python flow --imgdir face_img/ --model cfg/tiny-yolo-voc-custom.cfg --load -1 --threshold 0.1 --gpu 0.8

