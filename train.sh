


# python yolov5-master/train.py --img 640 \
#                  --batch 24 \
#                  --epochs 50 \
#                  --data data_640.yaml \
#                  --weights yolov5x.pt\
#                  --save-period 1\
#                  --seed 42\
#                  --project kaggle-siim-covid

python yolov8.py --image_size 640 --data data_640.yaml --batch_size 24
python yolov8.py --image_size 512 --data data_512.yaml

python effnetv2.py --image_size 640 --save_path model_640 --batch_size 10 --epochs 15

python 2_class.py --image_size 640 --save_path model_2class_640 --batch_size 10 --epochs 15