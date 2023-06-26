
cd yolov5-master
python train.py --img 512 \
                 --batch 32 \
                 --epochs 100 \
                 --data data.yaml \
                 --weights yolov5x.pt\
                 --save-period 1\
                 --project kaggle-siim-covid
                 