
cd yolov5-master
python train.py --img 512 \
                 --batch 16 \
                 --epochs 10 \
                 --data data.yaml \
                 --weights yolov5s.pt \
                 --save-period 1\
                 --project kaggle-siim-covid
                 