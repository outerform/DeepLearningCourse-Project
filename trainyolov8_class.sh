# for i in {0..4} 
# do
#     python yolov8.py --image_size 640 --data ./data_640/fold$i/data.yaml --batch_size 24 --fold $i --epochs 100
# done

for i in {0..4} 
do
    python yolov8.py --image_size 512 --data ./data_classfy/fold$i/data.yaml --batch_size 32 --fold $i --epochs 100
done

