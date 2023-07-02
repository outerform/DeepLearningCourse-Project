from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--data', type=str, default='data.yaml', help='data.yaml path')
parser.add_argument('--fold', type=int, default=0, help='fold')

args = parser.parse_args()
device = 'cuda'
model = YOLO(model = 'yolov8x-cls.pt')
model.to(device)
model.train(model = 'yolov8x-cls.pt',
            data = args.data,
            epochs = args.epochs,
            batch = args.batch_size,
            imgsz = args.image_size,
            save_period = 10,
            seed = 42,
            pretrained = True,
            num_worker = 16,
            project = "siim-covid19-detection",
            name = f"YOLOv8_{args.image_size}_fold{args.fold}")
            