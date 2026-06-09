import torch
import torchvision
from PIL import Image
import torchvision.transforms as T

def run_baseline_rcnn(image_path, output_path):
    print(f"Запуск Faster R-CNN для: {image_path}")
    # Это вызов реальной нейросети из библиотеки
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    print("Модель загружена успешно.")

if __name__ == "__main__":
    # Просто запускаем функцию
    run_baseline_rcnn("test_images/2023-04-17-10-19-20_mp4-14_jpg.rf.9dd30f521d625d70dc184de83b6eb5be", "runs/baseline_results/faster_rcnn/res.png")




def run_baseline_detr(image_path, output_path):
    print(f"Запуск DETR для: {image_path}")
    # Загружаем предобученную модель DETR
    model = torchvision.models.detection.detr_resnet50(pretrained=True)
    model.eval()
    print("Модель DETR загружена успешно (архитектура на базе трансформеров).")
    # ... логика детекции ...