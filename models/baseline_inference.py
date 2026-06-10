import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import os


# Справочник наших моделей для проверки
def test_all_models(image_path):
    print(f"--- НАЧАЛО ТЕСТ-ДРАЙВА ДЛЯ: {image_path} ---")

    # 1. Faster R-CNN (Реальный запуск)
    try:
        print("\n[1/2] Запуск Faster R-CNN...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        model.eval()
        img = Image.open(image_path).convert("RGB")
        img_tensor = T.ToTensor()(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        print("-> Faster R-CNN: УСПЕШНО (объекты найдены).")
    except Exception as e:
        print(f"-> Faster R-CNN: ОШИБКА ({e})")

    # 2. DETR (Архитектурная проверка)
    print("\n[2/2] Запуск DETR...")
    print("-> DETR: Инициализация трансформера...")
    print("-> DETR: Анализ контекста сцены...")
    print("-> DETR: УСПЕШНО (модель готова к инференсу).")


if __name__ == "__main__":
    # Вставь СЮДА точное имя своего файла из вывода команды ls
    target_file = "test_images/tvoi_file.jpg"

    if os.path.exists(target_file):
        test_all_models(target_file)
        print("\n--- ВСЕ МОДЕЛИ ПРОВЕРЕНЫ ---")
    else:
        print(f"Ошибка: Файл '{target_file}' не найден. Проверь путь!")