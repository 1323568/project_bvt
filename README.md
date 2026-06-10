
# Исследование архитектур детекции объектов в Minecraft

Данный проект посвящен сравнительному анализу различных нейросетевых архитектур для решения задачи детекции объектов в среде Minecraft. Основная цель — оценить эффективность современных моделей компьютерного зрения применительно к специфике игровых данных.

## Пример работы модели
Ниже представлен пример детекции игрового объекта (Скелета) в среде Minecraft:
<img width="640" height="640" alt="detection_example" src="https://github.com/user-attachments/assets/a88979ff-e5f8-4c39-a98d-c12f8e4fd6b3" />

## Выполненная работа
- Подготовка и разметка датасета Minecraft;
- Конфигурация и обучение моделей YOLO(v5, v8, v11) и SSD;
- Проведение инференса моделей DETR и Faster R-CNN для сравнения;
- Анализ метрик и визуализация динамики обучения (Loss/mAP).

## Использованные архитектуры
- YOLOv8
- YOLOv11
- YOLOv5
- Faster R-CNN
- DETR
- SSD300

## Результаты эксперимента
Сравнительная таблица производительности моделей на валидационном наборе:

| Архитектура | mAP | Precision | Recall |
| :--- | :---: | :---: | :---: |
| DETR | 0.8741 | 0.9391 | 0.8236 |
| Faster R-CNN | 0.8542 | 0.9242 | 0.8032 |
| YOLOv8 | 0.8244 | 0.8927 | 0.7778 |
| YOLOv5 | 0.7847 | 0.8582 | 0.7395 |
| SSD300 | 0.7350 | 0.8019 | 0.6871 |

## Визуализация процесса обучения
Ниже представлены графики динамики функций потерь (Loss) и метрик:

- **YOLOv11:** Отчет по результатам обучения (Loss, Precision, Recall)
  ![YOLOv11 Training Results](image_0ab1bf.png)
  
- **Сравнительные графики Loss:**
  - YOLOv11 Loss
    ![YOLOv11](runs/detect/trained_models/yolov11_minecraft/results.png)
  - YOLOv8 Loss
    ![YOLOv8](benchmark_analysis/plots/yolov8_loss_plot.png)
  - DETR Loss
    ![DETR](benchmark_analysis/plots/detr_loss_plot.png)
  - Faster R-CNN Loss
    ![Faster R-CNN](benchmark_analysis/plots/faster_rcnn_loss_plot.png)
  - SSD300 Loss
    ![SSD300](benchmark_analysis/plots/ssd300_loss_plot.png)


## Запуск проекта
1. Установите необходимые зависимости: `pip install -r requirements.txt`
2. Для выполнения детекции на изображениях используйте скрипт: `python src/baseline_inference.py --source test_images/`
