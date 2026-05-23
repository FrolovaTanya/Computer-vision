from ultralytics import YOLO
import yaml
import os

# 1. Проверяем, что датасет собран
if not os.path.exists('C:/Дурачок/1 курс/КЗ/hw5/people_dataset/images'):
    exit(1)

# Считаем количество файлов
image_files = [f for f in os.listdir('C:/Дурачок/1 курс/КЗ/hw5/people_dataset/images') if f.endswith('.jpg')]
label_files = [f for f in os.listdir('C:/Дурачок/1 курс/КЗ/hw5/people_dataset/labels') if f.endswith('.txt')]

print(f"📁 Найдено изображений: {len(image_files)}")
print(f"📁 Найдено файлов разметки: {len(label_files)}")

if len(image_files) < 10:
    exit(1)

# 2. Создаём конфигурационный файл для YOLO
dataset_config = {
    'path': 'C:/Дурачок/1 курс/КЗ/hw5/people_dataset',   
    'train': 'images',            
    'val': 'images',              
    'nc': 1,                      
    'names': ['person']           
}

with open('people.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

print("Создан файл people.yaml")

# 3. Загружаем предобученную YOLOv8 nano
model = YOLO('yolov8n.pt')  
print("Модель загружена")

# 4. Запускаем обучение
results = model.train(
    data='people.yaml',       
    epochs=50,                
    imgsz=640,                
    batch=8,                  
    name='people_detector',   
    patience=10,              
    verbose=True              
)

print("\nОбучение завершено!")

