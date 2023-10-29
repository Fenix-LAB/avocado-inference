# Modelo de IA para la deteccion de Palta Hass en mal estado

## Descripcion del proyecto
Para el modelo de red neuronal se utiliza YoloV7, el cual es un modelo de deteccion de objetos en tiempo real, el cual se entrena con un dataset de imagenes de palta hass en buen estado y en mal estado, para que el modelo pueda detectar si la palta esta en buen estado o no.

## Entrenamiento del modelo
Para el entrenamiento del modelo se realizo este repositorio que se encarga unicamente de entrenar el modelo de YoloV7, para esto se debe clonar el repositorio de YoloV7.

## Instalacion
Para instalar el proyecto se debe clonar el repositorio y luego instalar las dependencias necesarias para el proyecto, para esto se debe ejecutar el siguiente comando:

Clonar el repositorio de YoloV7
```bash
git clone https://github.com/WongKinYiu/yolov7.git
```

Instalar las dependencias necesarias
```bash
pip install -r requirements.txt
```

Para instalar Pytorch se debe ejecutar el siguiente comando:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Uso
Para correr el entrenamiento del modelo se debe ejecutar el siguiente comando:

```bash
python .\yolov7\train.py --workers 1 --device 0 --batch-size 4 --data .\avocado-detector-v1.v2i.yolov7pytorch\data.yaml --img 640 640 --cfg .\yolov7\cfg\training\yolov7.yaml --weights 'yolov7.pt' --name avocado --hyp .\yolov7\data\hyp.scratch.p5.yaml --epochs 4
```