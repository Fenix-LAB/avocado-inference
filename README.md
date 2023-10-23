# Inferencia con modelo de PyTorch

## Descripción
Para conseguir una inferencia se realiza el siguiente proceso:

- Dataset
- Entrenamiento
- Inferencia

### Dataset
Un dataset es un conjunto de datos que se utilizan para entrenar un modelo. En este caso se utiliza un dataset de imágenes de paltas en buen estado y mal estado. El dataset se encuentra en la carpeta `data` y se divide en 3 carpetas: `train`, `test` y `validation`. Cada una de estas carpetas contiene imágenes de paltas etiquetadas de acuerdo a su condicion.

Para la creacion del dataset se uso la herramienta de roboflow: https://roboflow.com/

### Entrenamiento
Para el entrenamiento se utiliza la tecnica de transfer learning, la cual consiste en utilizar un modelo pre-entrenado y re-entrenarlo con un dataset propio. En este caso se utiliza el modelo `yolov7`.

Información sobre el modelo: https://github.com/WongKinYiu/yolov7

### Inferencia
Para la inferencia se utiliza el modelo entrenado y se ejecuta en tiempo real. Para lograr esto se utiliza la funcion de torch `torch.hub.load` la cual permite cargar un modelo de la libreria de modelos de PyTorch.

## Instalación
### Entorno virtual
Para la instalación de las dependencias se recomienda utilizar un entorno virtual. Para crear un entorno virtual se debe ejecutar el siguiente comando:
```bash
python -m venv venv # en windows se debe ejecutar python -m venv venv
```
Para activar el entorno virtual se debe ejecutar el siguiente comando:
```bash
source venv/bin/activate # en windows se debe ejecutar venv\Scripts\activate
```

### Requerimientos
Para instalar las dependencias se debe ejecutar el siguiente comando:
```bash
pip install -r requirements.txt
```

### Instalación de PyTorch
Para instalar PyTorch en un equipo con GPU se debe ejecutar el siguiente comando:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

Para instalar PyTorch en un equipo sin GPU se debe ejecutar el siguiente comando:
```bash
pip3 install torch torchvision
```
