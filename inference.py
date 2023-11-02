import torch
import cv2
from PIL import Image
import numpy as np

# harware acceleration with CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# load model from local directory
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'model-avocado-detector/avocado_model.pt',
#                         force_reload=False, trust_repo=True)

# model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'model-avocado-ripeness/avocado_ripeness_detector.pt',
#                         force_reload=False, trust_repo=True)

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'model-avocado-export/best_263.pt',
                        force_reload=False, trust_repo=True)

# 1 - Palta Optima (verde)
# 2 - Palta No Optima (maduras)
# 3 - Palta Descartable (pasadas)


# revisar enfermas 

# checar lo de camara de celular

# set model to evaluation mode
# model.eval()
model.conf = 0.4 # confidence threshold (0-1)

# La precision durante el entrenamiento (0.85), es diferente a la precision de la inferencia (0.4)

# initialize webcam
cam = cv2.VideoCapture(0) # camara de la laptop, 1 para camara externa

while(True): 
    # read frame from webcam
    ret, frame = cam.read()
    # flip frame horizontally
    frame = cv2.flip(frame, 1)
    # convert frame to RGB
    frame = frame[:, :, [2,1,0]] # 
    # convert frame to PIL Image
    frame = Image.fromarray(frame) 
    # convert frame to BGR
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    # run inference on frame
    result = model(frame,size=640)
    # show frame
    cv2.imshow('Palta', np.squeeze(result.render()))

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release webcam
cam.release()
# close all windows
cv2.destroyAllWindows()