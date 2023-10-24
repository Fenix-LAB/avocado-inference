import torch
import cv2
from PIL import Image
import numpy as np
import pathlib
gen_path = pathlib.Path.cwd()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
  
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'model-avocado-detector/avocado_model.pt',
                        force_reload=False, trust_repo=True)
model.eval()
model.conf = 0.8
model.iou = 0.5
cam = cv2.VideoCapture(0)
  
while(True): 
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = frame[:, :, [2,1,0]]
    frame = Image.fromarray(frame) 
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    result = model(frame,size=640)
    cv2.imshow('YOLO', np.squeeze(result.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cam.release()
cv2.destroyAllWindows()