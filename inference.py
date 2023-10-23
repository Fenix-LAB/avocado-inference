import torch
import os
import cv2
import pandas

# Load fine-tuned custom model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'model-avocado-detector/avocado_model.pt',
                        force_reload=True, trust_repo=True)

image_path = 'avocado-example.jpg'
    
# Declaring some variables    
TABLE_CONFIDENCE = 0.50
CELL_CONFIDENCE = 0.50
OUTPUT_DIR = 'output'

# Bounding Boxes color scheme
ALPHA = 0.2
TABLE_BORDER = (0, 0, 255)
CELL_FILL = (0, 0, 200)
CELL_BORDER = (0, 0, 255)

os.makedirs(OUTPUT_DIR, exist_ok=True)
    
# Run the Inference and draw predicted bboxes
results = model(image_path)
df = results.pandas().xyxy[0]
table_bboxes = []
cell_bboxes = []
for _, row in df.iterrows():
    if row['class'] == 0 and row['confidence'] > TABLE_CONFIDENCE:
        table_bboxes.append([int(row['xmin']), int(row['ymin']),
                             int(row['xmax']), int(row['ymax'])])

    if row['class'] == 1 and row['confidence'] > CELL_CONFIDENCE:
        cell_bboxes.append([int(row['xmin']), int(row['ymin']),
                            int(row['xmax']), int(row['ymax'])])

image = cv2.imread(image_path)
overlay = image.copy()
for table_bbox in table_bboxes:
    cv2.rectangle(image, (table_bbox[0], table_bbox[1]),
                  (table_bbox[2], table_bbox[3]), TABLE_BORDER, 1)

for cell_bbox in cell_bboxes:
    cv2.rectangle(overlay, (cell_bbox[0], cell_bbox[1]),
                  (cell_bbox[2], cell_bbox[3]), CELL_FILL, -1)
    cv2.rectangle(image, (cell_bbox[0], cell_bbox[1]),
                  (cell_bbox[2], cell_bbox[3]), CELL_BORDER, 1)

image_new = cv2.addWeighted(overlay, ALPHA, image, 1-ALPHA, 0)
image_filename = image_path.split('/')[-1]
cv2.imwrite(f'{OUTPUT_DIR}/{image_filename}', image_new)