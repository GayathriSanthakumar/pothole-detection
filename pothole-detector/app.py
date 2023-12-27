import gradio as gr
import cv2
import requests
import os
 
from ultralytics import YOLO
 
model = YOLO('best.pt')
path  = [['image_0.jpg'], ['image_1.jpg']]

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Pothole detector",
    examples=path,
    cache_examples=False,
)


gr.TabbedInterface(
    [interface_image],
    tab_names=['Image inference']
).queue().launch()
