import gradio as gr
import numpy as np
from training import SimpleCNN
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt

#Import dict state of trained model
trained_model = SimpleCNN()
trained_model.load_state_dict(torch.load('final_model.pth'))
trained_model.eval()

#Prepare transformations to match model's train dataset
transform = v2.Compose(
    [   v2.ToImage() ,
        v2.Resize(size=(28,28)),
        v2.ToDtype(torch.float32, scale=True)  
    ]
)

#Create classification function for the application
def classify(image):
    image = np.array(image['composite'])
    image = 255 - image
    image = transform(image)
    image = image.unsqueeze(0)
    trained_model.eval()
    with torch.no_grad():
        output = trained_model(image)
        probs =  torch.softmax(output, dim=1)[0]
        probs_dict = {str(i): float(probs[i]) for i in range(10)}
        label = max(probs_dict,key=probs_dict.get)
    return "Classification: " + label

 #Create app's interface   
demo = gr.Interface(
    title="Digits classification with neural network",
    fn=classify,
    inputs=gr.Sketchpad(type="numpy",image_mode="L",label="Image for classification",show_label=True),
    outputs=["text"],
    api_name="predict"
)

#Run the app
demo.launch()