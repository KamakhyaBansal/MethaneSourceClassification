
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Define function for heatmap calculation
def grad_cam(model, image, layer_name):

    image = np.expand_dims(image, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model([image])
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


#Install required modules
!pip install grad_cam
!pip install torchinfo

#Load the image for deriving gradcam
import cv2
imagePath = '/content/drive/MyDrive/Data/METER_ML/CAFOs/val_CAFOs/34.75969091_-78.83480443.tif'
image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)

input_tensor = torch.nan_to_num(torch.Tensor(image_ar))

#Function for visualising RGB bands of multi-spectral images
def show_rgb(msi):
  return torch.stack((msi[4],msi[3],msi[2]),dim=2)/torch.max(torch.max(torch.max(msi[3]),torch.max(msi[2])),torch.max(msi[1]))

plt.imshow(show_rgb(input_tensor))
plt.axis('off')

#Select the target layer from model which has to be visualized
target_layers = [model.model.conv1_1]

#Derive heatmap
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]
grayscale_cam = cam(input_tensor.unsqueeze(0), targets=targets)

#Visualize heatmap with the original image
plt.imshow(show_rgb(input_tensor))
plt.imshow(grayscale_cam[0],alpha=0.7)
plt.axis('off')
