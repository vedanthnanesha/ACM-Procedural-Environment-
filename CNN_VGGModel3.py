#importing some libraries and tools
import torch
import torchvision
import torchvision.transforms as transf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#setting up the basic parameters
epochs = 6
batchsize = 50
learnrate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #checking the system available for use
print(device)

#Transforms for the dataset, including resizeing and the use of data augmentations
trans_train = transf.Compose([
    transf.Resize(size=(224, 224)),   #this is the size of the VGG input layer
    transf.RandomHorizontalFlip(),    #randomly flips an image horizontally
    transf.RandomRotation(10),        #randomly rotates by upto 10 degrees
    transf.ToTensor(),
    transf.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),])    
trans_test = transf.Compose([
    transf.Resize(size=(224, 224)),
    transf.ToTensor(),
    transf.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),])

#Downloading and Loading the data in 2 sets - Training data and Testing Data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download= True, transform = trans_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download= True, transform = trans_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = batchsize, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = batchsize, shuffle = False)

#the classes in the CIFAR10 dataset
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#mporting and setting up the VGG16 model for Transfer Learning and altering the number of output features
from torchvision.models import vgg16, VGG16_Weights    #importing the VGG16 model and its corresponding weights 
weights = VGG16_Weights.DEFAULT                        #setting our weights variable to the VGG16 weights
model = vgg16(weights=weights)                         #transfering the trained VGG16 (with the VGG16_Weights) to use as our model 
input_lastLayer = model.classifier[6].in_features      
model.classifier[6] = nn.Linear(input_lastLayer,10)    #Altering the VGG16 model to classify into 10 outputs as is needed for our dataset
model = model.to(device)                               #loading the transfered and altered model 

#Training the Model in steps according to the batchsize of the training and testing dataset, over a for loop
for epoch in range(epochs):
   for i, (image , labels) in enumerate(train_dataloader):
          image = image.to(device)
          labels = labels.to(device)

          labels_hat = model(image)                                     #finds predicted class for an image
          corrects = (labels_hat.argmax(axis=1)==labels).sum().item()   #checks if the predicted class is equal to the actual class
          loss_value = loss_func(labels_hat, labels)                    #computes the loss
          loss_value.backward()                                         #finds the gradients wrt loss to find the parameters
          optimiser.step()                                              #updates the parameters of the model 
          optimiser.zero_grad()
          if (i+1) % 250 == 0:
            print(f'epoch {epoch+1}/{epochs}, step: {i+1}/{total_step}: loss = {loss_value:.5f}, acc = {100*(corrects/labels.size(0)):.2f}%')

#Model Evaluation to determine the accuracy 
accurate = 0
total = 0

with torch.no_grad():
  for i, (testimages,testlabels) in enumerate(test_dataloader):  #unpacks the data into input data and labels 
    testimages = testimages.to(device)   
    testlabels = testlabels.to(device)
    predicted = model(testimages)                                # runs the test images in the model for the prediction 
    labelspredicted = predicted.argmax(axis=1)                   # predicts the class for the test images
    accurate += (labelspredicted==testlabels).sum().item()       #calculates and updates the number of accurate predictions
    total += testlabels.size(0)                                  #calculates total no of predictions  

print(f'Accuracy of the VGG16 CNN on the 10000 test images is : {100 * accurate / total} %')    #returns the accuracy of the VGG16 model 

