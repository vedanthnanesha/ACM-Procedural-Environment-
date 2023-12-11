#importing the required libraries and tools 
import torch
import torchvision
import torchvision.transforms as transf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#setting up the basic parameters and checking if the program is run on gpu or cpu
epochs = 5
batchsize = 40
learnrate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #checking the system available for use
print(device)

#Transforms for converting the dataset image values to tensors with additional Data Augmentation and Resizing
rans_train = transf.Compose([
    transf.Resize(size=(224, 224)),         #this is the size of the input layer of the resnet50 model
    transf.RandomHorizontalFlip(),          #randomly flips an image
    transf.RandomRotation(10),              #randomly rotates an image 
    transf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),      #changes brightness, contract and saturation levels of the images
    transf.ToTensor(),                      #transforms to a tensor
    transf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])    #these are the values of the mean and standard deviation of RGB
trans_test = transf.Compose([
    transf.Resize(size=(224, 224)),
    transf.ToTensor(),
    transf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])   

#Downloading and Loading up the Training and Testing Data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download= True, transform = trans_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download= True, transform = trans_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = batchsize, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = batchsize, shuffle = False)

#The classes in the CIFAR10 Dataset
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#Importing the pretrained Resnet50 model with its weights and finetuning the model for CIFAR10 dataset (Transfer Learning)
from torchvision.models import resnet50, ResNet50_Weights     #importing the model and its weights
weights = ResNet50_Weights.DEFAULT                            #setting our weights to the pretrained weights
model = resnet50(weights=weights)                             #transfering the trained resnet50 to use as our model 
print(model.fc)                                               #checking the initial fully connected layer 
inputlastlayer = model.fc.in_features
model.fc = nn.Linear(inputlastlayer, 10)                      #changing the output layer to have 10 outputs for the CIFAR10 Dataset
model = model.to(device)   
print(model.fc)                                               #printing the altered fully connected layer 
print(model)                                                  #checking the characteristics of the model 

#Formulating the Loss Function and Optimiser to update the parameters
loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = learnrate, momentum = 0.9,weight_decay=5e-4) # weight decay helps prevent overfitting   
total_step = len(train_dataloader)  

#Training the Resnet50 Model over multiple training loops
for epoch in range(epochs):
   for i, (image , labels) in enumerate(train_dataloader):
          image = image.to(device)               
          labels = labels.to(device)

          labels_hat = model(image)                                      #finds predicted class for an image
          corrects = (labels_hat.argmax(axis=1)==labels).sum().item()    #checks if predicted class is equal to the actual class
          loss_value = loss_func(labels_hat, labels)                     #computes the loss
          loss_value.backward()                                          #finds the gradients by back propogation 
          optimiser.step()                                               #updates the model parameters for more accuracy 
          optimiser.zero_grad()                                          #clears the gradient for the next loop
          if (i+1) % 250 == 0:
            print(f'epoch {epoch+1}/{epochs}, step: {i+1}/{total_step}: loss = {loss_value:.5f}, acc = {100*(corrects/labels.size(0)):.2f}%')


#Evaluation of the Implemented Resnet50 Model
accurate = 0
total = 0

with torch.no_grad():
  for i, (testimages,testlabels) in enumerate(test_dataloader):
    testimages = testimages.to(device)
    testlabels = testlabels.to(device)
    predicted = model(testimages)                     
    labelspredicted = predicted.argmax(axis=1)         #finds the predicted value of each image
    accurate += (labelspredicted==testlabels).sum().item()   #checks if predicted value = true label 
    total += testlabels.size(0)                        #computes total test images

print(f'Accuracy of the Resnet50 CNN on the 10000 test images is : {100 * accurate / total} %')   #outputs overall accuracy which is 95.11%

import matplotlib.pyplot as plt
import numpy as np

#Visualising some predictions based on the trained model 
def visualize_predictions(model, images, labels, classes):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        predicted = model(images)
        predicted_labels = predicted.argmax(dim=1).cpu().numpy()    #converting to a numpy array

    images = images.cpu().numpy() * 0.2 + 0.5  #will denormalize images
    images = np.transpose(images, (0, 2, 3, 1))  #(batch, channels, height, width) to (batch, height, width, channels)

    numimages = len(images)                 #will show the predictions of one batch of images
    rows = int(np.ceil(numimages / 4))      
    cols = min(4, numimages)

    plt.figure(figsize=(15,15))             #using matplotlib.pyplot for the visualisation
    for i in range(numimages):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.title(f'Predicted: {classes[predicted_labels[i]]} \nActual: {classes[labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

data_iterator = iter(test_dataloader)          #iterates through one batch of test data
images, labels = next(data_iterator)           #unpacks into labels and images 
visualize_predictions(model, images, labels, classes)   #we call the function to view the predictions



