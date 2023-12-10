#importing some libraries and tools
import torch
import torchvision
import torchvision.transforms as transf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#Transforming the testing data with the use of Data Augmentations to help artificially improve the dataset
trans_train = transf.Compose([
    transf.RandomHorizontalFlip(),  #flips the image horizontally
    transf.RandomRotation(10),      #randomly rotates the image by upto 10 degrees
    transf.ToTensor(), 
    transf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
trans_test = transf.Compose([
    transf.ToTensor(),
    transf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

#Obtaining the required data and loading it using the specific transforms 
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download= True, transform = trans_train) #notice the transform with the augmentation
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download= True, transform = trans_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 4, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 4, shuffle = False)

#The classes used in the dataset
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#Implementing our CNN Class with varied aspects such as more depth, batch normalisation and dropout
class convNet(nn.Module):
    def __init__(self):
        super(convNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)  
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)  #an increased number of convolutional layers (5) are used
        self.b1=nn.BatchNorm2d(16)                              #Batch Normalization layers to normalize the input of each layer, which helps with generalisation
        self.b2=nn.BatchNorm2d(64)
        self.b3=nn.BatchNorm2d(256)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)         #reduces the size by selecting max in a region 
        self.dropout=nn.Dropout(0.1)                           #dropout prevents overfitting by setting a fraction of inputs to 0
        self.fc1=nn.Linear(256,128)
        self.fc2=nn.Linear(128,64)
        self.out=nn.Linear(64,10)                              #3 fully connected layers


    def forward(self,x):
        x=self.pool(F.relu(self.b1(self.conv1(x))))        
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.b2(self.conv3(x))))
        x=self.pool(F.relu(self.conv4(x)))
        x=self.pool(F.relu(self.b3(self.conv5(x))))           #activation functions
        x=x.view(-1,256)
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.dropout(F.relu(self.fc2(x)))
        x=self.out(x)   
        return x

#Loss function and optimiser are imported
learnrate = 0.001
model = convNet()
loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = learnrate, momentum = 0.9)

#Training the Model over a high number of loops
epochs = 8
for epoch in range(epochs):

   runningloss = 0.0
   for i,batch in enumerate(train_dataloader, 0):

      inputs, labels = batch    #unpacks the training data
      optimiser.zero_grad()     #clears gradient to 0

      outputs = model(inputs)
      loss = loss_func(outputs, labels)    #computes the loss
      loss.backward()                      #computes the gradients wrt loss
      optimiser.step()                     #updates model parameters after computing the required gradients
      runningloss += loss.item()
      if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {runningloss / 2000:.3f}')
            runningloss = 0.0

print('Training is done')

#Evaluation of the Model
accurate = 0
total = 0

with torch.no_grad():
  for batch in test_dataloader:
    input, labels = batch
    outputs = model(input)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    accurate += (predicted == labels).sum().item()

print(f'Accuracy of the CNN on the 10000 test images is : {100 * accurate // total} %')

#Plotting a Confusion Matrix to visualise the accuracy of prediction for particular classes

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                         #importing additional libraries

def predictImage(img, model):                 #function to predict the class of an image

    x = img.unsqueeze(0)
    y = model(x)
    _, pred = torch.max(y, dim=1)

    return pred[0].item() 

predict = np.empty((0, len(test_data)), np.int32)      #initialising an empty array for predictions
actual = np.empty((0, len(test_data)), np.int32)     #initialising an empty array for actual values

with torch.no_grad():
    for i in range(0, len(test_data)):
        testimage, testlabel = test_data[i]
        predictedValue = predictImage(testimage, model)   #predicts each images class

        predict = np.append(predict, predictedValue)      #appends prediction to the array
        actual = np.append(actual, testlabel)             #appends corresponding actual value to the array
confusionMatrix = confusion_matrix(actual, predict)       #creates a confusion matrix relating the above 2 

confusionMatrixDf = pd.DataFrame(confusionMatrix, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (10,10))
sns.heatmap(confusionMatrixDf, annot=True, cmap='Blues', fmt='g')




