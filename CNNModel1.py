#import the required libraries and tools
import torch
import torchvision
import torchvision.transforms as transf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#Loading the CIFAR10 Dataset in the Training and Testing Data
train_data0 = torchvision.datasets.CIFAR10(root='./data', train=True, download= True)
test_data0 = torchvision.datasets.CIFAR10(root='./data', train=False, download= True)
train_data0, test_data0
display(train_data0[5][0]) #checking a random image of training data
display(test_data0[9][0]) #checking a random image of testing data
train_data0[0] #checking the size and format

#Transforming the Initial Training and Testing data into Tensors and Loading it
tensor_trans = transf.Compose([transf.ToTensor(), transf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download= True, transform = tensor_trans)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download= True, transform = tensor_trans) 
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 4, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 4, shuffle = False) #does not need to be shuffled as it tests for accuracy

#Dictionary of the Classes in the Dataset
classes = {0:'plane', 1: 'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

#Visualizing the Dataset
import matplotlib.pyplot as plt
def imshow(image):
    image = image / 2 + 0.5     
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
iterator = iter(test_dataloader)
images, labels = next(iterator)
imshow(torchvision.utils.make_grid(images))

#Creating our Convulational Neural Network Class
class CNN(nn.Module):
   
   def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)  #creates a 2d convolutional layer
        self.pool = nn.MaxPool2d(2,2)  #reduces the size by selecting the max element from a region
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lc1 = nn.Linear(16*5*5, 120)  #the linear layers are fully connected and connect each input to every output neuron
        self.lc2 = nn.Linear(120, 84)
        self.lc3 = nn.Linear(84,10)

   def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #activation function for non linear outputs
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.lc1(x))
        x = F.relu(self.lc2(x))
        x = self.lc3(x)
        return x

#Defining the Loss Function and Optimiser
learnrate = 0.001
model = CNN()
loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = learnrate, momentum = 0.9) #optimises using stochastic gradient descent

#Training the Model through Looping
epochs = 8
for epoch in range(epochs):  #loops over the training data
   
   runningloss = 0.0
   for i,batch in enumerate(train_dataloader, 0):

      inputs, labels = batch  #unpacks the train data into data and corresponding labels
      optimiser.zero_grad()  #clears the gradient accumalted to 0

      outputs = model(inputs)
      loss = loss_func(outputs, labels)
      loss.backward()   #computes gradients wrt loss
      optimiser.step()  #updates the model parameters
      runningloss += loss.item()
      if i % 2000 == 1999:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {runningloss / 2000:.3f}')
            runningloss = 0.0

print('Training is done')

#Evaluation of the Model
accurate = 0
total = 0

with torch.no_grad():  #gradient is not needed for this step
  for batch in test_dataloader:
    input, labels = batch  #unpacks the test data into data and corresponding labels
    outputs = model(input)  #passes input to the model to obtain predictions
    _, predicted = torch.max(outputs.data, 1)  #finds index with max value to get the predicted class
    total += labels.size(0)
    accurate += (predicted == labels).sum().item()

print(f'Accuracy of the CNN on the 10000 test images is : {100 * accurate // total} %')

