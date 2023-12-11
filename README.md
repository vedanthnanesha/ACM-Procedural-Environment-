# ACM-Procedural-Environment-

Model 1 = The main file where this can be found is CNNModel1.ipynb and the raw code is in the same .py file. 
In this model I have taken a simple approach and built a simple CNN with a few convolutional, pooling and fully connnected layers. While transforming and loading the dataset I have only used normalisation, no data augmentation techniques were used. 
This model achieved an accuracy of 59%. 

Model 2 = The main file where this can be found is CNNModel2.ipynb and the raw code is in the same .py file. 
In this model, I added upon my learnings from the first model and built an improved CNN by adding more convolutional layers, batch normalisation and dropout. I have also used data augmentation rotations and flips while transforming the dataset, thus effectively making the dataset larger and more versatile. As slightly different images are encountered in every loop, the model gets new data to work on which prevents overfitting. 
This model achieved an accuracy of 71%. 

Model 3 = The main file where this can be found is CNNVGG16Model3.ipynb and the raw code is in the same .py file. 
In this model I have used the principles of transfer learning to use a pretrained VGG-16 model on the CIFAR10 dataset. The VGG-16 model is a deep CNN with 13 convolutional layers, 5 maxpooling layers and 3 fully connected layers, thus allowing for a more thorough classification process. I imported this model with its weights from the torchvision.models module, and altered some of its characteristics for the use on the CIFAR10 dataset. For example the VGG16 model normally has 1000 output features but the CIFAR10 dataset only has 10 output classes, so we find these 10 outputs by the nn.linear function. Also, we resize all the images of CIFAR10 to 224x224 as this is the size of the input layer of the VGG16 model. Data augmentations were also used in the initial transforming of the images to tensors. The GPU runtime of this model was around 80 mins with 6 epochs. 
This model achieved an accuracy of 





I have added comments in the .py files explaining the steps I have taken for each function implemented.  
