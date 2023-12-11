# ACM-Procedural-Environment-

Model 1 = The main file where this can be found is CNNModel1.ipynb and the raw code is in the same .py file. 
In this model I have taken a simple approach and built a simple CNN with a few convolutional, pooling and fully connnected layers. While transforming and loading the dataset I have only used normalisation, no data augmentation techniques were used. 
This model achieved an accuracy of 59%. 


Model 2 = The main file where this can be found is CNNModel2.ipynb and the raw code is in the same .py file. 
In this model, I added upon my learnings from the first model and built an improved CNN by adding more convolutional layers, batch normalisation and dropout. I have also used data augmentation rotations and flips while transforming the dataset, thus effectively making the dataset larger and more versatile. As slightly different images are encountered in every loop, the model gets new data to work on which prevents overfitting. 
This model achieved an accuracy of 71%. 


Model 3 = The main file where this can be found is CNN_VGG16_Model3.ipynb and the raw code is in the same .py file. 
In this model I have used the principles of transfer learning to use a pretrained VGG-16 model on the CIFAR10 dataset. The VGG-16 model is a deep CNN with 13 convolutional layers, 5 maxpooling layers and 3 fully connected layers, thus allowing for a more thorough classification process. 
I imported this model with its weights from the torchvision.models module, and finetuned some of its characteristics for the use on the CIFAR10 dataset. For example the VGG16 model normally has 1000 output features but the CIFAR10 dataset only has 10 output classes, so we find these 10 outputs by the nn.linear function. Also, we resize all the images of CIFAR10 to 224x224 as this is the size of the input layer of the VGG16 model. Data augmentations were also used in the initial transforming of the images to tensors. The GPU runtime of this model was around 80 mins with 6 epochs. 
This model achieved an accuracy of 93.48%. 


Model 4 = The main file where this can be found is CNN_Resnet50_Model4.ipynb and the raw code is in the same .py file. 
In this model I have used tranfer learning to use the Resnet50 model with its weights on the CIFAR10 dataset. I chose the Resnet50 model over types such as Resnet101 and Resnet152 due to its ease of computational load and performance compared to the other two. The Resnet50 model is 50 layer deep residual neural network that stacks blocks together to form a network, yielding a high classification accuracy. 
I imported the pretrained model from the torchvision.models module and changed the number of output features in its fully connected layer from 1000 to 10, as needed for the CIFAR10 classification. Once again, I resized the images while transforming them to tensors as the input taken at the resnet50 input layer is 224x224x3. I also used the random horizontal flip, rotation and additional color jittering(adjusting contrast, brightness and exposure) data augmentations to artficilly introduce a larger variety of data to the model. I also explored a slightly varied normalisation using the mean and standard deviations of Red, Green and Blue to help the model converge faster during training. Using the similar general process used in the previous models along with these small variations in the Resnet50 model optimised the performance even further. The gpu runtime was around 45 mins, less than the time required for VGG16 due to Resnet50s reduced space requirement owing to global pooling rather than the use of many fully connected layers. 
This Model achieved an accuracy of 95.11%. 



We have seen that the use of transfer learning (along with data augmentations) improves the accuracy of the image classification manifold. If it is possible to finetune the imported models to act on a new dataset, than it is often preferable to use these techniques rather than building a model from scratch. 






I have added comments in the .py files explaining the steps I have taken for each function implemented.  
