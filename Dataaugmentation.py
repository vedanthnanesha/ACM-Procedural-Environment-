#the transformation of the training data is altered by using data augmentations, an example is given below 
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),       # Randomly rotate images by up to 10 degrees
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Random affine transformations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust brightness, contrast, and saturation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#the testing data transformation remains the same
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Loading the CIFAR-10 Dataset with the new data augmentations
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #note the transformation used with augmentations
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

#The rest of the code remains the same, we can alter the specific parameters of the data augmentation based on experimentation on the dataset taken
