from torch.utils import data
from skimage import io
import pandas

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt


class SportLoader(data.Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        self.sport = pandas.read_csv(f"{mode}.csv")
        self.img_name = []
        self.label = []
        self.transform = transform
        
        self.img_name = self.sport['names'].to_list()
        self.label = self.sport['label'].to_list()
        
    def __len__(self):
        return len(self.img_name)
        
    def __getitem__(self, index):
        
        image_path = f"{self.mode}/{self.img_name[index]}"
        self.img = io.imread(image_path)#/255.0
#         print((self.img))
        self.target = self.label[index]
        
        if self.transform:
            self.img = self.transform(self.img)
        
        return self.img, self.target

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool11 = nn.MaxPool2d(2, 2)
        
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool16 = nn.MaxPool2d(2, 2)
        
        self.conv19 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool21 = nn.MaxPool2d(2, 2)
        
        self.ftn1 = nn.Flatten()
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self, input):
        output = F.relu((self.conv1(input)))
        output = F.relu(self.pool3(output))

        output = F.relu((self.conv4(output)))
        output = F.relu(self.pool6(output))

        output = F.relu((self.conv7(output)))
        output = F.relu(self.pool11(output))
    
        output = F.relu((self.conv12(output)))
        output = F.relu(self.pool16(output))
    
        output = F.relu((self.conv19(output)))
        output = F.relu(self.pool21(output))
        
        output = self.ftn1(output)
        output = self.fc3(output)
        
        return output
        



def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader):
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    bestModelAcc = 0.0
    bestModelepoch = 0

    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        model.train()
        for i, (train_x, train_y) in enumerate(train_loader):
            print("=", end='', flush=True)
            optimizer.zero_grad()
            train_x = Variable(train_x.float())#.permute(0, 3, 1, 2))
            train_x, train_y = train_x.to('cuda'), train_y.to('cuda')
            train_pred = model(train_x)
            train_loss = loss_func(train_pred, train_y)
            train_loss.backward()
            optimizer.step()
            
            predicted = torch.max(train_pred.data, 1)[1]
            total_train += len(train_y)
            correct_train += (predicted == train_y).float().sum()
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        training_loss.append(train_loss.data)
        
        correct_val = 0
        total_val = 0
        model.eval()
        torch.no_grad()
        for i, (val_x, val_y) in enumerate(val_loader):
            val_x = Variable(val_x.float())#.permute(0, 3, 1, 2))
            val_x, val_y = val_x.to('cuda'), val_y.to('cuda')
            val_pred = model(val_x)
            val_loss = loss_func(val_pred, val_y)
            predicted = torch.max(val_pred.data, 1)[1]
            total_val += len(val_y)
            correct_val += (predicted == val_y).float().sum()
        
        val_accuracy = 100 * correct_val / float(total_val)
        validation_accuracy.append(val_accuracy)
        
        #Save best val_acc model
        if val_accuracy > bestModelAcc:
            bestModelAcc = val_accuracy
            bestModelepoch = epoch+1
            torch.save(model.state_dict(), f"./models/best.pt")
        
        validation_loss.append(val_loss.data)
        print('\nTrain Epoch: {}/{} Traing_Loss: {:.3f} Traing_acc: {:.3f}% Val_Loss: {:.3f} Val_accuracy: {:.3f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    
    print(f"Best Acc: {bestModelAcc}\nEpoch: {bestModelepoch}")
    
    return training_loss, training_accuracy, validation_loss, validation_accuracy


#Load train&val data
mean = [0.5, 0.5, 0.5]
std = [0.1, 0.1, 0.1]
train_transforms = transforms.Compose([
    
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5, expand=False, center=(112, 112)),
    transforms.CenterCrop((210, 210)),
    transforms.Resize((128, 128), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = SportLoader("train", transform=train_transforms)
val_dataset = SportLoader("val", transform=train_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

#build&fit model
model = Network().to('cuda')
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.008, momentum=0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)
input_shape = (-1,3,128,128)
num_epochs = 150

training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader)

#Show training fig
for i in range(len(training_loss)):
    training_loss[i] = training_loss[i].tolist()
    training_accuracy[i] = training_accuracy[i].tolist()
    validation_loss[i] = validation_loss[i].tolist()
    validation_accuracy[i] = validation_accuracy[i].tolist()

plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
plt.title('Training & Validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
