from torch.utils import data
from skimage import io
import os

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import torch

from torch.autograd import Variable
import pandas


class SportLoader(data.Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        self.img_name = sorted(os.listdir(f"./{mode}"))
        self.transform = transform
        
        
    def __len__(self):
        return len(self.img_name)
        
    def __getitem__(self, index):
        
        image_path = f"./{self.mode}/{self.img_name[index]}"
        self.img = io.imread(image_path)
        
        if self.transform:
            self.img = self.transform(self.img)
        
        return self.img, self.img_name[index]

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

#Load test data        
mean = [0.5, 0.5, 0.5]
std = [0.1, 0.1, 0.1]
train_transforms = transforms.Compose([
    
    transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(5, expand=False, center=(112, 112)),
#     transforms.CenterCrop((210, 210)),
    transforms.Resize((128, 128), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_dataset = SportLoader("test", transform=train_transforms)
test_loader = DataLoader(test_dataset, shuffle=False)

#Load model
path = "./HW1_311551144.pt"
model = Network()
model.load_state_dict(torch.load(path))
model.eval()
torch.no_grad()

#save predict result
df = pandas.DataFrame(columns=('names', 'label'))
cnt = 0
for test_x, file in test_loader:
    test_x = Variable(test_x.float())
    test_pred = model(test_x)
    predicted = torch.max(test_pred.data, 1)[1]
#     print(file[0], predicted.tolist()[0])
    df.loc[cnt] = [file[0]] + predicted.tolist()
#     df.append({'names':file[0], 'label':predicted.tolist()[0]}, ignore_index=True)
    cnt+=1
df.to_csv('test.csv', index=False)

