import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim

#transforms processed data into tensor format
train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))
#maintains seperate and balanced datasets for training and testing
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()#define the structure of the hidden layers
        #feed forward network -> data passes in one direction
        self.fc1 = nn.Linear(784, 64) #fully connected layer 1 9784 --> 28*28 images, output -->hidden layer
        self.fc2 = nn.Linear(64, 64) #input --> takes in 64 connections from fc1
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #(input, output --> only 10 layers )
    def forward(self, x):
        #activation functions run on output, not input
        x = F.relu(self.fc1(x)) #F.relu() rectify linear, activation function --> restricts output
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #final output distribution.
        return F.log_softmax(x,dim=1) #Multi-class -> softmax, dim=1 -> we are distributing across actual tensor output


net = Net()
optimizer = optim.Adam(net.parameters(), lr=.001) #.parameters corresponds to everything that is adjust in model, lr=learning rate
EPOCH = 3 #number of iterations through datasets

for epoch in range(EPOCH):
    for data in trainset:
        #data is a batch of featuresets and labels
        x,y = data #x input, y expected output
        net.zero_grad() #start at zero with correction gradients
        output = net(x.view(-1,28*28)) #network's guess
        loss = F.nll_loss(output, y)
        loss.backward() #backpropogate MAGIC
        optimizer.step() #adjusts weights of hidden layers
    #print(loss)

correct = 0
total = 0
with torch.no_grad(): #not calculating with gradients yet, line functions as net.train() ... net.eval()
    for data in trainset:
        x,y = data
        output = net(x.view(-1,784)) #784->28*28pixels
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct +=1
            total +=1
print('accuracy: ', round(correct/total,3))
#visually display accuracy data
plt.imshow(x[0].view(28,28))
plt.show()
