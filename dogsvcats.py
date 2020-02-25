import os
import cv2
import numpy as np
from tqdm import tqdm #progress bar
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #optimizer

REBUILD_DATA = False #True when data must be rebuilt for use

class dogsvscats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS:0, DOGS:1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS: #iterating through the directories
            print(label)
            for f in tqdm(os.listdir(label)): #f->file name
                try:
                    path = os.path.join(label,f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #color does not add demensions but channels in a convonet
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount +=1
                    elif label == self.DOGS:
                        self.dogcount +=1
                except Exception as e: #if some image data proves to be bad
                    #print(str(e))
                    pass
            np.random.shuffle(self.training_data) #shuffles in place
            np.save("training_data.npy", self.training_data)
            print("cats:",self.catcount)
            print("dogs:",self.dogcount)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._to_linear = None #linear value to flatten 2d data
        self.conv1 = nn.Conv2d(1,32,5)#2d conv-> (inputs,outpurss,kernalsize)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5) #need to flatten data to linear layers
        #we need to find how ther 3 conv layers affect the sizing of the data
        x = torch.randn(50,50).view(-1,1,50,50) #will create fake data to see how it scales after layers
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2) #final output // prediction

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #(2,2) -> pooling size
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2] #flattened value
        return x

    def forward(self, x):
        x = self.convs(x) #pass through all conv layers, comes out 2d (not flat)
        x = x.view(-1, self._to_linear) # flatten data according to the _to_linear value found earlier
        x = F.relu(self.fc1(x)) #begin passing through linear layers
        x = self.fc2(x)
        #return x #activation functions are not necessary with this sort of problem, vector mulitiplication does not need normalization
        return F.softmax(x,dim=1) #softmax -> activation layer (normalizing function)
        #diminish = 1

if REBUILD_DATA:
    dogsvscats = dogsvscats()
    dogsvscats.make_training_data()

net = Net()
optimizer = optim.Adam(net.parameters(), lr=.001)
loss_function = nn.MSELoss()
x = torch.Tensor([i[0] for i in training_data]).view(-1,50,50) #data currently in 0-255 because of colored pixels
x = x/255.0 #data now 0-1
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = .1 #percentage of our data we will be testing against (validating)
val_size = int(len(x)*VAL_PCT)

train_x = x[:-val_size]
train_y = y[:-val_size]

test_x = x[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 100 #lower if memory error
EPOCHS = 1 #num of iterations through data

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
        #print(i, i+BATCH_SIZE) #slices through training data
        batch_x = train_x[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]

        #optimizer.zero_grad() #can use optimizer.zero_grad, however newt.zero_grad()
        net.zero_grad() #tends to be safer, if one optimizer for network it does not matter
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y) #determine loss function from expected outputs
        loss.backward() #train model backwards using current loss function
        optimizer.step() #step through model optimizing bias and variance
print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_x))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_x[i].view(-1,1,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct+=1
        total +=1

print("Accruacy:", round(correct/total,3))  #determines effeciency of model
