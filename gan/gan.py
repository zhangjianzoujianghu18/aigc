import pandas as pd 
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.v2 import PILToTensor,Compose
import torchvision
batch_size=1024
# 1 dataload
minist=torchvision.datasets.MNIST("../data",train=True,download=True,transform=Compose(
    [torchvision.transforms.Resize(28),torchvision.transforms.ToTensor()
     ]))
dataSet=DataLoader(minist,batch_size=batch_size,shuffle=True,drop_last=True)
print(len(minist))


# 2 model
class Discriminator(nn.Module):
    def __init__(self, input_channel) -> None:

        super().__init__()
        self.model=nn.Sequential(
        nn.Conv2d(input_channel,64,kernel_size=3,stride=1,padding="same"),
        nn.Conv2d(64,128,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.Conv2d(128,256,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.Conv2d(256,1024,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.Conv2d(1024,1024,kernel_size=2,stride=2)
        )
        self.line1=nn.Linear(1024,256)
        self.sig1=nn.ReLU()
        self.line2=nn.Linear(256,1)
        self.sig2=nn.Sigmoid()

    def forward(self,x):
        bs=x.shape[0]
        x=self.model(x)
        y=x.reshape(bs,-1)
        return  self.sig2(self.line2(self.sig1(self.line1(y))))


class Generator(nn.Module):
    def __init__(self,input_channel) -> None:
        super().__init__()
        self.model=nn.Sequential(
        nn.Conv2d(input_channel,64,kernel_size=3,stride=1,padding="same"),
        nn.Conv2d(64,128,kernel_size=3,stride=2),
        nn.Conv2d(128,256,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.Conv2d(256,1024,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(1024,256,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(256,128,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(128,64,kernel_size=5,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64,1,kernel_size=4,stride=1),
        )

    def forward(self,x):
        x=self.model(x)
        return x
device=torch.device("cuda")
# 3 train
gen=Generator(1).to(device)
disc=Discriminator(1).to(device)
g_optim=optim.Adam(gen.parameters(),lr=1e-4)
d_optim=optim.Adam(disc.parameters(),lr=1e-4)
loss_fn=nn.BCELoss()

num_epoch=500
i=0
for epoch in range(num_epoch):
    for data in dataSet:
        x=(data[0]/255).to(device)
        z=torch.randn(batch_size,28*28).reshape(batch_size,1,28,28).to(device)
        # z=torch.randn(batch_size,1,28,28).to(device)
        pred_z=gen(z)
        disc_g=disc(pred_z)

        g_optim.zero_grad()
        target_g=torch.ones(batch_size,1).to(device)
       
        g_loss=loss_fn(disc_g,target_g)
        g_loss.backward()
        g_optim.step()

        # Discriminator loss_fn
        z=torch.randn(batch_size,28*28).reshape(batch_size,1,28,28).to(device)
        d_optim.zero_grad()
        disc_x=disc(x)
        d_loss=0.5*(loss_fn(disc_x,target_g))+0.5*loss_fn(disc(gen(z).detach()),torch.zeros(batch_size,1).to(device))

        d_loss.backward()
        d_optim.step()
        i+=1
        if i%100==0:
            print(f"epoch: {epoch} i: {i} d_loss : {d_loss} g_loss :{g_loss}")
            for index,image in enumerate(pred_z[:10]*255):
                torchvision.utils.save_image(image,f"image_{index}.png")





