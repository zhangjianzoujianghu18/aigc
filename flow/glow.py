# 1 import package
import torch
import  torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.utils import save_image
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from scipy import linalg as la
import numpy as np
import math
BATCH_SIZE=256
EPOCHS=200
image_size=28
channel=1
z_dim=128
n_flow=4
n_block=2
n_channel=1
n_sampls=5
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
def show_mnist_image(image_array, label):
    plt.imshow(image_array, cmap='Greys')
    plt.title(f"Label: {label}")
# 1 data loader 
dataset=datasets.MNIST("../data/",train=True,transform=transforms.Compose([
    transforms.Resize(28),transforms.ToTensor(),transforms.Normalize(0.5,0.5)
]))
mnist=DataLoader(dataset,shuffle=True,batch_size=BATCH_SIZE,drop_last=True)
for data in mnist:
    print(data[0].shape)
    show_mnist_image(data[0][0][0], data[1][0])
    break

    
    
logabs=lambda x : torch.log(torch.abs(x))
class Actnorm(nn.Module):
    def __init__(self,in_channel,logdet=True):
        super(Actnorm,self).__init__()
        self.bias=nn.Parameter(torch.zeros(1,in_channel,1,1))
        self.scale=nn.Parameter(torch.ones(1,in_channel,1,1))
        self.register_buffer("initialized",torch.tensor(0))
        self.logdet=logdet
    
    # 求出channel维度的均值和方差作为初始值
    def initialize(self,input):
        with torch.no_grad():
            flatten=input.permute(1,0,2,3).contiguous().view(input.shape[1],-1)
            mean=flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1,0,2,3)
            std=flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1,0,2,3)
            self.bias.data.copy_(-mean)
            self.scale.data.copy_(1/(std+1e-6)) # 防止数据下益
    def forward(self,input):
        bs,channel,height,width=input.shape
        if self.initialized.item()==0:
            self.initialize(input)
            self.initialized.fill_(1)
        
        y=(input+self.bias)*self.scale
        
        if self.logdet:
            logdet=height*width*torch.sum(logabs(self.scale))
            return y,logdet
        else :
            return y
        
    def reverse(self,input):
        return (input-self.bias)/self.scale
class InvConv2d(nn.Module):
    def __init__(self,in_channel):
        super(InvConv2d,self).__init__()
        self.weight=nn.Parameter(torch.randn(in_channel,in_channel,1,1))
    def forward(self,input):
        bs,channel,height,width=input.shape
        out=nn.functional.conv2d(input,self.weight)
        logdet=height*width*torch.slogdet(self.weight.double())[1].float()
        return out ,logdet
    def reverse(self,input):
        out=nn.functional.conv2d(input,self.weight.inverse(),unsequeeze().unsequeeze(3))
class InvConv2d_lu(nn.Module):
    def __init__(self,in_channel):
        super(InvConv2d_lu,self).__init__()
        weight=np.random.randn(in_channel,in_channel)
        q,r=la.qr(weight)
        p,l,u=la.lu(q.astype(np.float32))
        s=np.diag(u)
        u=np.triu(u,1)
        u_mask=np.triu(np.ones_like(u),1)
        l_mask=u_mask.T
        
        p=torch.from_numpy(p)
        l=torch.from_numpy(l)
        s=torch.from_numpy(s)
        u=torch.from_numpy(u)
        u_mask=torch.from_numpy(u_mask)
        l_mask=torch.from_numpy(l_mask)
        
        self.register_buffer("p",p)
        self.register_buffer("u_mask",u_mask)
        self.register_buffer("l_mask",l_mask)
        self.register_buffer("s_sign",torch.sign(s))
        self.register_buffer("l_eye",torch.eye(l_mask.shape[0]))
        self.l=nn.Parameter(l)
        self.s=nn.Parameter(logabs(s))
        self.u=nn.Parameter(u)
        
    def calc_weight(self):
        
        weight=self.p@(self.l*self.l_mask+self.l_eye)@ \
        (self.u*self.u_mask+torch.diag(self.s_sign*torch.exp(self.s)))
        return weight.unsqueeze(2).unsqueeze(3)
    
    def forward(self,input):
        bs,channel,height,width=input.shape
        weight=self.calc_weight()
        out=nn.functional.conv2d(input,weight)
        logdet=height*width*torch.sum(self.s)
        return out ,logdet
    
    def reverse(self,input):
        weight=self.calc_weight()
        out=nn.functional.conv2d(input,weight.inverse(),unsequeeze().unsequeeze(3)) 
class ZeroConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,padding=1):
        super().__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,3,padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale=nn.Parameter(torch.zeros(1,out_channel,1,1))
    def forward(self,input):
        out=nn.functional.pad(input,[1,1,1,1],value=1)
        out=self.conv(out)
        out=out*torch.exp(self.scale*3)
        return out
class AffineCoupling(nn.Module):
    def __init__(self,in_channel,filter_size=256,affine=True):
        super(AffineCoupling,self).__init__()
        self.affine=affine
        self.net=nn.Sequential(
        nn.Conv2d(in_channel//2,filter_size,3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(filter_size,filter_size,1),
        nn.ReLU(inplace=True),
        ZeroConv2d(filter_size,in_channel if self.affine else in_channel//2)
        )
        
        self.net[0].weight.data.normal_(0,0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0,0.05)
        self.net[2].bias.data.zero_()     
    def forward(self,input):
        in_a,in_b=input.chunk(2,1)
        if self.affine:
            logs,t=self.net(in_b).chunk(2,1)
            y_a=torch.exp(logs)*in_a+t
            y_b=in_b
            logdet=torch.sum(logabs(torch.exp(logs)))
        else:
            y_b=in_b
            y_a=in_a+self.net(in_b)
            logdet=None
#         return y_a,y_b
        return torch.cat([y_a,y_b],1),logdet
    def reverse(self,output):
        y_a,y_b=output.chunk(2,1)
        if self.affine:
            in_b=y_b
            logs,t=self.net(in_b)
            in_a=(y_a-t)/torch.exp(logs)
        else :
            in_b=y_b
            in_a=self.net(in_b)-y_a
        return torch.cat([y_a,y_b],1)    
        
class Flow(nn.Module):
    def __init__(self,in_channel,affine=True,conv_lu=True):
        super(Flow,self).__init__()
        self.actorm=Actnorm(in_channel)
        if conv_lu:
            self.invConv2d=InvConv2d_lu(in_channel)
        else:
            self.invConv2d=InvConv2d(in_channel)
        self.coupling=AffineCoupling(in_channel,256)
    def forward(self,input):
        out,det1=self.actorm(input)
        out,det2=self.invConv2d(out)
        out,det3=self.coupling(out)
        logdet=det1+det2
        if det3 is not None:
            logdet+=det3
        return out,logdet
    
    def reverse(self,output):
        input_=self.coupling.reverse(output)
        input_=self.invConv2d.reverse(input_)
        input_=self.actorm(input_)
        return input_
    
def gaussian_log_p(x,mean,log_std):
    return -0.5*math.log(2* math.pi)-log_std-0.5*(x-mean)**2/torch.exp(2*log_std)


def gaussian_sample(eps,mean,log_std):
    return mean+torch.exp(log_std)*eps
class Block(nn.Module):
    def __init__(self,in_channel,n_flow,split=True,affine=True,conv_lu=True):
        super().__init__()
        squeeze_dim=in_channel*4
        self.flows=nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim,affine=affine,conv_lu=conv_lu))
        self.split=split
        
        if split:
            self.prior=ZeroConv2d(in_channel*2,in_channel*4)
        else: 
            self.prior=ZeroConv2d(in_channel*4,in_channel*8)
        
    def forward(self,input):
        bs,n_channel,height,width=input.shape
        squeezed=input.view(bs,n_channel,height//2,2,width//2,2)
        squeezed=squeezed.permute(0,1,3,5,2,4)
        out=squeezed.contiguous().view(bs,n_channel*4,height//2,width//2)
        logdet=0

        for flow in self.flows:
            out,det=flow(out)
            logdet+=det
        
        if self.split:
            out,z_new=out.chunk(2,1)
            mean,logstd=self.prior(out).chunk(2,1)
            log_p=gaussian_log_p(out,mean,logstd)
            log_p=log_p.view(bs,-1).sum(1)
            
        else:
            zero=torch.zeros_like(out)
            mean,logstd=self.prior(zero).chunk(2,1)
            log_p=gaussian_log_p(out,mean,logstd)
            log_p=log_p.view(bs,-1).sum(1)
            z_new=out
        return out,logdet,log_p,z_new
    
    def reverse(self,output,eps=None,reconstruct=False):
        input=output
        
        if reconstruct:
            if self.split:
                input=torch.cat([output,eps],1)
            else:
                input=eps
        else:
            if self.split:
                mean,logstd=self.prior(input).chunk(2,1)
                z=gaussian_sample(eps,mean,logstd)
                input=torch.cat([output,z],1)
                
            else:
                zero=torch.zeros_like(input)
                mean,log_sd=self.prior(zero).chunk(2,1)
                z=gaussian_sample(eps,mean,logstd)
                input=z
        for flow in self.flows[::-1]:
            input=flow.reverse(input)
            
        bs,n_channel,height,width=input.shape
        unsqueezed=input.view(bs,n_channel//4,2,2,height,width)
        unsqueezed=unsqueezed.permute(0,1,4,2,5,3)
        unsqueezed=unsqueezed.contiguous().view(bs,n_channel//4,height*2,width*2)
        
        return unsqueezed
            
        
class Glow(nn.Module):
    def __init__(self,in_channel,n_flow,n_block,affine=True,conv_lu=True):
        super().__init__()
        self.blocks=nn.ModuleList()
        n_channel=in_channel
        for i in range(n_block-1):
            self.blocks.append(Block(n_channel,n_flow,affine,conv_lu))
            n_channel*=2
        self.blocks.append(Block(n_channel,n_flow,split=False,affine=affine))
        
    def forward(self,input):
        log_p_sum=0
        logdet=0
        out=input
        z_outs=[]
        for block in self.blocks:
            out,det,log_p,z_new=block(out)
            z_outs.append(z_new)
            logdet=logdet+det
            
            if log_p is not None:
                log_p_sum=log_p_sum+log_p
        return log_p_sum,logdet,z_outs
    
    def reverse(self,z_list,reconstruct=False):
        for i,block in enumerate(self.blocks[::-1]):
            if i ==0:
                input=block.reverse(z_list[-1],z_list[-1],reconstruct=reconstruct)
                
            else:
                input=block.reverse(input,z_list[-(i+1)],reconstruct=reconstruct)
        return input
    
    
glow=Glow(n_channel,n_flow,n_block)
glow.to(device)
optimizer=torch.optim.Adam(glow.parameters(),1e-4)

def calc_loss(logp,logdet,imagesize):
    loss=logdet+logp
    return -loss.mean(),logp.mean(),logdet.mean()
def calc_z_shapes(n_channel,input_size,n_flow,n_block):
    z_shapes=[]
    for i in range(n_block-1):
        input_size//=2
        n_channel*=2
        z_shapes.append((n_channel,input_size,input_size))
    input_size//=2
    z_shapes.append((n_channel*4,input_size,input_size))
    return z_shapes
z_sample=[]
z_shapes=calc_z_shapes(1,image_size,n_flow,n_block)
for z in z_shapes:
    z_new=torch.randn(n_sampls,*z)
    z_sample.append(z_new.to(device))
index=0
EPOCHS=100
for epoch in range(EPOCHS):
    for data,label in mnist:
        x=data.to(device)
       
        if index==0:
            with torch.no_grad():
                log_p,logdet,_=glow(x)
                index+=1
                continue
        else:
            log_p,logdet,_=glow(x)
            index+=1
        logdet=logdet.mean()
        
        loss,log_p,log_det=calc_loss(log_p,logdet,image_size)
        glow.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch} index : {index}, loss {loss} , logdet :{log_det}")
        
        if index%10==0:
            print(f"epoch {epoch} index : {index}, loss {loss} , logdet :{log_det}")
            with torch.no_grad():
                save_image(
                glow.reverse(z_samples).cpu().data,
                    f"sample{ str(index+1).zfill(6)}.png",
                    normmalize=True,
                    nrow=5,
                    range=(-0.5,0.5)
                )
        