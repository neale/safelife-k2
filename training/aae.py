import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, z, c=10):
        super(Encoder, self).__init__()
        self.z = z
        self.conv1 = nn.Conv2d(c, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.linear = nn.Linear(64*3*3, z)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
    
    def forward(self, input):
        #print ('encoder in', input.shape)
        x = F.selu(self.bn1(self.conv1(input)))
        x = F.selu(self.bn2(self.conv2(x)))
        x = F.selu(self.bn3(self.conv3(x)))
        x = x.view(-1, 3*3*64)
        x = self.linear(x)
        #print ('encoder out', x.shape)
        return x.view(-1, self.z)


class Decoder(nn.Module):
    def __init__(self, z, c=10):
        super(Decoder, self).__init__()
        self.z = z
        self.c = c
        self.linear1 = nn.Linear(self.z, 3*3*64)
        self.conv1 = nn.ConvTranspose2d(64, 64, 3, stride=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, c, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #print ('decoder in', input.shape)
        x = self.linear1(input)
        x = x.view(-1, 64, 3, 3)
        x = F.selu(self.bn1(x))
        x = F.selu(self.bn1(self.conv1(x)))
        x = F.selu(self.bn2(self.conv2(x)))
        x = self.sigmoid(self.conv3(x))
        #print ('decoder out', x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, z, c=10):
        super(Discriminator, self).__init__()
        self.z = z
        self.c = c
        self.conv1 = nn.Conv2d(c, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.linear = nn.Linear(64*3*3, 1)

    def forward(self, x):
        #print ('disc in', x.shape)      
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = x.view(-1, 3*3*64)
        x = self.linear(x)
        #print ('disc out', x.shape)     
        return x


class AAE(nn.Module):
    def __init__(self, device, z_dim=32, lr_ae=1e-3, lr_d=1e-4, replay_size=1e5,
                 batch_size=128, training_epochs=100, gp=True, gp_scale=10.):
        super(AAE, self).__init__()
        self.lr_d = lr_d
        self.lr_ae = lr_ae
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.gp = gp
        self.gp_scale = gp_scale
        self.epochs = training_epochs
        self.z_dim = z_dim
        self.device = device

        self.encoder = Encoder(c=10, z=z_dim).to(self.device)
        self.decoder = Decoder(c=10, z=z_dim).to(self.device)
        self.discriminator = Discriminator(c=10, z=z_dim).cuda()

        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(),
                                              lr=lr_ae, weight_decay=1e-4)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(),
                                              lr=lr_ae, weight_decay=1e-4)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=lr_d, weight_decay=1e-4)

    def set_data(self, frames: torch.Tensor):
        self.frames = frames
        dataset = torch.utils.data.TensorDataset(frames)
        loader = torch.utils.data.DataLoader(dataset, self.batch_size, drop_last=True, shuffle=True)
        self.dataloader = loader

    def grad_penalty(self, real, fake):
        alpha = torch.randn(self.batch_size, 1).cuda()
        alpha = alpha.expand(self.batch_size, real.nelement()//self.batch_size)
        alpha = alpha.contiguous().view(self.batch_size, 10, 25, 25)
        interpolates = alpha * real + ((1 - alpha) * fake).cuda()
        disc_interpolates = self.discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_scale
        return gradient_penalty
    
    def generate_ae_image(self, data, save_path=None):
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        if data.dim() == 3:
            data = data.unsqueeze(0)
        if data.shape[1] == data.shape[2]:
            data = data.transpose(1, 3)
        data = data.float().cuda()
        self.encoder.eval()
        self.decoder.eval()
        qz = self.encoder(data)
        samples = self.decoder(qz)
        if save_path is not None:
            save_image(samples, save_path+'.png')#, normalize=True)
            save_image(data, save_path+'.png')#, normalize=True)
        return samples
    
    def autoencoder_loss(self, data):
        qz = self.encoder(data)
        pz = self.decoder(qz)
        ae_loss = F.mse_loss(pz, data)
        return ae_loss
    
    def decoder_loss(self):
        z = torch.randn(self.batch_size, self.z_dim).cuda()
        z.requires_grad_(True)
        fake = self.decoder(z)
        cls = self.discriminator(fake).mean()
        return cls

    def train(self):
        steps = 0
        for epoch in range(self.epochs):
            for idx, data in enumerate(self.dataloader):
                data = torch.stack(data).squeeze(0)
                data = data.transpose(1, 3).float()
                data = data.cuda()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                ae_loss = self.autoencoder_loss(data)
                ae_loss.backward()
                self.optim_encoder.step()
                self.optim_decoder.step()

                """ Update D network """
                for p in self.discriminator.parameters():  
                    p.requires_grad = True 
                
                self.decoder.zero_grad()
                self.discriminator.zero_grad()
                d_real = self.discriminator(data).mean()
                d_real.backward(-torch.ones_like(d_real).cuda(), retain_graph=True)
                # train with fake data
                z = torch.randn(self.batch_size, self.z_dim).cuda().requires_grad_(True)
                with torch.no_grad():
                    fake = self.decoder(z)
                fake.requires_grad_(True)
                d_fake = self.discriminator(fake).mean()
                d_fake.backward(torch.ones_like(d_fake).cuda(), retain_graph=True)
                # train with gradient penalty 
                if self.gp:
                    gp = self.grad_penalty(data, fake)
                    gp.backward()
                else:
                    gp = torch.zeros(1)
                d_cost = d_fake - d_real + gp
                w1_dist = d_real - d_fake
                self.optim_discriminator.step()
            
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                self.decoder.zero_grad()
                cls = self.decoder_loss()
                cls.backward(torch.ones_like(cls).cuda())
                decoder_cost = -cls
                self.optim_decoder.step()
                steps += 1

                if steps % 1000 == 0:
                    print('ITER: ', steps, 'd cost', d_cost.cpu().item())
                    print('ITER: ', steps,'g cost', decoder_cost.cpu().item())
                    print('ITER: ', steps,'gp', gp.cpu().item())
                    print('ITER: ', steps,'w1 distance', w1_dist.cpu().item())
                    print('ITER: ', steps,'ae cost', ae_loss.data.cpu().item())
     
