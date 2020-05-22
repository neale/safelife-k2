import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from bvae_model import BetaVAE_H, BetaVAE_B


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class Trainer(object):
    def __init__(self, z_dim=4, beta=4, gamma=1000, C_max=25, C_stop_iter=1e5, objective='B',
		 model='B', lr=1e-4, beta1=.9, beta2=.999, training_epochs=200, exp='./'):
        self.use_cuda = torch.cuda.is_available()
        self.training_epochs = training_epochs
        self.global_iter = 0

        self.z_dim = z_dim
        self.beta = beta
        self.gamma = gamma
        self.C_max = C_max
        self.C_stop_iter = C_stop_iter
        self.objective = objective
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.exp = exp
        self.nc = 3
	net = BetaVAE_B

        self.save_epoch = 10
        self.display_epoch = 5
        self.batch_size = 32

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))


        self.ckpt_dir = 'models/bvae/{}/'.format(exp)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.output_dir = 'images/bvae/{}/reconstructions/'.format(exp)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = torch.utils.data.TensorDataset(data)
        self.data_loader = torch.utils.data.DataLoader(dataset, self.batch_size, drop_last=True, shuffle=True)

    def train(self):
        self.net_mode(train=True)
        self.C_max = cuda(torch.FloatTensor([self.C_max]), self.use_cuda)
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for epoch in training_epochs:
                for x in self.data_loader:
                    self.global_iter += 1
                    pbar.update(1)

                    x = cuda(x, self.use_cuda)
                    x_recon, mu, logvar = self.net(x)
                    recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                    if self.objective == 'H':
                        beta_vae_loss = recon_loss + self.beta*total_kld
                    elif self.objective == 'B':
                        C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                        beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

                    self.optim.zero_grad()
                    beta_vae_loss.backward()
                    self.optim.step()

                    if epoch % self.display_epoch == 0:
                        pbar.write('[{}/{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                            epoch, self.training_epochs, recon_loss.data[0], total_kld.data[0],
                            mean_kld.data[0]))

                        var = logvar.exp().mean(0).data
                        var_str = ''
                        for j, var_j in enumerate(var):
                            var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                        pbar.write(var_str)

                        if self.objective == 'B':
                            pbar.write('C:{:.3f}'.format(C.data[0]))

                    if epoch % self.save_epoch == 0:
                        self.save_checkpoint(str(epoch))
                        pbar.write('Saved checkpoint(epoch:{})'.format(epoch))
                        x_recon = x_recon[0].view(-1, 1, 90, 90).cpu()*255.
                        save_image(x_recon, self.output_dir+'recon_{}.png'.format(str(epoch)),
                                   nrow=min(x_recon.size(0),8), normalize=True)

        pbar.write("[Training Finished]")
        pbar.close()

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def encode_state(state, device):
        state = state.to(device)
        mu, std, _ = self.net(state.view(-1, 90*90))
        z = reparameterize(mu, std)
        return z

    
