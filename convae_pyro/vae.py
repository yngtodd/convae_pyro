import torch
import torch.nn as nn
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones

from networks import Encoder, Decoder


# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400, enc_kernel1=5, enc_kernel2=5, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, enc_kernel1, enc_kernel2)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # setup hyperparameters for prior p(z)
        # the type_as ensures we get cuda Tensors if x is on gpu
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)
        # decode the latent code z
        mu_img = self.decoder.forward(z)
        # score against actual images
        pyro.observe("obs", dist.bernoulli, x.view(-1, 784), mu_img)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder.forward(x)
        # sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_mu, z_sigma = self.encoder(x)
        # sample in latent space
        z = dist.normal(z_mu, z_sigma)
        # decode the image (note we don't sample in image space)
        mu_img = self.decoder(z)
        return mu_img

    def model_sample(self, batch_size=1):
        # sample the handwriting style from the constant prior distribution
        prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
        prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
        zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma)
        mu = self.decoder.forward(zs)
        xs = pyro.sample("sample", dist.bernoulli, mu)
        return xs, mu
