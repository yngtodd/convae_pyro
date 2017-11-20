import torch
import torch.nn as nn


fudge = 1e-7

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dim=784, enc_kernel1=5, enc_kernel2=5):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=enc_kernel1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=enc_kernel2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*32, hidden_dim) # Need to fix hard coding.
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.view(-1, 784)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        # then compute the hidden units
        conv2_reshaped = conv2.view(x.size(0), -1)
        hidden = self.softplus(self.fc1(conv2_reshaped))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dim=784):
        super(Decoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_dim)
        #self.deconv1 = nn.ConvTranspose2d(7*7*32, )
        #self.deconv2 = nn.ConvTranspose2d(, input_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        # fixing numerical instabilities of sigmoid with a fudge
        mu_img = (self.sigmoid(self.fc21(hidden))+fudge) * (1-2*fudge)
        return mu_img
