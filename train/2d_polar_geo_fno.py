import torch.nn.functional as F
from Geo_FNO.Adam import Adam
from Geo_FNO.utilities3 import *
import torch.nn as nn
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(6, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, f, j):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, torch.unsqueeze(f,-1), torch.unsqueeze(j,-1), grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# load data and data normalization
################################################################
def main():
    path = '../data/geofno'
    #path = 'data/chara_fno'#'../data/polar_line/chara_fno'

    PATH_Sigma = path + '/u_omesh.npy'
    PATH_XY = path + '/xy.npy'
    PATH_F = path + '/f_mesh.npy'
    PATH_J = path + '/jump.npy'


    INPUT_XY = np.load(PATH_XY)
    OUTPUT_Sigma = np.load(PATH_Sigma)
    INPUT_F = np.load(PATH_F)
    INPUT_J = np.load(PATH_J)
    #OUTPUT_Sigma = OUTPUT_Sigma[:,:,::-1].copy()

    Ntotal = 4000
    ntrain = 3000
    ntest = 1000

    batch_size = 20
    learning_rate = 1e-3

    epochs = 2000
    step_size = 1
    gamma = 0.5

    modes = 12
    width = 32

    # h = int(((41 - 1) / r) + 1)
    s1 = int(100)
    s2 = int(100)

    input = torch.tensor(INPUT_XY, dtype=torch.float)
    output = torch.tensor(OUTPUT_Sigma, dtype=torch.float)
    f = torch.tensor(INPUT_F, dtype=torch.float)
    j = torch.tensor(INPUT_J, dtype=torch.float)
    x_train = input[:ntrain].reshape(ntrain, s1, s2, 2)
    y_train = output[:ntrain].reshape(ntrain, s1, s2)
    x_test = input[-ntest:].reshape(ntest, s1, s2, 2)
    y_test = output[-ntest:].reshape(ntest, s1, s2)
    f_train = f[:ntrain].reshape(ntrain, s1, s2)
    f_test = f[-ntest:].reshape(ntest, s1, s2)
    j_train = j[:ntrain].reshape(ntrain, s1, s2)
    j_test = j[-ntest:].reshape(ntest, s1, s2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, f_train, j_train, y_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, f_test, j_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width).cuda()
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)

    loss_history = []
    for ep in range(epochs):
        model.train()
        #t1 = default_timer()
        train_l2 = 0
        for x, f, j, y in train_loader:
            x, f, j, y = x.cuda(), f.cuda(), j.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x, f, j)

            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, f, j, y in test_loader:
                x, f, j, y = x.cuda(),f.cuda(), j.cuda(),y.cuda()
                out = model(x, f, j)

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        #t2 = default_timer()

        if (ep+1)%step_size==0:
            loss_history.append([ep, train_l2, test_l2])
            print(ep, train_l2, test_l2)

    torch.save(model, 'fno_chara_{}_{}_{}.pt'.format(modes,width, epochs))
    np.savetxt('loss_chara_{}_{}.txt'.format(modes,width), np.array(loss_history))


if __name__ == '__main__':
    main()
