import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Geo_FNO.Adam import Adam
from Geo_FNO.utilities3 import *

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# 3D fourier layer
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        # 需要8个权重，对应8个象限
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,z), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # 3D FFT

        # Multiply relevant Fourier modes
        s1, s2, s3 = x.size(-3), x.size(-2), x.size(-1)
        out_ft = torch.zeros(batchsize, self.out_channels, s1, s2, s3 // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # 处理8个象限
        # 象限1: 前 modes1, 前 modes2, 前 modes3
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)

        # 象限2: 后 modes1, 前 modes2, 前 modes3
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)

        # 象限3: 前 modes1, 后 modes2, 前 modes3
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)

        # 象限4: 后 modes1, 后 modes2, 前 modes3
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # 象限5: 前 modes1, 前 modes2, 后 modes3
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:], self.weights5)

        # 象限6: 后 modes1, 前 modes2, 后 modes3
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:], self.weights6)

        # 象限7: 前 modes1, 后 modes2, 后 modes3
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:], self.weights7)

        # 象限8: 后 modes1, 后 modes2, 后 modes3
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:], self.weights8)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(s1, s2, s3))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, padding=9):
        super(FNO3d, self).__init__()

        """
        3D FNO network
        input: the solution of the coefficient function and locations (a(x, y, z), x, y, z)
        input shape: (batchsize, x=s, y=s, z=s, c=4)
        output: the solution 
        output shape: (batchsize, x=s, y=s, z=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(4, self.width)  # input channel: (a(x,y,z), j, x, y, z)

        # 6个Fourier层
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        # 6个卷积层（残差连接）
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.w4 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: input tensor shape (batch, s1, s2, s3)
        # j: additional input tensor shape (batch, s1, s2, s3)

        grid = self.get_grid(x.shape, x.device)

        # 连接输入: (a, j, x, y, z)
        x = torch.cat((torch.unsqueeze(x, -1), grid), dim=-1)

        x = self.fc0(x)  # (batch, s1, s2, s3, width)

        # 改变维度顺序: (batch, s1, s2, s3, width) -> (batch, width, s1, s2, s3)
        x = x.permute(0, 4, 1, 2, 3)

        # 如果需要padding
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        # 第1层
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # 第2层
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # 第3层
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # 第4层
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        # 第5层
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2

        # 移除padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]

        # 改变维度顺序: (batch, width, s1, s2, s3) -> (batch, s1, s2, s3, width)
        x = x.permute(0, 2, 3, 4, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.squeeze(-1)  # 移除最后一个维度

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]

        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])

        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])

        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float, device=device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])

        return torch.cat((gridx, gridy, gridz), dim=-1)

def trilinear_interp_batch(input_tensor):
    """
    输入: tensor of shape [batch, D, H, W] = [batch, 21, 21, 21]
    输出: tensor of shape [batch, D, H, W] = [batch, 41, 41, 41]
    """
    # 1. 检查输入形状
    batch_size = input_tensor.shape[0]

    # 2. 添加channel维度 [batch, 1, D, H, W]
    input_with_channel = input_tensor.unsqueeze(1)  # shape: [batch, 1, 21, 21, 21]

    # 3. 三线性插值
    output = F.interpolate(
        input_with_channel,
        size=(41, 41, 41),
        mode='trilinear',  # 3D插值
        align_corners=True
    )  # shape: [batch, 1, 41, 41, 41]

    # 4. 移除channel维度
    output = output.squeeze(1)  # shape: [batch, 41, 41, 41]

    return output


################################################################
# load data and data normalization
################################################################
def main():
    #path = 'data'
    path = '../data/3d'

    PATH_Sigma = path + '/solve.npy'
    PATH_XY = path + '/area_sdf_11.npy'

    INPUT_XY = np.load(PATH_XY)
    OUTPUT_Sigma = np.load(PATH_Sigma)

    Ntotal = 2000
    ntrain = 1000
    ntest = 1000

    batch_size = 5
    learning_rate = 1e-3

    epochs = 1000
    step_size = 1
    gamma = 0.5

    modes = 6
    width = 32

    # h = int(((41 - 1) / r) + 1)
    s1 = int(11)
    s2 = int(11)
    s3 = int(11)

    input = torch.tensor(INPUT_XY, dtype=torch.float)
    output = torch.tensor(OUTPUT_Sigma, dtype=torch.float)
    x_train = input[:ntrain].reshape(ntrain, s1, s2, s3)
    y_train = output[:ntrain].reshape(ntest, 41, 41, 41)
    x_test = input[-ntest:].reshape(ntest, s1, s2, s3)
    y_test = output[-ntest:].reshape(ntest, 41, 41, 41)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO3d(modes, modes, modes, width).cuda()
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)

    loss_history = []
    for ep in range(epochs):
        model.train()
        #t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            pre = trilinear_interp_batch(out.squeeze())

            loss = myloss(pre.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)

                pre = trilinear_interp_batch(out.squeeze())

                test_l2 += myloss(pre.view(batch_size,-1),y.view(batch_size,-1)).item()

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