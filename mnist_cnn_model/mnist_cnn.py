from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# MNIST图片大小为28×28*1, channel is 1,
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolution:
        # in_channels: Number of channels in the input image, 此处为1
        # out_channels: Number of channels produced by the convolution, 此处为20
        # kernel:5*5
        # strike: 1
        # conv1: kernel为5×5, 深度为20, 步长为1,
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # conv2: kernel为5×5, 深度为50, 步长为1
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # 全连接层1
        # 参数：in_features: 输入矩阵大小； out_features: 输出矩阵大小; bias：是否使用bias
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # 全连接层2
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # 第一层卷积、激活
        x = F.relu(self.conv1(x))
        # 第一层池化  参数为input:x,  kernel:2*2, strike: 2
        x = F.max_pool2d(x, 2, 2)
        # 处理后变为12*12*20
        # 第二层卷积、激活
        x = F.relu(self.conv2(x))
        # 第二层池化  参数为input:x,  kernel:2*2, strike: 2
        x = F.max_pool2d(x, 2, 2)
        # 处理后变为4*4*50
        # 补充，view做了reshape操作, 放在这里似乎有点多余？
        # 再次补充，这步处理是把矩阵变为一维，为了最后的全连接层做准备，这步是必须的，因为最后的输出参数
        # 只有10个
        x = x.view(-1, 4 * 4 * 50)
        # 全连接层1、激活
        x = F.relu(self.fc1(x))
        # 全连接层2
        x = self.fc2(x)
        # 到此输出为[10]
        # log_softmax, log(exp(x_i)/exp(x).sum()), dim表示维度, 取值有范围限制
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    # 设置构建的cnn网络为train模式
    model.train()
    # 可以看出train_loader可以像这样迭代，每次取出训练数据与训练label
    # 需要注意的是每次取出batch_size个数据, 在本程序中默认是64
    for batch_idx, (data, target) in enumerate(train_loader):
        # to(device)是把数据放在cpu or gpu中计算
        data, target = data.to(device), target.to(device)
        # gradient全部初始化为0
        optimizer.zero_grad()
        # 此处应该是调用了forward函数, 完成batch_size个数据的一次前向传播过程
        output = model(data)
        # loss指的是损失, 参数为训练答案、正确答案
        loss = F.nll_loss(output, target)
        # 定义了loss的微分公式, 返回值是梯度
        loss.backward()
        # 进行单次优化, 更新所有的参数， 与backward()一起使用
        optimizer.step()

        # loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    # 设置构建的cnn网络为evaluation模式
    model.eval()
    # test_loss为总的损失和
    test_loss = 0
    correct = 0
    # torch.no_grad()禁用梯度计算，减小内存开销
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # reduction = 'sum'表示输出之和将会被累加
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # 返回output中最大值的索引
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # 判断二者是否相等，如果相等则corrent+1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    # epochs迭代次数
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    # lr 学习率
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    # momentum未知
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    # no-cuda指定是否使用cuda
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # seed未知
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # save-model的路径
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # 设定生成随机数的种子，并返回一个 torch._C.Generator对象.
    torch.manual_seed(args.seed)

    # device指定使用cpu还是gpu
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # train = True : 使用训练数据
    # train = False : 使用测试数据
    # transform似乎是对图片按transform列表中的操作进行叠加处理
    # datasets.MNIST有一个len方法返回数据集长度, getitem方法返回index索引的数据集以及标签
    # transform.ToTensor()是把图片范围由「0-255」变为「0-1」
    # transform.Normalize()是把图片按mean与std进行处理,第一个参数是一个channel的均差，另一个参数
    # 是一个channel的标准差
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    # 优化参数、学习率、momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        # 传入cl参数、训练网络model、训练device：cpu or gpu、
        # 训练数据train_loader、优化函数optimizer、训练轮数epoch
        train(args, model, device, train_loader, optimizer, epoch)
        # 传入cl参数、训练网络model、训练device：cpu or gpu、
        # 测试数据test_loader
        test(args, model, device, test_loader)

    # 如果指定模型保存路径则将其保存
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
