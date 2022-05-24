class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 上面相当于对每个不同的图片，卷积核去自适应的进行改变
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # padding=(kernel_size - 1) // 2 相当于paddin=same，即保持输出图片大小不变的操作
        # 为啥这里进入的通道数是1呢，是因为前面有个自适应层，将图片变成了1*1*channel这个样子，在下面经过维度变换，此时将维度变成了b*1*c，然后conv1d是对最后一维进行卷积的（同理conv2d是对最后两维进行卷积的）因此就是对channel这个维度进行了一个卷积，此时就可以相当于把一个长方体横过来看（或者说换成了channel和长这个面）此时相当于宽为以前的通道数即1.
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y.squeeze(-1)是将最后一个维度删掉即宽这个维度就没有了，transpose(-1, -2)是将最后一个和倒数第二个维度进行互换，即现在的维度变成了b，1，c这三个维度，1是由于前面的自适应平均层变成了1*1的图像，所以长在这里就是1。unsqueeze(-1)是增加最后一个维度
        y = self.sigmoid(y)
        return x * y.expand_as(x)
