import torch
import torch.nn  as nn
import numpy as np
from einops import rearrange, reduce, repeat

class HD(nn.Module):
    def __init__(self, infeature, patch_size:float, reduction_ratio=16):
        super(HD, self).__init__()
        self.patch_size = patch_size
        self.linear1 = nn.Linear(in_features=infeature, out_features=infeature//reduction_ratio)
        self.relu = nn.ReLU(inplace= True)
        self.linear2 = nn.Linear(in_features=infeature//reduction_ratio, out_features=infeature)
        self.softmax = nn.Softmax(2)
        

    """ calculate the defference between local and global contextual inofrmation """
    def difference(self, x) -> torch.Tensor:
        #x (b, c, h, w)
        # print(x.size(2), self.patch_size)#self.patch_size*
        # print(x.size())#self.patch_size*
        """ extract the local and global contextual information and merge them """
        lci = rearrange(x, 'b c (h1 h2) (w1 w2) ->b c h1 h2 w1 w2', h2 = int(self.patch_size*x.size(2)), w2 = int(self.patch_size*x.size(3))) #(b, c, h/patch_size, patch_size, w/patch_size, patch_size)
        lci = rearrange(lci, 'b c h1 h2 w1 w2 -> b c h1 w1 h2 w2') #(b, c, h/patch_size, w/patch_size, patch_size, patch_size)
        lci = reduce(lci, 'b c h1 w1 h2 w2 -> b c h1 w1', 'mean') #(b, c, h/patch_size, w/patch_size)
        lci = rearrange(lci, 'b c h1 w1 -> b (h1 w1) c') #(b (hw/patch_size^2) c)
        gci = reduce(x, 'b c h w -> b 1 c', 'mean') #(b 1 c)
        ci = torch.cat((lci, gci), 1) #(b (hw/patch_size^2+1) c)
        """ MLP operation """
        ci = self.linear1(ci) #(b (hw/patch_size^2+1) c/reduction_ratio)
        ci = self.relu(ci)
        ci = self.linear2(ci) #(b (hw/patch_size^2+1) c)
        """ Softmax operation """
        ci = self.softmax(ci) #(b (hw/patch_size^2+1) c)
        """ log operation """
        # ci = torch.log(ci) #(b (hw/patch_size^2+1) c)
        """ split the local and global contextual information """
        lci, gci = torch.split(ci, (ci.shape[1]-1, 1), 1) #(b (hw/patch_size^2) c) (b 1 c)
        gci = rearrange(gci, 'b k c -> b c k') #(b c 1)
        """ crossentropy -> calculate the difference """
        result = torch.matmul(lci, gci) #(b (hw/patch_size^2) 1)
        return reduce(result, 'b k c -> b k', 'mean')

    def forward(self, x):
        identity = x 
        diff = self.difference(x) #(b (hw/patch_size^2))
        diff = rearrange(diff, 'b (h1 w1) -> b 1 h1 w1 1 1', h1 = x.size(2)//int(self.patch_size*x.size(2)), w1 = x.size(3)//int(self.patch_size*x.size(3))) #(b h/patch_size, w/patch_size)
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1 h2 w2', h2 = int(self.patch_size*x.size(2)), w2 = int(self.patch_size*x.size(3)))
        x = diff * x
        x = rearrange(x, 'b c h1 w1 h2 w2 -> b c (h1 h2) (w1 w2)', h2 = int(self.patch_size*identity.size(2)), w2 = int(self.patch_size*identity.size(3)))

        #short_cut
        return identity + x 
