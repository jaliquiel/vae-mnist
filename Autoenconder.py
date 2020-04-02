#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset

seed_value = 1
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
    
    #if use_cuda: 
torch.cuda.manual_seed_all(seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False

#random_seed(1234, True)

# random_seed(1234, False)
#torch.manual_seed(0)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class CustomView:
    def __init__(self,new_size):
        self.new_size = new_size
    def __call__(self, img):
        return torch.reshape(img, self.new_size)


# In[3]:


batch_size = 100
transform = [torchvision.transforms.ToTensor(), CustomView((-1,))]
train_loader = train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/data',
                                                        train=True, download=True, 
                                                        transform = torchvision.transforms.Compose(transform)),
                                                        batch_size=batch_size, shuffle=True)


# In[4]:


print(next(iter(train_loader))[0].size())

'''
batches, channels, dim_x, dim_y
batches=image.shape[0], (flatten(dim_x, dim_y))
'''


# In[21]:


class VAE(nn.Module):
    def __init__(self,num_inputs, num_hidden, code_size, drop_prob):
        super(VAE, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.code_size = code_size
        self.drop_prob = drop_prob

        self.Q = nn.Linear(self.num_inputs, self.num_hidden)
#         self.Q = nn.Linear(784, 200)

        self.mu = nn.Linear(self.num_hidden, self.code_size)
        self.var = nn.Linear(self.num_hidden, self.code_size)
  
        #self.code = nn.Linear(self.num_hidden, self.code_size)
        
        self.P = nn.Linear(self.code_size, self.num_hidden)
        self.output = nn.Linear(self.num_hidden, self.num_inputs)
        
        
        self.dropout = nn.Dropout(self.drop_prob)
        self.sigmoid = nn.Sigmoid()
        
        

    def forward(self, x):
        # Max pooling over a (2, 2) window
        
        x = self.dropout(F.tanh(self.Q(x)))
        
        mu = self.dropout(F.tanh(self.mu(x)))
        sigma = torch.exp(self.dropout(F.tanh(self.var(x))))
 
        z = torch.empty(self.code_size).normal_(mean=0,std=1) * sigma + mu
        
        p = F.tanh(self.P(z))
        out = self.sigmoid(self.output(p))
        return out, mu, sigma



# In[44]:


def main():
    num_inputs=784 
    num_hidden=200 
    code_size=32 
    drop_prob=0.5
    
    model = VAE(num_inputs=num_inputs, num_hidden=num_hidden, code_size=code_size, drop_prob=drop_prob)
    images, labels = next(iter(train_loader))
    out, mu, sigma = model(images)
#     example = out[50].detach().numpy().reshape(28,28)
#     plt.imshow(example, cmap='binary')
#     plt.show()

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




