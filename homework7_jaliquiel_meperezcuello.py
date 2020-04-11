#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print('device', device, torch.version.cuda)

seed_value = 1
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
#if use_cuda: 
torch.cuda.manual_seed_all(seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False

get_ipython().run_line_magic('matplotlib', 'inline')


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.files = file_path
        self.data = np.load(self.files)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.data).float()
        return x[index]
        
    def __len__(self):
        return len(self.data)


class CustomView:
    def __init__(self,new_size):
        self.new_size = new_size
    def __call__(self, img):
        return torch.reshape(img, self.new_size)


batch_size = 50
transform = [torchvision.transforms.ToTensor(), CustomView((-1,))]
mnist_trainset = MyDataset('mnist_train_images.npy')
train_loader= torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

test_batch_size = 16
test_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=test_batch_size, shuffle=True)


print(next(iter(test_loader)).size())

'''
batches, channels, dim_x, dim_y
batches=image.shape[0], (flatten(dim_x, dim_y))
'''


class VAE(nn.Module):
    def __init__(self,num_inputs, num_hidden, code_size, drop_prob):
        super(VAE, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.code_size = code_size
        self.drop_prob = drop_prob

        self.Q = nn.Linear(self.num_inputs, self.num_hidden)

        self.mu = nn.Linear(self.num_hidden, self.code_size)
        self.var = nn.Linear(self.num_hidden, self.code_size)
          
        self.P = nn.Linear(self.code_size, self.num_hidden)
        self.output = nn.Linear(self.num_hidden, self.num_inputs)
        
        self.dropout = nn.Dropout(self.drop_prob)
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x):
        x = self.dropout(torch.tanh(self.Q(x)))
        mu = self.dropout(torch.tanh(self.mu(x)))
        sigma = self.dropout(torch.tanh(self.var(x))) # should we multiple for 0.5 or hjust exp
        return mu, sigma
    
    def decoder(self, z):
        p = torch.tanh(self.P(z))
        out = self.sigmoid(self.output(p))
        return out
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        mu, sigma = self.encoder(x)
        z = torch.empty(self.code_size).normal_(mean=0,std=1).to(device) * torch.exp(sigma) + mu #[batch, L=monte carlo steps, z_dim]
        out = self.decoder(z)
        return out, mu, sigma


def criterion(output, ground_truth, mu, var, L):
    BCE = F.binary_cross_entropy(output, ground_truth.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + var - mu**2 - torch.exp(var))
    return (L*BCE) + KLD


def epoch_time(init_time, end_time):
    elapsed_time = end_time - init_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = float(end_time - init_time)
    return elapsed_mins, elapsed_secs


def training(loader, model, optimizer, L):
    train_loss = 0
    train_losses = []
        
    model.train()
    for i, images in enumerate(loader, start=1):
        model.zero_grad()
        images = images.to(device)
            
        outputs, mu, var = model(images)
        loss = criterion(outputs, images, mu, var, L).to(device)
            
        loss.backward()
        optimizer.step()
            
        train_loss += loss.detach().item()
        
    train_loss = train_loss/i
    train_losses.append(train_loss)
    
    return train_loss, train_losses    
    

def main():
    num_inputs=784 
    num_hidden=200 
    code_size=32 

    drop_probs = 0.05
    learning_rates = 1e-4 
    epochs = 500 
    L = 0.1
        
    hyperparams = [(drop_probs, learning_rates, epochs, L)]
    best_train_loss = float('inf')
    
    for drop_prob, lr, epochs, l in hyperparams:
        
        model = VAE(num_inputs=num_inputs, num_hidden=num_hidden, code_size=code_size, drop_prob=drop_prob).to(device)    
        optimizer = optim.Adam(model.parameters(), lr=lr)    
        
        print('Drop_ prob [{}], Learning Rate[{}], Epochs[{}], L[{}]'.format(drop_prob, lr, epochs, l)) 
        
        for epoch in range(epochs):
            init_timer=time.time()

            train_loss, train_losses = training(train_loader, model, optimizer, l)

            end_timer = time.time()
            epoch_mins, epoch_secs = epoch_time(init_timer, end_timer)

            print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print('\t Train Loss:%.4f'%(train_loss))
            
        
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print(f"\t\t Better accuracy: {best_train_loss}")
                checkpoint = {'model':model,
                              'train_loss':best_train_loss,
                              'hyperparams': [drop_prob, lr, epochs, L],
                              'state_dict':model.state_dict(),
                              'optimizer':optimizer.state_dict()}

                torch.save(checkpoint, 'last_checkpoint.pth')
                
        checkpoint = {'model':model,'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, f'data/checkpoint_{train_loss}_{epochs}_{lr}_{drop_prob}_{l}.pth')
            
        
    example = torch.rand(1,32).to(device)
    example = model.decoder(example)
    print(example.size())
    example = example.cpu().detach().numpy().reshape(28,28)
    plt.imshow(example, cmap='gray', interpolation='none')
    plt.show()

        
if __name__ == '__main__':
    main()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model


model = load_checkpoint('last_checkpoint.pth') #last_checkpoint.pth')

def generator(model):
    z=torch.randn(1,32).to(device) 
    rand_sample = model.decoder(z)
    return rand_sample.cpu().detach().numpy().reshape(28,28)

image = np.zeros((28,1))
images_2 = np.zeros((28,1))
generated_images = np.zeros((28,1))

for img in next(iter(test_loader)):
    a = img.numpy().reshape(28,28)
    image = np.concatenate((image,a), axis=1)
    # recreate the images
    outputs, mu, var  = model(img.to(device))
    images_2 = np.concatenate((images_2, outputs.cpu().detach().numpy().reshape(28,28)),axis=1)
    # generate images
    generated_images = np.concatenate((generated_images, generator(model)),axis=1)

a = np.vstack((image,images_2))
a = np.vstack((a,generated_images))
    
plt.imshow(a, cmap='gray', interpolation='none')
plt.show()
