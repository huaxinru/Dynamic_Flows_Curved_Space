from MNIST.digits_model import *

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from MNIST.data import CustomTensorDataset,NormalizeRangeTanh,UnNormalizeRangeTanh
from torchvision.utils import make_grid
import urllib
from torch.utils.data.dataloader import default_collate


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5,range=(0.0,1.0))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.show()

    
class classifierFTest(): 
    def __init__(self, eps, use_gpu=True):
        self.log = {}
        self.log['best_model'] = None
        self.log['train_loss'] = []
        self.log['val_loss'] = []
        self.log['train_accuracy'] = []        
        self.log['val_accuracy'] = []
        self.eps = eps
        self.use_gpu = use_gpu
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None
    
    def create_data_loaders(self, dataset):
        self.dataset = dataset
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        transform = transforms.Compose([
            torchvision.transforms.Resize((20,20)),
            transforms.ToTensor(),
        ])
        
        def my_collate(batch):
            modified_batch = []
            for item in batch:
                image, label = item
                if label <5:
                    modified_batch.append(item)
            return default_collate(modified_batch)
        
        if dataset=='MNIST':
            train_set = datasets.MNIST(root='/home/ubuntu/datasets/', train=True, download = True, transform=transform)
            size = len(train_set)
            val_size = int(size*0.2)
            mnist_train, mnist_val = train_set[:-val_size], train_set[-val_size:]
            
            self.train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(mnist_val, batch_size=256, shuffle=True, num_workers=8)

            test_set = datasets.MNIST(root='/var/local/', train=False, download = True, transform=transform)
            self.test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=8)


    def create_model(self):
        self.model = LeNet5().cuda()
    
    def create_loss_function(self):
        self.loss_function = nn.CrossEntropyLoss()

    def create_optimizer(self):
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
    
    def train_model(self, start_epoch, num_epochs, **kwargs):

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels = data
 
                if self.use_gpu:       
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                inputs.requires_grad=True
#                 print(torch.max(inputs))
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                if epoch>0 and self.adversarial:
                    
#                     x_ad, y_ad = self.fast_pgd(inputs, labels, self.eps)
                    if torch.randn(1)>0:
                        x_ad, y_ad = self.SSIM_attack(inputs, labels, self.eps)
                    else:
                        x_ad, y_ad = self.SSIM_rev_attack(inputs, labels, self.eps)
#                     show_tensor_images(x_ad)
                    outputs = self.model(x_ad)
                    loss = self.loss_function(outputs, y_ad)         
                else:        
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)            
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.cpu().detach().numpy()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.data).sum()    
            
            correct = 1. * correct / total
            running_loss = running_loss / len(self.train_loader)
            print('[%dth epoch]' % (start_epoch+epoch))
            print('training loss: %.4f   accuracy: %.3f%%' % (running_loss, 100 * correct))
            self.log['train_loss'].append(running_loss)
            self.log['train_accuracy'].append(correct)
            self.log['best_model'] = self.model.state_dict()
            checkpoint = './models/'+self.dataset+"_lenet_" + str(start_epoch+epoch) + '.tar'
            torch.save(self.log, checkpoint)
            
            acc = self.test_model(self.val_loader)
            print('validation accuracy: %.3f%%' % (100 * acc))
        print('Finished Training')       
                        
        acc = self.test_model(self.test_loader)
        print('Test accuracy: %.3f%%' % (100 * acc))
   
    def test_model(self, dataloader):
        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                #inputs = torch.cat((inputs, inputs, inputs), 1)    

                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                running_loss += loss.cpu().detach().numpy()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.data).sum()

            correct = 1. * correct / total
            running_loss = running_loss / len(self.test_loader)
    #         self.log['val_loss'].append(running_loss)
    #         self.log['val_accuracy'].append(correct)
        return correct
