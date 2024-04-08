import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from utils import *
from flow import *
import os
import torch.multiprocessing
from pathlib import Path
import sys
from matplotlib import offsetbox
from datetime import datetime
import pytz

data_path = "./data/MNIST_FMNIST.tar"
device = 'cpu'
data = torch.load(data_path,map_location=torch.device(device))
source_data = data["source_data"].to(device).float()
source_mean = data["source_mean"].to(device).float()
source_cov = data["source_cov"].to(device).float()
target_data = data["target_data"].to(device).float()
target_data_cpu = data["target_data"].float().cpu().numpy()
target_mean = data["target_mean"].to(device).float()
target_cov = data["target_cov"].to(device).float()
clustered_labels = data["labels"]

data_one_class = 300
one_class = 20
target_size = 1

image_size = (1,1,20,20)
image_shape = (1,20,20)
alpha = 0.002
beta = 0.01
gamma = 50.0

step_size_0 = 0.5
step_size = step_size_0
momentum=0.9
rmsprop = 0.9
momentum_flag = False
rmsprop_flag = True
adam_flag = False
adam_beta1 = 0.9
adam_beta2 = 0.999
eps=1e-8

noise_beta_0 =0.02
noise_beta = noise_beta_0
num_steps = 150
num_classes = end_class-start_class
num_target_images = num_classes*target_size

num_images = num_classes*one_class
mini_batch = num_images/2

images = source_data
mean = source_mean
source_mean_max = torch.max(mean)
source_mean_min = torch.min(mean)
mean = (mean-source_mean_min)/(source_mean_max-source_mean_min)
cov = source_cov

K_images = target_data
K_mean = target_mean
target_mean_max= torch.max(K_mean)
target_mean_min=torch.min(K_mean)
K_mean = (K_mean-target_mean_min)/(target_mean_max-target_mean_min)
K_cov = target_cov
clustered_labels = clustered_labels.astype(int)


x_shape = (num_images,images.shape[1])
target_shape = (num_target_images,images.shape[1])
mean_shape = (num_images,mean.shape[1])
cov_shape = (num_images,cov[0].shape[0],cov[0].shape[1])

x_tau = torch.zeros(x_shape, device=device)
mean_tau = torch.zeros(mean_shape, device=device)
cov_tau = torch.zeros(cov_shape, device=device)

last_x_grad = torch.zeros_like(x_tau)
last_mean_grad = torch.zeros_like(mean_tau)
last_cov_grad = torch.zeros_like(cov_tau)

labels = np.zeros(num_images)

K_x_tau = torch.zeros(target_shape, device=device)
labels_target = np.zeros(num_target_images)

objectives = []
identity = torch.eye(cov[0].shape[0],cov[0].shape[1], device=device)

now = datetime.now()
dt_string = now.strftime("%d_%m_%H_%M")
path = "results_M_F/1_shot/"+dt_string+"/"

Path(path).mkdir(parents=True, exist_ok=True)

with open(path+"setup.txt", "w") as f:
    f.write(f"data: {data_path}\n")
    f.write(f"number of images for each class: {one_class}, number of classes: {num_classes}, minibatch: {mini_batch}\n")
    f.write(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}, step_size: {step_size}\n")
    if momentum_flag:
        f.write(f"use momentum, {momentum}\n")
    elif rmsprop_flag:
        f.write(f"use rmsprop, {rmsprop}\n")
    elif adam_flag:
        f.write(f"use adam, beta1: {adam_beta1}, beta2: {adam_beta2}\n")
    f.write(f"noise level: {noise_beta}\n")


indices = np.arange(num_images)
indices_j = np.arange(num_images)
one_class_indices = np.arange(one_class)
np.random.shuffle(indices)
with torch.no_grad():
    for k in range(num_steps): 
        #select the minibatch    
        if k%2==0:
            batches = indices[:int(num_images/2)]
        else:
            batches = indices[int(num_images/2):]
            np.random.shuffle(indices)

        if k%40==0 and k!=0:
            step_size = step_size_0/(1+k/40)
            noise_beta = np.sqrt(1/(k*step_size_0))*noise_beta_0

        if k ==0:
            np.random.shuffle(one_class_indices)
            x_index = 0

            for c in range(num_classes):
                for i in range(one_class):
                    index = c*data_one_class+i
                    
                    x_tau[x_index] = images[index]
                    mean_tau[x_index] = mean[c]
                    cov_tau[x_index] = cov[c]
                    labels[x_index] = c
                    x_index +=1
                        
                for i in range(target_size):
                    index = c*data_one_class+one_class_indices[i]
                    K_x_tau[c*target_size+i] = K_images[index]
                    labels_target[c*target_size+i]=c
                
            labels = labels.astype(int)   
            labels_target = labels_target.astype(int)
            target_obj = compute_target_obj_images(K_x_tau, K_mean, K_cov, alpha, beta, gamma, labels_target, size=num_target_images)  

            save_tensor_images(torch.reshape(x_tau,(num_images,image_shape[0],image_shape[1],image_shape[2])),path, "init_source",num_images)
            save_tensor_images(torch.reshape(K_x_tau,(num_target_images,image_shape[0],image_shape[1],image_shape[2])),path, "init_target", num_images)

        for i in batches:     
            psi_bar_x = torch.zeros(images.shape[1], device=device)
            psi_bar_mean = torch.zeros(mean.shape[1], device=device)
            psi_bar_cov = torch.zeros((cov[0].shape[0],cov[0].shape[1]), device=device)

            psi_x = torch.zeros(images.shape[1], device=device)
            psi_mean = torch.zeros(mean.shape[1], device=device)
            psi_cov = torch.zeros((cov[0].shape[0],cov[0].shape[1]), device=device)          
            
            noise = torch.randn(400, device = device)*noise_beta
            noise_mean = torch.randn(2, device = device)*noise_beta
            noise_cov = torch.randn(2,2, device = device)*noise_beta
            noise_cov[1,0] = noise_cov[0,1]
            
            lya_sol = solve_lyapunov(cov_tau[i], noise_cov).to(device)
            cov_tau_p= (identity+step_size*lya_sol)@cov_tau[i]@(identity+step_size*lya_sol)  
            
            for j in range(num_target_images):           
                label_j = int(labels_target[j])

                psi1, psi2, psi3 = gradient(alpha, beta, gamma, x_tau[i]+noise, mean_tau[i]+noise_mean, cov_tau_p, K_x_tau[j], K_mean[label_j], K_cov[label_j])
                psi_bar_x += psi1.squeeze()

                psi_bar_mean += psi2.squeeze() 
                psi_bar_cov += psi3.squeeze()

            for j in batches:
                psi1, psi2, psi3 = gradient(alpha, beta, gamma, x_tau[i]+noise, mean_tau[i]+noise_mean, cov_tau_p, x_tau[j], mean_tau[j], cov_tau[j])

                psi_x += psi1.squeeze()
                psi_mean += psi2.squeeze()
                psi_cov += psi3.squeeze()

            psi_bar_x = psi_bar_x/num_target_images
            psi_bar_mean = psi_bar_mean/num_target_images
            psi_bar_cov = psi_bar_cov/num_target_images

            psi_x = psi_x/mini_batch
            psi_mean = psi_mean/mini_batch
            psi_cov = psi_cov/mini_batch

            # update x
            if momentum_flag:
                x_tau[i] += step_size*(psi_bar_x - psi_x)+momentum*last_x_grad[i]
            elif rmsprop_flag:
                if k==0:
                    last_x_grad[i]=0.1*torch.square(psi_bar_x - psi_x) 
                else:
                    last_x_grad[i]=0.9*last_x_grad[i]+0.1*torch.square(psi_bar_x - psi_x)        
                x_tau[i] += step_size*torch.div(psi_bar_x - psi_x, torch.sqrt(last_x_grad[i]+eps))

            else:
                last_x_grad[i]=adam_beta1*last_x_grad[i]+(1-adam_beta1)*(psi_bar_x - psi_x)
                last_x_v[i]=adam_beta2*last_x_v[i]+(1-adam_beta2)*torch.square(psi_bar_x - psi_x)
                
                last_x_grad_scaled = last_x_grad[i]/adam_scale1            
                last_x_v_scaled = last_x_v[i]/adam_scale2

                x_tau[i] += step_size*torch.div(last_x_grad_scaled, torch.sqrt(last_x_v_scaled)+eps)
                
            # update mean
            if momentum_flag:
                mean_tau[i] += step_size*(psi_bar_mean - psi_mean)+momentum*last_mean_grad[i]
            elif rmsprop_flag:    
                if k==0:
                    last_mean_grad[i]=0.1*torch.square(psi_bar_mean - psi_mean)
                else:
                    last_mean_grad[i]=rmsprop*last_mean_grad[i]+(1-rmsprop)*torch.square(psi_bar_mean - psi_mean)
                mean_tau[i] += step_size*torch.div(psi_bar_mean - psi_mean, torch.sqrt(last_mean_grad[i]+eps))
                
            else:
                last_mean_grad[i]=adam_beta1*last_mean_grad[i]+(1-adam_beta1)*(psi_bar_mean - psi_mean)
                last_mean_v[i]=adam_beta2*last_mean_v[i]+(1-adam_beta2)*torch.square(psi_bar_mean - psi_mean)
                
                last_mean_grad_scaled = last_mean_grad[i]/adam_scale1
                last_mean_v_scaled = last_mean_v[i]/adam_scale2
                mean_tau[i] += noise_mean+step_size*torch.div(last_mean_grad_scaled, torch.sqrt(last_mean_v_scaled)+eps)
              
            lya_sol = solve_lyapunov(cov_tau_p, psi_bar_cov-psi_cov).to(device)
            cov_tau[i] = (identity+step_size*lya_sol)@cov_tau[i]@(identity+step_size*lya_sol)           

        ob = compute_objective_images(x_tau, mean_tau, cov_tau, K_x_tau, K_mean, K_cov, alpha, beta, gamma, labels_target, num_images)
        objectives.append(0.5*(ob+target_obj))   

        # save flow results every 10 steps
        if k %10==0:           
            print("step: ",k)
            source_images = []
            source_mean_cpu = mean_tau.cpu().numpy()
            source_mean_plt = []       
            save_tensor_images(x_tau.reshape(num_images,image_shape[0],image_shape[1],image_shape[2]), path, k, num_images, image_shape)
            results_tau = {}
            results_tau["source_data"] = x_tau
            results_tau["source_mean"] = mean_tau
            results_tau["source_cov"] = cov_tau
            results_tau["labels"] = labels
            data_path = path+str(k)+".tar"
            torch.save(results_tau, data_path)

            plt.clf()
            plt.plot(
                range(int(len(objectives))), 
                torch.Tensor(objectives),
            )
            plt.ylabel('objective')
            plt.xlabel('steps')
            plt.savefig(path+str(k)+"_obj.png")
            plt.close('all')
   
