import numpy as np
from scipy import linalg
import torch

def compute_gradcov(gamma, sigma_x, sigma_y):
    return 2*gamma*(2*sigma_x.dot(sigma_x)-sigma_x.dot(sigma_y)-sigma_y.dot(sigma_x))

def compute_gradm(beta, mean_x, mean_y):
    return beta*(mean_x-mean_y)

def sqrtm_gpu(A, func='symeig'):
    if func == 'symeig':
        s, v = A.symeig(eigenvectors=True) # This is faster in GPU than CPU, fails gradcheck. See https://github.com/pytorch/pytorch/issues/30578
    elif func == 'svd':
        _, s, v = A.svd()                 
    else:
        raise ValueError()

    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps

    s = torch.where(above_cutoff, s, torch.zeros_like(s))

    sol =torch.matmul(torch.matmul(v,torch.diag_embed(s.sqrt(),dim1=-2,dim2=-1)),v.transpose(-2,-1))

    return sol

def sqrtm(A):
    device = A.device
    return torch.from_numpy(linalg.sqrtm(A.detach().cpu().numpy())).to(device)

def gradient(alpha, beta, gamma, x, mean_x, sigma_x, y, mean_y, sigma_y):

    grad_x = alpha*(x-y)
    grad_mean = beta*(mean_x-mean_y)
    grad_cov = torch.zeros_like(sigma_x)
    
    grad_cov = 2.0*gamma*(2.0*sigma_x@sigma_x-sigma_x@sigma_y-sigma_y@sigma_x)

    multiplier = -2.0*torch.exp(-alpha*torch.norm(x-y)**2-beta*torch.norm(mean_x-mean_y)**2-gamma*torch.norm(sigma_x-sigma_y)**2)
#     multiplier = -2.0*torch.exp(-alpha*torch.norm(x-y)**2-beta*torch.norm(mean_x-mean_y)**2)
    
#     print(alpha*torch.norm(x-y)**2, beta*torch.norm(mean_x-mean_y)**2, gamma*torch.norm(sigma_x-sigma_y)**2)
#     print(multiplier)
    return (multiplier*grad_x).squeeze(), (multiplier*grad_mean).squeeze(), (multiplier*grad_cov).squeeze()


def label_dist(mean_x, mean_y, cov_x, cov_y):
    bures = torch.trace(cov_x + cov_y - 2*sqrtm_gpu(sqrtm_gpu(cov_x)@cov_y@sqrtm_gpu(cov_x)))
#     print(bures,torch.norm(mean_x-mean_y)**2)
    return torch.sqrt(bures+torch.norm(mean_x-mean_y)**2)

def gradient_batch(alpha, beta, gamma, x, mean_x, sigma_x, y, mean_y, sigma_y):
    grad_x = alpha*(x-y)
    grad_mean = beta*(mean_x-mean_y)
    grad_cov = torch.zeros_like(sigma_x)
    
    grad_cov = 2.0*gamma*(2.0*torch.bmm(sigma_x,sigma_x)-torch.bmm(sigma_x,sigma_y)-torch.bmm(sigma_y,sigma_x))

    multiplier = -2.0*torch.exp(-alpha*torch.norm(x-y,dim=1)**2-beta*torch.norm(mean_x-mean_y,dim=1)**2-gamma*torch.norm(sigma_x-sigma_y,dim=(1,2))**2).unsqueeze(1)
    return torch.mul(multiplier.repeat(1,400),grad_x).squeeze(), torch.mul(multiplier.repeat(1,2),grad_mean).squeeze(), torch.mul(multiplier.unsqueeze(2).repeat(1,2,2),grad_cov).squeeze()

def solve_lyapunov(A, V):
    sol = linalg.solve_continuous_lyapunov(A.cpu().numpy(), V.cpu().numpy())
    return torch.from_numpy(sol)

def solve_lyapunov_cpu(A, V):
    sol = linalg.solve_continuous_lyapunov(A.cpu().numpy(), V.cpu().numpy())
    return torch.from_numpy(sol)

# implement a lyapunov solver here
def solve_lyapunov_gpu(A, V):
    D, U = torch.linalg.eigh(A)
    W = U@V@U
    n = A.shape[0]
    G = torch.zeros(n,n)
    Dx, Dy = torch.meshgrid(D,D)
    G = torch.div(W, (Dx+Dy))
    return U@G@U

def solve_lyapunov_gpu2(A, V):
    device = A.get_device()
    n = A.shape[0]    
    s = A.reshape(-1)
    v = V.reshape(-1)
    I = torch.eye(n, device=device)
    M = torch.kron(I,A)+torch.kron(A,I)
    X = torch.linalg.solve(M, v)
                 
    return X.reshape(n,n)
    
def compute_objective(x_tau, mean_tau, cov_tau, target_data, target_mean, target_cov, alpha, beta, gamma, labels, samemean, size):
    obj = 0.0
    for i in range(size):
        for j in range(size):
            label_j = labels[j]
            obj += torch.exp(-alpha*torch.norm(x_tau[:,i]-x_tau[:,j])**2-beta*torch.norm(mean_tau[:,i]-mean_tau[:,j])**2-gamma*torch.norm(cov_tau[i]-cov_tau[j])**2)
            obj -= 2*torch.exp(-alpha*torch.norm(x_tau[:,i]-target_data[:,j])**2-beta*torch.norm(mean_tau[:,i]-target_mean[label_j])**2-gamma*torch.norm(cov_tau[i]-target_cov[label_j])**2)

    obj = obj/(size*size)
    return obj

def compute_objective_images(x_tau, mean_tau, cov_tau, target_data, target_mean, target_cov, alpha, beta, gamma, labels, size):
    obj = 0.0
    size = x_tau.shape[0]
    size2 = target_data.shape[0]
    for i in range(size):
        for j in range(size):
            obj += torch.exp(-alpha*torch.norm(x_tau[i]-x_tau[j])**2-beta*torch.norm(mean_tau[i]-mean_tau[j])**2-gamma*torch.norm(cov_tau[i]-cov_tau[j])**2)/(size*size)
     
    for i in range(size):
        for j in range(size2):        
            label_j = labels[j]
            obj -= 2*torch.exp(-alpha*torch.norm(x_tau[i]-target_data[j])**2-beta*torch.norm(mean_tau[i]-target_mean[label_j])**2-gamma*torch.norm(cov_tau[i]-target_cov[label_j])**2)/(size*size2)

    return obj

def compute_target_obj(target_data, target_mean, target_cov, alpha, beta, gamma, labels, size):
    obj = 0.0
    for i in range(size):
        for j in range(i,size):
            label_j = labels[j]
            obj += 2*torch.exp(-alpha*torch.norm(target_data[:,i]-target_data[:,j])**2-beta*torch.norm(target_mean[label_j]-target_mean[label_j])**2-gamma*torch.norm(target_cov[label_j]-target_cov[label_j])**2)

    obj = obj/(size*size)
    return obj

def compute_target_obj_images(target_data, target_mean, target_cov, alpha, beta, gamma, labels, size):
    obj = 0.0
    for i in range(size):
        for j in range(i,size):
            label_i = labels[i]
            label_j = labels[j]
            obj += 2*torch.exp(-alpha*torch.norm(target_data[i]-target_data[j])**2-beta*torch.norm(target_mean[label_i]-target_mean[label_j])**2-gamma*torch.norm(target_cov[label_i]-target_cov[label_j])**2)

    obj = obj/(size*size)
    return obj