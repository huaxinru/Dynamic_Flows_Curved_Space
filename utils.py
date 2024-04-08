import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import torchvision
from torchvision.utils import make_grid
from flow import label_dist
import ot
# compute the mean and variance given a dataloader
# assume there are 10 classes
def gauss_2d(n, mu, sigma):
#     x = np.zeros((n,2))
    x = np.random.multivariate_normal(mean=mu, cov=sigma*0.02, size=(n))
#     for i in range(n):
#         x[i] = np.random.multivariate_normal(mean=mu, cov=sigma, size=(1))
#         p = multivariate_normal.pdf(x[i], mean=mu, cov=sigma)
#         while p<0.7:
#             x[i] = np.random.multivariate_normal(mean=mu, cov=sigma, size=(1))
#             p = multivariate_normal.pdf(x[i], mean=mu, cov=sigma)
    return x[:,0], x[:,1]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', mean = None, cov=None, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    if cov is None:
        cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      alpha = 0.2, fc = facecolor, ec="None", **kwargs)
#     ellipse.set_fill(True)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    if mean is None:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
    else:
        mean_x = mean[0]
        mean_y = mean[1]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#different numbers of class
def plot_data_diffclass(source, target, labels, labels2, source_color, target_color, numclass=2,numclass2=2, xlim=None, ylim=None, source_mean=None, source_cov=None, target_mean=None, target_cov=None, ax=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(source[0], source[1], s=3, c=source_color)
    scale = 0.2
    if source_mean is not None:
        for i in range(numclass):

            mask = labels==i
            if i==0:
                confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=1.0*scale,
                               label='source', facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
            else:
                confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=1.0*scale,
                                facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
            confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=2.0*scale,
                                facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
            confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=3.0*scale,
                                facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
    else:
        for i in range(numclass):

            mask = labels2==i
            if i==0:
                confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=1.0*scale,
                               label='source', facecolor=source_color)
            else:
                confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=1.0*scale,
                                facecolor=source_color)
            confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=2.0*scale,
                                facecolor=source_color)
            confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=3.0*scale,
                                facecolor=source_color)

    ax.scatter(target[0], target[1], s=3, c=target_color)
    if target_mean is not None:           
        for i in range(numclass2):
            mask = labels==i

            if i==0:
                confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=1.0*scale,
                               label='target', facecolor=target_color, mean = target_mean[i], cov = target_cov[i])
            else:
                confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=1.0*scale,
                                facecolor=target_color, mean = target_mean[i], cov = target_cov[i])
            confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=2.0*scale,
                                facecolor=target_color, mean = target_mean[i], cov = target_cov[i])
            confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=3.0*scale,
                                facecolor=target_color, mean = target_mean[i], cov = target_cov[i])
        
    else:   
        for i in range(numclass2):
            mask = labels==i

            if i==0:
                confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=1.0*scale,
                               label='target', facecolor=target_color)
            else:
                confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=1.0*scale,
                                facecolor=target_color)
            confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=2.0*scale,
                                facecolor=target_color)
            confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=3.0*scale,
                                facecolor=target_color)
    ax.legend()
    if xlim!=None:
        plt.xlim([xlim[0],xlim[1]])
    if ylim!=None:
        plt.ylim([ylim[0],ylim[1]])
    plt.show()
    return ax.get_xlim(), ax.get_ylim()

#same number of classes
def plot_data_sameclass(source, target, labels, source_color, target_color, numclass, source_mean, source_cov, target_mean, target_cov, path = None, step = None, xlim=None, ylim=None, ax=None):    
    if ax==None:
        fig, ax = plt.subplots(figsize=(10, 10))
    scale = 0.5
    ax.scatter(source[0], source[1], s=3, c=source_color)
    for i in range(numclass):

        mask = labels==i
        if i==0:
            confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=1*scale,
                           label='source', facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
        else:
            confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=1*scale,
                            facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
        confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=2*scale,
                            facecolor=source_color, mean =source_mean[i], cov=source_cov[i])
        confidence_ellipse(source[0][mask], source[1][mask], ax, n_std=3*scale,
                            facecolor=source_color, mean =source_mean[i], cov=source_cov[i])

    ax.scatter(target[0], target[1], s=3, c=target_color)
 
    for i in range(numclass):
        mask = labels==i
               
        if i==0:
            confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=1*scale,
                           label='target', facecolor=target_color, mean =target_mean[i], cov=target_cov[i])
        else:
            confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=1*scale,
                            facecolor=target_color, mean =target_mean[i], cov=target_cov[i])
        confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=2*scale,
                            facecolor=target_color, mean =target_mean[i], cov=target_cov[i])
        confidence_ellipse(target[0][mask], target[1][mask], ax, n_std=3*scale,
                            facecolor=target_color, mean =target_mean[i], cov=target_cov[i])
    ax.legend()
    if xlim!=None:
        plt.xlim([xlim[0],xlim[1]])
    if ylim!=None:
        plt.ylim([ylim[0],ylim[1]])

    plt.show()
    return ax.get_xlim(), ax.get_ylim()


def get_covariance(features):
    return np.cov(np.array(features), rowvar=False)

def get_mean(features):
    return np.mean(np.array(features),axis=0)

def compute_stats(dataloader, size):
    images = [[] for _ in range(10)]
    means = []
    covs = []
    for i, (image, label) in enumerate(dataloader): 
        images[label].append(image.numpy().reshape(-1))
        if i == size:
            break
        
    for i in range(10):
        means.append(get_mean(images[i]))
        covs.append(get_covariance(images[i]))
                          
    return np.array(means),np.array(covs)
        
        
def compute_matrix_barycenter(covs, nbIter = 1000):
    # Input: 
    # covs: a list of covariance matrices
    # nbIter: number of iterations for the fixed point algorithm
    
    # Dependency: scipy.linalg, numpy.linalg
    
    # find the number of matrices and the matrix dimension
    n = len(covs)
    dim = covs[0].shape[0]
    
    # all weights are equal
    weights = [1 / n for _ in range(n)]

    wb_cov = np.eye(dim)
    itr_count = 0
    while itr_count < nbIter:
        itr_count += 1

        cov_rt = sqrtm(wb_cov) # S_n
        inv_cov_rt = np.linalg.inv(cov_rt)

        Q = 0
        for j in range(n):
            tmp = np.matmul(cov_rt, np.matmul(covs[j], cov_rt))
            tmp = sqrtm(tmp)
            tmp = weights[j] * tmp
            Q += tmp
        Q = np.matmul(Q, Q)
        wb_cov = np.matmul(inv_cov_rt, np.matmul(Q, inv_cov_rt)) # S_{n+1}

    
    return wb_cov

def plot_embedding(X, images, target_images, target_means, one_class, step, title=None):
    all_points = np.concatenate((X,target_means))
    x_min, x_max = np.min(all_points, 0), np.max(all_points, 0)
    X = (X - x_min) / (x_max - x_min)
    target_means= (target_means - x_min) / (x_max - x_min)
    plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
        
#     for i in range(X.shape[0]):
#         plt.text(X[i, 0], X[i, 1], str(y[i]),
#                  color=plt.cm.Set1(y[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
    for i in range(target_means.shape[0]):
        plt.text(target_means[i, 0], target_means[i, 1], str(i),
                 color=plt.cm.Set1(i / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        for i in range(target_means.shape[0]):
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(target_images[i*one_class].reshape(20,20), cmap=plt.cm.gray_r,zoom=2.0), target_means[i])
            ax.add_artist(imagebox)
        shown_images = np.array([[1., 1.]])
        for i in range(X.shape[0]):
#             dist = np.sum((X[i] - shown_images) ** 2, 1)
#             print(dist)
#             if np.mean(dist)>0.25:
#                 continue
#             if np.min(dist) < 8e-4:
#                 don't show points that are too close
#                 continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i].reshape(20,20), cmap=plt.cm.gray_r,zoom=1.0), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.savefig(path+str(step)+"_TSNE.png")
    plt.close()

def solve_labels_POT(x_tau, mean_tau, cov_tau, target_images, target_mean, target_cov):
    num_class = 10
    num_images = x_tau.shape[0]
    kept_mask = np.ones(num_images)
    for i in range(num_images):
        min_dist = 10.0
        for j in range(num_class):
            dist = torch.norm(x_tau[i].cpu()-target_images[j].reshape(-1).cpu())
            if dist<min_dist:
                min_dist = dist
        if min_dist>2.0:
            kept_mask[i]=0

    num_final_images = int(np.sum(kept_mask))
    print("Number of final images is",num_final_images) 
    final_images = x_tau[kept_mask==1]
    final_mean = mean_tau[kept_mask==1]
    final_cov = cov_tau[kept_mask==1]
    
    
    M = np.zeros((num_final_images,num_class))
    a = np.zeros(num_final_images)+1.0/num_final_images
    b = np.zeros(num_class)+1.0/num_class
    mean_target_images = torch.zeros(num_class,400)
    for i in range(num_final_images):
        for j in range(num_class):
            M[i,j]=label_dist(final_mean[i],target_mean[j],final_cov[i],target_cov[j])

    theta = ot.emd(a, b, M)
    labels = torch.from_numpy(np.argmax(theta,axis=1))

    return final_images, labels

def save_tensor_images(image_tensor, path, step, num_images=25, size=(1, 28, 28)):
    '''
    Function for saving images
    '''
    image_tensor = image_tensor*0.2712+0.1266
    image_tensor[torch.where(image_tensor<0)] =0
    image_tensor[torch.where(image_tensor>1)] =1
    image_tensor = 1.0-image_tensor
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=10,normalize=True,scale_each=True,padding=0)
    plt.imsave(path+str(step)+".png", image_grid.permute(1, 2, 0).squeeze().cpu().numpy())
    plt.close()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((20,20)),
    torchvision.transforms.ToTensor()
])

def save_tensor_images_cifar(image_tensor, path, step, num_images=25, size=(3, 32, 32)):
    '''
    Function for saving images
    '''
    image_tensor = image_tensor/2+0.5
    image_tensor[torch.where(image_tensor<0)] =0
    image_tensor[torch.where(image_tensor>1)] =1
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=10,normalize=True,scale_each=True,padding=0)
    plt.imsave(path+str(step)+".png", image_grid.permute(1, 2, 0).squeeze().cpu().numpy())
    plt.close()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((20,20)),
    torchvision.transforms.ToTensor()
])



def show_tensor_images(image_tensor, num_images=200, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
#     image_tensor = image_tensor.clamp(0.0,1.0)
    image_unflat = image_tensor.detach().cpu()
    plt.figure(figsize=(20,10))
   
    image_grid = make_grid(image_unflat[:num_images], nrow=10,normalize=True,scale_each=True)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
