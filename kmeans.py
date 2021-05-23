import numpy as np
from cv2 import imread, imwrite

from utils import plot_losses

def make_assignment(X, means):
    loss = 0
    assignment = []
    for x in X:
        distances = np.sum((means - x)**2, axis=-1) ** 0.5
        idx = np.argmin(distances)
        loss += distances[idx]
        assignment.append(idx)
    assignment = np.array(assignment)
    return assignment, loss

def calculate_means(X, assignment, K):
    means = np.zeros((K, 3))
    counts = np.full(K, 0)
    for x, idx in zip(X, assignment):
        means[idx] += x
        counts[idx] += 1
    for k in range(K):
        if counts[k] > 0:
            means[k] /= counts[k]
        else:
            # If no points are assigned to this cluster, re-initialize mean
            means[k] = np.random.rand(3) 
    return means

def KMClustering(X, K):
    # random initialization
    means = np.random.rand(K, 3)
    loss = 0; old_loss = 1e-3; losses = []
    itr = 1
    # Stop when loss changes by less than 1 percent
    while abs((old_loss - loss) / (old_loss + 1e-10)) >= 0.01:
        old_loss = loss
        assignment, loss = make_assignment(X, means)
        means = calculate_means(X, assignment, K)
        print("iteration #",itr,"loss:",loss)
        itr += 1
        losses.append(loss)
    return means, assignment, losses

def kmeans(image:str) -> None:
    imarr = imread(image)
    imshape = imarr.shape
    X = imarr.reshape(-1, 3) / 255.0
    for K in [3, 5, 7]:
        print("---------------K = {} clusters-------------".format(K))
        np.random.seed(42)
        means, assignment, losses = KMClustering(X, K)
        compressed_X = np.array([means[idx] for idx in assignment])
        compressed_im = compressed_X.reshape(imshape) * 255.0
        imwrite('compressed_K-{}.png'.format(K), compressed_im)
        plot_losses(losses, 'KMeans_loss_K-{}.png'.format(K))

if __name__ == "__main__":
    kmeans('umn_csci.png')