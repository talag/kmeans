
# coding: utf-8

import numpy as np
import random
import math

# returns k random centroids from the given dataset
def random_centroids(points, k):
    centroids = points[np.random.choice(points.shape[0], k, replace=False), :]
    return centroids

# returns an array with indexes to the nearest centroid for each point in points
def nearest_neigbours(points, centroids):
    # add new dimension to centroids, use broadcasting to substract all centroids from each point, 
    # such that each sub space of credentials contains results for one point vs. centroids
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

# returns updated centroids according to the means of the assigned points
def update_centroids(points, centroids, labels):
    return np.array([points[labels == k].mean(axis=0) for k in range(centroids.shape[0])])


# function kmeans(df, k, max_iter)
# fill 'cluster' column in the given dataset with k clusters
# input: 
#   points - a numpy matrix containing all points
#   k - number of clusters 
#   max_iter - maximum number of iterations

def kmeans(points, k=4, max_iter=300):
    # first step: initialization: generate k random clusters 
    centroids = random_centroids(points, k)
    
    # label datapoints with current centroids
    labels = nearest_neigbours(points, centroids)
    
    # this turns to false when clusters remained the same after some iteration, which means we should stop
    clusters_updated = True
    
    # initialize cluster columon which will later contain the label 
    i = 0
    
    while (i < max_iter and clusters_updated):
        # Save centroids for testing convergence.
        old_centroids = centroids
        
        # Calculate new centroids based on datapoint current labels
        centroids = update_centroids(points, centroids, labels)

        # label datapoints with new centroids
        labels = nearest_neigbours(points, centroids)
    
        # convergence test
        if (np.array_equal(centroids, old_centroids)):
            clusters_updated = False
        
        i += 1
    
    # the output labels should start from index 1
    labels += 1
    return centroids, labels

def createRandomInput(len):
    points = np.random.uniform(low=0.0, high=100.0, size=(len,2))
    return points

import webapp2

def main(argv):
    #points, k, max_iter = handleInput()
    #if max_iter == None:
    #    max_iter = 300
    #if k == None:
    #    k = 4
    #if max_iter <= 0:
    #    raise ValueError("Number of iterations should be a positive number, got %d instead" % max_iter)
    #if k > n_samples:
    #    raise ValueError( "n_samples=%d should be larger than k=%d" % (n_samples, k))

    #points = createRandomInput(100)
    #centroids, labels = kmeans(points)
    #print labels

class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Hello, World!')


app = webapp2.WSGIApplication([
    ('/', MainPage),
], debug=True)

#if __name__ == "__main__":
#    main(sys.argv)



