# coding: utf-8

import numpy as np
import random
import math

# Kmeans algorithm:
# fill 'cluster' column in the given dataset with k clusters
# input: 
#   points - a numpy matrix containing all points
#   k - number of clusters 
#   max_iter - maximum number of iterations


class KMeans(object):
    def __init__(self, k=4, max_iter=300):
        self.k = k
        self.max_iter = max_iter

    # returns k random centroids from the given dataset
    def _random_centroids(self, points):
        centroids = points[np.random.choice(points.shape[0], self.k, replace=False), :]
        return centroids
    
    # returns an array with indexes to the nearest centroid for each point in points
    def _nearest_neigbours(self, points, centroids):
        # add new dimension to centroids, use broadcasting to substract all centroids from each point, 
        # such that each sub space of credentials contains results for one point vs. centroids
        distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    # returns updated centroids according to the means of the assigned points    
    def _update_centroids(self, points, centroids, labels):
        return np.array([points[labels == k].mean(axis=0) for k in range(centroids.shape[0])])
    
    def kmeans(self, points):
        # first step: initialization: generate k random clusters 
        centroids = self._random_centroids(points)
    
        # label datapoints with current centroids
        labels = self._nearest_neigbours(points, centroids)
    
        # this turns to false when clusters remained the same after some iteration, which means we should stop
        clusters_updated = True
    
        # initialize cluster columon which will later contain the label 
        i = 0
    
        while (i < self.max_iter and clusters_updated):
            # Save centroids for testing convergence.
            old_centroids = centroids
        
            # Calculate new centroids based on datapoint current labels
            centroids = self._update_centroids(points, centroids, labels)

            # label datapoints with new centroids
            labels = self._nearest_neigbours(points, centroids)
    
            # convergence test
            if (np.array_equal(centroids, old_centroids)):
                clusters_updated = False
        
            i += 1
    
        # the output labels should start from index 1
        labels += 1
        points = np.c_[points,labels] 
        return points

def createRandomInput(len):
    points = np.random.uniform(low=0.0, high=100.0, size=(len,2))
    return points

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
#points = KMeans(k=3, max_iter=100).kmeans(points)

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

class Labels (webapp2.RequestHandler):
        def post(self):
                #points = createRandomInput(100)
                #centroids, labels = kmeans(points)
                k = self.request.get('num_clusters', default_value=4)
                max_iter = self.request.get('max_iterations', default_value=300)
                points = self.request.body
                self.response.headers['Content-Type'] = 'text/plain'
                self.response.write('k is {k}, max_iter is {max_iter}, points are {points}'.format(k=k, max_iter=max_iter, points=points))


app = webapp2.WSGIApplication([
    ('/clustering/labels', Labels),
], debug=True)

#if __name__ == "__main__":
#    main(sys.argv)



