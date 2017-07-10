import numpy as np
import random
import math
import urllib
import webapp2
import logging
from paste import httpserver

# Kmeans algorithm:
# fill 'cluster' column in the given dataset with k clusters
# input:
#   points - a numpy matrix containing all points
#   k - number of clusters
#   max_iter - maximum number of iterations


class KMeans(object):
    def __init__(self, k, max_iter):
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
        points = np.array(points)
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
        labels = np.add(labels, 1)
        points = np.c_[points,labels]
        return points

class Labels (webapp2.RequestHandler):
        DEFAULT_K = 4
        DEFAULT_MAX_ITER = 300

        def _parse_input_data(self, points):
                #remove '[]' brackets and convert to list
                if points[0] != '[' or points[-1] != ']':
                        raise Exception('Data point should be wrapped with brackets')
                        return

                points_list = points[1:-1].split(";")
                new_list = []
                for i in range(len(points_list)):
                        new_list.append(list(float(x) for x in points_list[i].split(',')))
                return new_list

        def _points_str_for_output(self, points):
                output_str = '['
                for i in points:
                        #convert each inner array to string and add ';'
                        output_str += '{:.1f},{:.1f},{:d}'.format(i[0],i[1],int(i[2])) + ';'
                #remove last ';' and add closing ']'
                output_str = output_str[:-1] + ']'         
                return output_str

        def post(self):
                try:
                        # initialize with request paramenters
                        k = int(self.request.get('num_clusters', default_value = self.DEFAULT_K))
                        max_iter = int(self.request.get('max_iterations', default_value = self.DEFAULT_MAX_ITER))
                        points = self._parse_input_data(self.request.body)

                except Exception as err:
                        handle_400(self.request, self.response, err)
                        return

                # prepare the response
                self.response.headers['Content-Type'] = 'text/plain'
                try:
                        labeled_points = KMeans(k, max_iter).kmeans(points)
                        labeled_points = self._points_str_for_output(labeled_points)
                        self.response.write('{points}'.format(points=labeled_points))
                except Exception as err:
                        handle_500(self.request, self.response, err)
                        return

def handle_400(request, response, exception):
        logging.exception(exception)
        response.write('Bad Request!')
        response.set_status(400)

def handle_500(request, response, exception):
        logging.exception(exception)
        response.write('A server error occurred!')
        response.set_status(500)

app = webapp2.WSGIApplication([
    ('/clustering/labels', Labels),
], debug=True)


def main():
        app.error_handlers[400] = handle_400
        app.error_handlers[500] = handle_500
        httpserver.serve(app, host='0.0.0.0', port='80')

if __name__ == '__main__':
    main()

