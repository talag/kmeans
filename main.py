import numpy as np
import urllib
import webapp2
import logging
from paste import httpserver
from kmeans import KMeans

# this class handles the request and returns the labeled data points
class Labels (webapp2.RequestHandler):
        # default num_clusters and max_iterations values
        DEFAULT_K = 4
        DEFAULT_MAX_ITER = 300

        # convert the given data points to an array of pair lists
        def _parse_input_data(self, points):
                #remove '[]' brackets and convert to list
                points_list = points[1:-1].split(";")
                new_list = []
                for i in range(len(points_list)):
                        new_list.append(list(float(x) for x in points_list[i].split(',')))
                return new_list

        # prepare a string to output for the labeled data, in the required format
        def _points_str_for_output(self, points):
                output_str = '['
                for i in points:
                        #convert each inner array to string and add ';'
                        output_str += '{:.1f},{:.1f},{:d}'.format(i[0],i[1],int(i[2])) + ';'
                #remove last ';' and add closing ']'
                output_str = output_str[:-1] + ']'
                return output_str

        # validate input request parameters
        def _validate_input(self, k, max_iter, points):
                if (points[0] != '[' or points[-1] != ']'):
                        raise Exception('Data points should be wrapped with square brackets')
                if (k < 1 or max_iter < 1):
                        raise Exception('num_clusters and max_iterations should be at least 1')
                if (k > points.count(';')+1):
                        raise Exception('number of clusters should be at least as number of data points')

        def post(self):
                try:
                        # initialize with request paramenters
                        k = int(self.request.get('num_clusters', default_value = self.DEFAULT_K))
                        max_iter = int(self.request.get('max_iterations', default_value = self.DEFAULT_MAX_ITER))
                        points = self.request.body

                        self._validate_input(k, max_iter, points)

                        points = self._parse_input_data(points)

                except Exception as err:
                        # Bad request
                        handle_400(self.request, self.response, err)
                        return

                try:
                        labeled_points = KMeans(k, max_iter).fit(np.array(points))
                        labeled_points = self._points_str_for_output(labeled_points)

                        # prepare the response
                        self.response.headers['Content-Type'] = 'text/plain'
                        self.response.write('{points}'.format(points=labeled_points))

                except Exception as err:
                        # Internal server error
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
