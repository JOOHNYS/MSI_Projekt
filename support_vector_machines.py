from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import math

class SupportVectorMachines(BaseEstimator, ClassifierMixin):

    def __init__(self, iterations = 300, random_state = 0, optimization_factor = 0.1):
        
        self.opt_factor = optimization_factor
        self.iter = iterations
        self.hyperplanes = []
        self.rs = random_state
        if self.rs == 0:
            self.rs = int(np.random.random() * 10000)

    def fit(self, x_train, y_train):

        self.n_classes = len(set(y_train))

        for i in range(self.n_classes):
            
            good_x = []
            bad_x = []
            for j in range(len(y_train)):
                if y_train[j] == i:
                    good_x.append(x_train[j])
                else:
                    bad_x.append(x_train[j])

            self.hyperplanes.append(self.get_hyperplane(np.array(good_x), np.array(bad_x)))

        return self
    
    def predict(self, x_test):

        test_items_number = len(x_test)

        y_predicted = [self.n_classes - 1] * test_items_number

        for i in range(test_items_number):

            for j in range(len(self.hyperplanes) - 1):
                if self.is_x_under_hyperlane(x_test[i], self.hyperplanes[j]):
                    y_predicted[i] = j
                    break

        return y_predicted
    
    def get_hyperplane(self, good_x, bad_x) -> list:

        line_parameters = good_x.shape[1]
        hyperlane = [0] * (line_parameters)
        good_mean = np.mean(good_x, axis = 0)
        bad_mean = np.mean(bad_x, axis = 0)

        for i in range(line_parameters):
            hyperlane[i] = np.mean([good_mean[i], bad_mean[i]])

        for _ in range(self.iter):

            new_hyperlane = hyperlane
            np.random.seed(self.rs)

            index_to_change = int(np.random.random() * 10000) % len(new_hyperlane)
            change_factor = (np.random.random() -0.5) * self.opt_factor
            new_hyperlane[index_to_change] += change_factor

            if self.is_new_hyperlane_better(hyperlane, new_hyperlane, good_x, bad_x):
                hyperlane = new_hyperlane

        print(hyperlane)
        return hyperlane

    def distance_from_point_to_hyperlane(self, x_point: list, hyperlane: list) -> float:

        # Z definicji odległości punktu od płaszczyzny:
        numerator = abs(np.dot(x_point, hyperlane))
        denominator = math.sqrt(np.sum(hyperlane**2))

        return numerator / denominator
    
    def get_min_distance(self, data_x, hyperlane: list) -> float:

        min_distance = 99999
        for data in data_x:

            tested_distance = self.distance_from_point_to_hyperlane(data, np.array(hyperlane))
            if tested_distance < min_distance:
                min_distance = tested_distance

        return min_distance
    
    def is_new_hyperlane_better(self, old_hyperlane, new_hyperlane, good_x, bad_x) -> bool:

        old_distance_from_good = self.get_min_distance(good_x, old_hyperlane)
        old_distance_from_bad = self.get_min_distance(bad_x, old_hyperlane)

        new_distance_from_good = self.get_min_distance(good_x, new_hyperlane)
        new_distance_from_bad = self.get_min_distance(bad_x, new_hyperlane)

        return new_distance_from_good > old_distance_from_good and new_distance_from_bad > old_distance_from_bad
    
    def is_x_under_hyperlane(self, x: list, hyperlane: list) -> bool:

        for i in range(len(x)):

            if x[i] < hyperlane[i]:
                return False
            
        return True
