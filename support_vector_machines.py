from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import math

class Hyperlane:

    def __init__(self):

        self.w = 0
        self.b = 0

"""
Klasa SupportVectorMachines służy do przeprowadzenia uczenia maszynowego za pomocą metody fit oraz klasyfikacji za pomocą metody predict.
"""
class SupportVectorMachines(BaseEstimator, ClassifierMixin):

    """
    Konstruktor klasy tworzy listę na przechowywanie hiperpłaszczyzn potrzebnych w SVMie (dla 2 cech są to linie proste).
    """
    def __init__(self):

        self.hyperlanes = []

    """
    Metoda fit służy do przeprowadzenia uczenia - dostosowania klasyfikatora do danych uczących.

    Paramaetry:
    - x_train - wektor cech ciągu uczącego,
    - y_train - poprawne wyniki klasyfikacji ciągu uczącego,
    - batch_size - ilość danych brana "naraz" (domyślnie 100),
    - learning_rate - stała ograniczająca zmiany generowanych prostych (domyślnie 0.0001),
    - epochs - ilość iteracji algorytmu (domyślnie 100).

    Zwraca: self - wyuczony obiekt klasyfikatora.
    """
    def fit(self, x_train, y_train, batch_size=100, learning_rate=0.0001, epochs=100):

        # Zbieramy informację o ilości cech, klas oraz obiektów do nauki:
        n_features = x_train.shape[1]
        self.n_classes = np.max(y_train) + 1
        number_of_samples = x_train.shape[0]

        # Dla każdej klasy tworzymy osobną hiperpłaszczyznę:
        for i in range(self.n_classes):

            hyperlane = Hyperlane()

            # Dzielimy zbiór poprawnych odpowiedzi na te z obrabianej obecnie klasy (wartość 1) oraz wszystkich innych klas (wartość -1).
            # Inne klasy muszą mieć wartość przeciwną, aby znajdować się po drugiej stronie hiperpłaszczyzny:
            divided_Y = np.copy(y_train)
            for j in range(len(divided_Y)):
                if divided_Y[j] == i:
                    divided_Y[j] = 1
                else:
                    divided_Y[j] = -1

            # Tworzymy listę numerów obiektów do klasyfikacji:
            ids = np.arange(number_of_samples)

            # Tworzymy początkowe (zerowe) dane hiperpłaszczyzny - wektor normalny i bias:
            w = np.zeros((1, n_features))
            b = 0

            # Dla każdej epoki poprawiamy odpowiednio dane płaszczyzny:
            for i in range(epochs):

                # Wykonujemy poprawkę gradientów kilka razy:
                for batch_initial in range(0, number_of_samples, batch_size):
                    gradw = 0
                    gradb = 0

                    # Na tym etapie najważniejsze jest aby wyliczyć ti:
                    for j in range(batch_initial, batch_initial + batch_size):
                        if j < number_of_samples:
                            x = ids[j]
                            ti = divided_Y[x] * (np.dot(w, x_train[x].T) + b)

                            # Jeśli ti jest większe niż 1, nie zmieniamy gradintów:
                            if ti > 1:
                                gradw += 0
                                gradb += 0
                            else:

                                # Gdy ti jest mniejsze lub równe 1, wyliczamy poprawki gradientów:
                                gradw += divided_Y[x] * x_train[x]
                                gradb += divided_Y[x]

                    # Po poprawkach gradientów dodajemy je do danych hiperpłaszczyzny (uwzględniając learning_rate):
                    w = w - learning_rate * w + learning_rate * gradw
                    b = b + learning_rate * gradb

            # Wpisujemy dane do obiektu hiperpłaszczyzny oraz dodajemy hiperpłaszczyznę do listy hiperpłaszczyzn:
            hyperlane.w = w
            hyperlane.b = b

            self.hyperlanes.append(hyperlane)

        return self

    """
    Metoda predict służy do przeprowadzenia klasyfikacji na zadanym wektorze cech.

    Parametry:
    - x_test - wektor cech.

    Zwraca: y_pred - wyniki klasyfikacji wyliczone przez algorytm.
    """
    def predict(self, x_test):

        y_pred = []

        # Dla kazdego obiektu z ciągu testowego spradzamy, dla której klasy ma najwięcej punktów. Tą klasę wybieramy.
        for item in x_test:

            class_points = []
            for hyperplane in self.hyperlanes:
                # Przeprowadzamy równanie wektora normalnego: w * x + b.
                # Jest to po prostu iloczyn skalarny wektora normalnego hiperpłaszczyzny z wektorem cech danego testowanego obiektu plus bias:
                points = np.dot(item, hyperplane.w[0]) + hyperplane.b
                class_points.append(points)

            item_class = np.argmax(class_points)
            y_pred.append(item_class)

        return y_pred

    """
    Metoda get_hyperplane_value służy do wyznaczenia punktów danej hiperpłaszczyzny (w wypadku 2 cech - prostej).
    Celem jest potem narysowanie tej prostej na wykresie.

    Parametry:
    - x - punkt początkowy, dla którego wyliczana jest prosta,
    - plane_num - numer prostej (numer klasy).

    Zwraca: Punkt końcowy prostej.
    """
    def get_hyperplane_value(self, x, plane_num: int):

        margin = 1
        return (-1* self.hyperlanes[plane_num].w[0][0] * x + self.hyperlanes[plane_num].b + margin) / self.hyperlanes[plane_num].w[0][1]
