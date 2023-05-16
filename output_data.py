import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

"""
Klasa ArgScores przechowywuje dane wynikowe walidacji krzyżowej danego algorytmu. Dane te potrzebne są do metryk i testów.
"""


class ArgScores:
    """
    Konstruktor zapisujący w obiekcie podane dane.

    Parametry:
    - algname - nazwa algorytmu,
    - m_prec - średnia metryki precyzji,
    - m_rec - średnia metryki recallu,
    - std_prec - odchylenie metryki precyzji,
    - std_rec - odchylenie metryki recallu,
    - scores - lista wszyskich metryk dla wszystkich foldów (do testu T studenta).
    """

    def __init__(self, algname: str, m_prec: float, m_rec: float, std_prec: float, std_rec: float, scores: list):
        self.algname = algname
        self.mean_precision = m_prec
        self.mean_recall = m_rec
        self.std_precision = std_prec
        self.std_recall = std_rec
        self.all_scores = scores

    """
    Metoda __str__ zwracająca opis wyników metryk.

    Zwraca: message - string opisujący wyniki przebiegu algorytmu.
    """

    def __str__(self) -> str:
        message = "Metryki algorytmu " + self.algname + ":\tPrecision: %.3f" % self.mean_precision
        message += "(" + str(int(1000 * self.std_precision)) + ")\tRecall: "
        message += "%.03f" % self.mean_recall + "(" + str(int(1000 * self.std_recall)) + ")"
        return message


"""
Klasa OutputData zawiera metody przydatne do przetwarzania danych wynikowych algorytmów.
"""


class OutputData:
    algs = []  # Przechowalnia wyników algorytmów.

    """
    Metoda save_one_fold_plot służy to wygenerowania i zapisania wykresu jednego foldu z zaznaczonymi obiektami zbioru testowego
    (poprawne klasyfikacje po lewej i predykcja po prawej)

    Parametry:
    - file_name - nazwa pliku do zapisu,
    - x_text - wektor cech ciągu testowego,
    - y_test - poprawne wyniki klasyfikacji,
    - y_predicted - rzeczywiste wyniki klasyfikacji,
    - alg_name - nazwa algorytmu,
    - lines - lista punktów do wygenerowania linii (SVM).
    """

    def save_one_fold_plot(self, file_name: str, x_test, y_test, y_predicted, alg_name="", lines=[]):

        fig = plt.figure()
        # Lewy:
        plt.subplot(1, 2, 1)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='spring')
        plt.title("Test")

        # Prawy:
        plt.subplot(1, 2, 2)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predicted, cmap='spring')
        plt.title("Prediction")

        # Plotowanie linii w SVMie:
        if len(lines) == 8:
            plt.plot([lines[0], lines[1]], [lines[2], lines[3]], "violet")
            plt.plot([lines[0], lines[1]], [lines[4], lines[5]], "red")
            plt.plot([lines[0], lines[1]], [lines[6], lines[7]], "yellow")

        plt.suptitle(alg_name)
        plt.savefig(file_name)

    """
    Metoda save_folds_plot służy do wygenerowania i zapisania wykresu metryk poszczególnych foldów.

    Parametry:
    - file_name - nazwa pliku do zapisu,
    - prec_scores - lista wyników metryki precyzji,
    - recc_scores - lista wyników metryki recallu,
    - mean_prec - średnia precyzja,
    - mean_recc - średnia recall,
    - alg_name - nazwa algorytmu.
    """

    def save_folds_plot(self, file_name: str, prec_scores: list, recc_scores: list, mean_prec: float, mean_recc: float,
                        alg_name=""):

        fig = plt.figure()
        # Precision:
        plt.plot(range(len(prec_scores)), prec_scores, linestyle="", marker="o", color="b")
        plt.axhline(y=mean_prec, color='b')

        # Recall:
        plt.plot(range(len(recc_scores)), recc_scores, linestyle="", marker="o", color="g")
        plt.axhline(y=mean_recc, color='g')

        plt.grid()
        plt.legend(["Precision", "", "Recall", ""])
        plt.title(alg_name)
        plt.savefig(file_name)

    """
    Metoda save_results służy do zapisania wyników metryk do pliku.

    Parametry:
    - file_name - nazwa pliku do zapisu.
    """

    def save_results(self, file_name: str):

        alg_names = ["Name"]
        m_prec = ["Precision (mean)"]
        s_prec = ["Precision (std)"]
        m_rec = ["Recall (mean)"]
        s_rec = ["Recall (std)"]

        for algorithm in self.algs:
            alg_names.append(algorithm.algname)
            m_prec.append(algorithm.mean_precision)
            m_rec.append(algorithm.mean_recall)
            s_prec.append(algorithm.std_precision)
            s_rec.append(algorithm.std_recall)

        res = [alg_names, m_prec, m_rec, s_prec, s_rec]
        np.savetxt(file_name, res, delimiter=",", fmt="%s")

    """
    Metoda t_student służy do przeprowadzenia testów T studenta.

    Parametry:
    - alpha - poziom ufności (pomiędzy 0 a 1).
    """

    def t_student(self, alpha: float):

        clfs_scores = []
        clfs_names = []

        for alg in self.algs:
            clfs_scores.append(alg.all_scores)
            clfs_names.append(alg.algname)

        clfs_num = len(clfs_scores)

        t_stat = np.zeros((clfs_num, clfs_num))
        p = np.zeros((clfs_num, clfs_num))
        better_res = np.zeros((clfs_num, clfs_num), dtype=bool)
        is_significant = np.zeros((clfs_num, clfs_num), dtype=bool)

        # Wyliczanie macierzy testu statystycznego:
        for i in range(clfs_num):
            for j in range(clfs_num):
                t_stat[i, j], p[i, j] = ttest_ind(clfs_scores[i], clfs_scores[j])

        better_res[t_stat > 0] = True
        is_significant[p < alpha] = True
        significant_better_res = better_res * is_significant

        # Wypisanie wyników:
        test_results_str = "T student:\n" + self.add_headers_to_data(t_stat, clfs_names)
        test_results_str += "\np:\n" + self.add_headers_to_data(p, clfs_names)
        test_results_str += "\nBetter results:\n" + self.add_headers_to_data(better_res, clfs_names)
        test_results_str += "\nSignificant:\n" + self.add_headers_to_data(is_significant, clfs_names)
        test_results_str += "\nBetter significant:\n" + self.add_headers_to_data(significant_better_res, clfs_names)

        print(test_results_str)

    """
    Metoda add_headers_to_data służy do dodania nagłówków do macierzy, tak aby wyglądały one lepiej.

    Parametry:
    - matrix - macierz z danymi,
    - names - lista z nagłówkami.

    Zwraca: str(data) - string będący ładnie sformatowaną macierzą.
    """

    def add_headers_to_data(self, matrix: np.array, names: list) -> str:

        row_names = []
        for alg in names:
            row_names.append(alg + " lepszy niż:")

        data = pd.DataFrame(matrix, index=row_names, columns=names)

        return str(data)
