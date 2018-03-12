import numpy as np
import sympy as sp
import itertools as it
import sys


def func_module(x_):
    # return 1/2*(sp.tanh(x)+1)
    return (x_ * (1 + sp.Abs(x_)) ** (-1) + 1) / 2
    # return 1/(1+ sp.exp(-x))


def get_combs(all_indexes):
    all_index_combs = []
    for i in range(len(all_indexes) + 1):
        for c in it.combinations(all_indexes, i):
            all_index_combs.append(c)
    return all_index_combs[1:]


def get_all_indexes(row_number):
    all_indexes = []
    for i in range(row_number):
        all_indexes.append(i)
    return all_indexes


class Neuron:
    def __init__(self, eta=0.3):
        x_ = sp.symbols('x', real=True)
        dx = func_module(x_).diff(x_)
        self.dx_ = sp.lambdify(x_, dx)
        self.sum_errors_ = []
        self.eta = eta
        self.truth_table = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                                     [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1],
                                     [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0],
                                     [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                                     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0],
                                     [1, 1, 1, 1]])
        self.min_choose = self.truth_table.shape[0]
        self.truth_table = self.bool_func()
        self.all_indexes = get_all_indexes(self.truth_table.shape[0])
        self.combs = get_combs(self.all_indexes)

    def train(self, input_matrix, treshold_func=True, need_print =True):
        self.zero_error = -1
        self.epochs = 1
        self.weights_ = np.zeros(5)
        while 0 != self.zero_error and self.epochs < 100:
            self.sum_error = 0
            row_number = 0
            self.y_out_arr = []
            for row in input_matrix[:, [0, 1, 2, 3, 4]]:
                if treshold_func:
                    y_out = self.act_function_treshold(row[0:4], row_number, need_recount=True)
                else:
                    y_out = self.act_function_module(row[0:4], row_number, need_recount=True)
                self.y_out_arr.append(y_out)
                self.sum_error += abs(row[4] - y_out)
                row_number += 1
            self.epochs += 1
            self.zero_error = self.sum_error
        if self.epochs < 100:
            if need_print:
                self.outputinf(self.sum_error, input_matrix[:,4])
            return self.epochs
        else:
            return -1

    def check(self, input_matrix, treshold_func=True):
        self.sum_error = 0
        row_number = 0
        for row in input_matrix[:, [0, 1, 2, 3,4]]:
            if treshold_func:
                y_out = self.act_function_treshold(row[0:4], row_number, need_recount=False)
            else:
                y_out = self.act_function_module(row[0:4], row_number, need_recount=False)
            self.sum_error += abs(row[4] - y_out)
        self.zero_error = self.sum_error
        return self.zero_error

    def minimize(self, input_matrix, treshold_func=True):
        max_epochs = 1000
        self.min_choose = 16
        j = 0
        for i in range(len(self.combs)):
            train_assembly = np.asarray(self.combs[i])
            train_matrix = np.insert(input_matrix[train_assembly], 5, train_assembly, axis=1)
            epochs = self.train(train_matrix, treshold_func, need_print = False)
            if epochs != -1 and len(train_assembly) < 16:
                test_assembly = np.array(list(set(self.all_indexes) - set(train_assembly)))
                test_matrix = np.insert(input_matrix[test_assembly], 5, test_assembly, axis=1)
                sum_err = self.check(test_matrix, treshold_func)
                if sum_err == 0 and self.min_choose > train_matrix.shape[0] and epochs < max_epochs:
                    self.min_choose = train_matrix.shape[0]
                    max_epochs = epochs
                    min_assembly = train_matrix
                    print("Минимальный набор: \n", min_assembly)
                    print("Номер набора \n", i)
                    print("Кол-во эпох: \n", max_epochs)
                    return min_assembly

    def net_input(self, row):
        return np.dot(row, self.weights_[1:]) + self.weights_[0]

    def act_function_treshold(self, row, row_number, need_recount=True):
        yi = int(np.where(self.net_input(row) >= 0, 1, 0))
        if need_recount:
            delta = self.eta * (self.truth_table[row_number, 4] - yi)
            self.weights_[0] += delta
            self.weights_[1:] += delta * row
        return yi

    def act_function_module(self, row, row_number, need_recount=True):
        net = self.net_input(row)
        yi = float(func_module(net))
        if need_recount:
            delta = self.eta * (self.truth_table[row_number, 4] - yi) * self.dx_(net)
            self.weights_[0] += delta
            self.weights_[1:] += delta * row
        return int(np.where(yi >= 0.5, 1, 0))

    def outputinf(self, error, result_column):
        print("Входной вектор:", result_column)
        print("Выходной вектор y", self.y_out_arr)
        print("Суммарная ошибка: ", error)
        print("Вектор весов: ", np.around(self.weights_, decimals=2))
        print("Кол-во эпох обучения ", self.epochs)

    def bool_func(self):
        f_ = []
        for row in self.truth_table:
            elem = int(not ((row[0] or row[1]) and row[2] or row[3]))
            # elem = int(not(row[0] and row[1]) and row[2] and row[3])
            # elem = int(not (((row[0] or row[1]) and row[2]) or (row[2] and row[3])))
            f_.append(elem)
        table = np.insert(self.truth_table, 4, np.array(f_), axis=1)
        return table


x = Neuron()
print("Вариант 19")
print("Обучение на пороговой функции")
x.train(x.truth_table, treshold_func=True)
print("Обучение на функции с модулем")
x.train(x.truth_table, treshold_func=False)
print("Минимизация выборки для пороговой функции")
x.minimize(x.truth_table, treshold_func=True)
print("Минимизация выборки для модульной функции")
x.minimize(x.truth_table, treshold_func=False)
