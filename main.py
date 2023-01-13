# Мономы, Гивенс, Глобальная СЛАУ, N = 5

import numpy as np
from matplotlib import pyplot as plt
import time
from prettytable import PrettyTable
import pandas as pd


def original_function(x):
    return x ** 2 * (1 - x) ** 2 * np.exp(x)


def function(x):
    return (x ** 4 + 14 * x ** 3 + 49 * x ** 2 + 32 * x - 12) * np.exp(x)


def boundary_function_1(x):
    return 0


def boundary_function_2(x):
    return 0


def polynomial(degree: int, x: float, ) -> float:
    return x ** degree


def first_derivative_polynomial(degree: int, x: float) -> float:
    return degree * x ** (degree - 1)


def second_derivative_polynomial(degree: int, x: float) -> float:
    return degree * (degree - 1) * x ** (degree - 2)


def third_derivative_polynomial(degree: int, x: float) -> float:
    return degree * (degree - 1) * (degree - 2) * x ** (degree - 3)


def fourth_derivative_polynomial(degree: int, x: float) -> float:
    return degree * (degree - 1) * (degree - 2) * (degree - 3) * x ** (degree - 4)


def approx_function(x, matrix, cells, degree):
    h = (cells[1] - cells[0]) / 2

    if x == cells[0]:
        cell_id = 0
    elif x == cells[-1]:
        cell_id = len(cells) - 2
    else:
        cell_id = int(x // (2 * h))

    cell_center = (cells[cell_id] + cells[cell_id + 1]) / 2
    y = (x - cell_center) / h

    return sum([matrix[cell_id, i] * polynomial(i, y) for i in range(degree + 1)])


def givens(matrix: np.ndarray, vector: np.ndarray):
    m, n = matrix.shape
    R = matrix.copy()
    b = vector.copy()

    for col in range(n):
        # выбор главного элемента
        index_max = np.argmax(np.absolute(R[col:, col])) + col
        R[[index_max, col]] = R[[col, index_max]]
        b[[index_max, col]] = b[[col, index_max]]

        for row in range(col + 1, m):
            if R[row, col] != 0:
                r = np.sqrt(R[col, col] ** 2 + R[row, col] ** 2)
                c, s = R[col, col] / r, -R[row, col] / r

                R[col], R[row] = R[col]*c + R[row]*(-s), R[col]*s + R[row]*c
                b[col], b[row] = b[col] * c + b[row] * (-s), b[col] * s + b[row] * c

    return R[:n], b[:n]


def householder(matrix: np.ndarray, vector: np.ndarray):
    A = matrix.copy()
    b = vector.copy()
    m, n = A.shape
    for j in range(n):
        if np.any(A[j + 1:, j]):
            e = np.zeros(m - j)
            e[0] = 1
            if A[j, j] >= 0:
                u = A[j:, j] + np.linalg.norm(A[j:, j]) * e
            else:
                u = A[j:, j] - np.linalg.norm(A[j:, j]) * e
            u /= np.linalg.norm(u)
            A[j:, j:] -= np.outer(2 * u, u @ A[j:, j:])
            b[j:] -= 2 * u * (u @ b[j:])
    return A, b


def gauss_reverse(a, b):
    n = a.shape[1]
    x = np.zeros(n)
    for i in range(len(x) - 1, -1, -1):
        x[i] = (b[i] - sum(x * R for x, R in zip(x[(i + 1):], a[i, (i + 1):]))) / a[i][i]
    return x


def QR(matrix, vector):
    a, b = givens(matrix, vector)
    # a, b = householder(matrix, vector)
    x = gauss_reverse(a, b)
    return x


def set_Global_Matrix(N_degree, N_cells, GlobalNodes):
    N_colloc = N_degree + 1  # количество точек коллокации
    N_eq = N_colloc + 4
    X0 = 0.0
    X1 = 1.0
    y_ = np.linspace(-1., 1., N_degree + 1 + 2, endpoint=True)
    GlobalMatrix = np.zeros((N_eq * N_cells, (N_degree + 1) * N_cells))
    GlobalVector = np.zeros(N_eq * N_cells)

    step = GlobalNodes[1] - GlobalNodes[0]

    # Уравнения в левой ячейке
    # граничные условия
    GlobalMatrix[0, 0:N_degree + 1] = [polynomial(j, y_[0]) for j in range(N_degree + 1)]
    GlobalMatrix[1, 0:N_degree + 1] = [first_derivative_polynomial(j, y_[0]) for j in range(N_degree + 1)]
    GlobalVector[0] = boundary_function_1(X0)
    GlobalVector[1] = (step / 2) * boundary_function_2(X0)

    # условия согласования
    GlobalMatrix[N_eq - 2, 0:N_degree + 1] = [polynomial(j, y_[-1]) + first_derivative_polynomial(j, y_[-1]) for j in
                                              range(N_degree + 1)]
    GlobalMatrix[N_eq - 1, 0:N_degree + 1] = [
        second_derivative_polynomial(j, y_[-1]) + third_derivative_polynomial(j, y_[-1]) for j in
        range(N_degree + 1)]
    GlobalVector[N_eq - 2] = 0
    GlobalVector[N_eq - 1] = 0
    GlobalMatrix[N_eq - 2, 1 * (N_degree + 1): 2 * (N_degree + 1)] = [
        -(polynomial(j, y_[0]) + first_derivative_polynomial(j, y_[0])) for j in range(N_degree + 1)]
    GlobalMatrix[N_eq - 1, 1 * (N_degree + 1): 2 * (N_degree + 1)] = [
        - (second_derivative_polynomial(j, y_[0]) + third_derivative_polynomial(j, y_[0])) for j in range(N_degree + 1)]

    # условия коллокации
    xc = (GlobalNodes[0] + GlobalNodes[1]) / 2
    x = np.array([y_[i] * step / 2 + xc for i in range(1, N_colloc + 1)])
    for i in range(2, N_colloc + 2):
        GlobalMatrix[i][4:N_degree + 1] = [fourth_derivative_polynomial(j, y_[i - 1]) for j in range(4, N_degree + 1)]
        # правая часть в точках коллокации
    GlobalVector[2: N_colloc + 2] = (step / 2) ** 4 * function(x)

    # Уравнения в правой ячейке
    index = (N_cells - 1) * N_eq
    index_row = (N_cells - 1) * (N_degree + 1)
    # условия согласования
    GlobalMatrix[index, index_row + 0: index_row + N_degree + 1] = [
        polynomial(j, y_[0]) - first_derivative_polynomial(j, y_[0]) for
        j in range(N_degree + 1)]
    GlobalMatrix[index + 1, index_row + 0: index_row + N_degree + 1] = [
        second_derivative_polynomial(j, y_[0]) - third_derivative_polynomial(j, y_[0]) for j in range(N_degree + 1)]
    GlobalVector[index + 0] = 0
    GlobalVector[index + 1] = 0
    GlobalMatrix[index, index_row - 1 * (N_degree + 1): index_row - 0 * (N_degree + 1)] = [
        -(polynomial(j, y_[-1]) - first_derivative_polynomial(j, y_[-1])) for j in range(N_degree + 1)]
    GlobalMatrix[index + 1, index_row - 1 * (N_degree + 1): index_row - 0 * (N_degree + 1)] = [
        - (second_derivative_polynomial(j, y_[-1]) - third_derivative_polynomial(j, y_[-1])) for j in
        range(N_degree + 1)]

    # краевые условия
    GlobalMatrix[index + N_eq - 2, index_row + 0: index_row + N_degree + 1] = [polynomial(j, y_[-1]) for j in
                                                                               range(N_degree + 1)]
    GlobalMatrix[index + N_eq - 1, index_row + 0: index_row + N_degree + 1] = [first_derivative_polynomial(j, y_[-1])
                                                                               for j in
                                                                               range(N_degree + 1)]
    GlobalVector[index + N_eq - 2] = boundary_function_1(X1)
    GlobalVector[index + N_eq - 1] = (step / 2) * boundary_function_2(X1)

    step = GlobalNodes[1] - GlobalNodes[0]
    # условия коллокации
    xc = (GlobalNodes[N_cells - 1] + GlobalNodes[N_cells]) / 2
    x = np.array([y_[i] * step / 2 + xc for i in range(1, N_colloc + 1)])
    for i in range(2, N_colloc + 2):
        GlobalMatrix[index + i][index_row + 4:index_row + N_degree + 1] = [fourth_derivative_polynomial(j, y_[i - 1])
                                                                           for j in
                                                                           range(4, N_degree + 1)]
        # правая часть в точках коллокации
    GlobalVector[index + 2:index + N_colloc + 2] = (step / 2) ** 4 * function(x)

    # Уравнения во внутренних ячейках
    for i_cell in range(1, N_cells - 1):
        index = i_cell * N_eq
        index_row = i_cell * (N_degree + 1)

        # условия согласования
        GlobalMatrix[index, index_row + 0: index_row + N_degree + 1] = [
            polynomial(j, y_[0]) - first_derivative_polynomial(j, y_[0])
            for j in range(N_degree + 1)]
        GlobalMatrix[index + 1, index_row + 0: index_row + N_degree + 1] = [
            second_derivative_polynomial(j, y_[0]) - third_derivative_polynomial(j, y_[0]) for j in range(N_degree + 1)]
        GlobalVector[index + 0] = 0
        GlobalVector[index + 1] = 0
        GlobalMatrix[index, index_row - 1 * (N_degree + 1): index_row - 0 * (N_degree + 1)] = [
            -(polynomial(j, y_[-1]) - first_derivative_polynomial(j, y_[-1])) for j in range(N_degree + 1)]
        GlobalMatrix[index + 1, index_row - 1 * (N_degree + 1): index_row - 0 * (N_degree + 1)] = [
            - (second_derivative_polynomial(j, y_[-1]) - third_derivative_polynomial(j, y_[-1])) for j in
            range(N_degree + 1)]

        # условия согласования
        GlobalMatrix[index + N_eq - 2, index_row + 0: index_row + N_degree + 1] = [
            polynomial(j, y_[-1]) + first_derivative_polynomial(j, y_[-1]) for j in range(N_degree + 1)]
        GlobalMatrix[index + N_eq - 1, index_row + 0: index_row + N_degree + 1] = [
            second_derivative_polynomial(j, y_[-1]) + third_derivative_polynomial(j, y_[-1]) for j in
            range(N_degree + 1)]
        GlobalVector[index + N_eq - 2] = 0
        GlobalVector[index + N_eq - 1] = 0
        GlobalMatrix[index + N_eq - 2, index_row + 1 * (N_degree + 1): index_row + 2 * (N_degree + 1)] = [
            -(polynomial(j, y_[0]) + first_derivative_polynomial(j, y_[0])) for j in range(N_degree + 1)]
        GlobalMatrix[index + N_eq - 1, index_row + 1 * (N_degree + 1): index_row + 2 * (N_degree + 1)] = [
            - (second_derivative_polynomial(j, y_[0]) + third_derivative_polynomial(j, y_[0])) for j in
            range(N_degree + 1)]

        step = GlobalNodes[1] - GlobalNodes[0]
        # условия коллокаций
        # правая часть в точках коллокации
        xc = (GlobalNodes[i_cell] + GlobalNodes[i_cell + 1]) / 2
        x = np.array([y_[i] * step / 2 + xc for i in range(1, N_colloc + 1)])
        for i in range(2, N_colloc + 2):
            GlobalMatrix[index + i][index_row + 4:index_row + N_degree + 1] = [
                fourth_derivative_polynomial(j, y_[i - 1]) for j in
                range(4, N_degree + 1)]
        GlobalVector[index + 2:index + N_colloc + 2] = (step / 2) ** 4 * function(x)
    return GlobalMatrix, GlobalVector


def plot_solution(matrix, cells, degree, left, right):
    # достаточно мелкая сетка, для сравнения решений и отрисовки
    X_cor = np.arange(left, right, 0.01)
    # значение численного решения сетке
    u_ = np.array([approx_function(x=i, matrix=matrix, cells=cells, degree=degree) for i in X_cor])
    # значение точного решения на сетке
    u_ex = np.array([original_function(i) for i in X_cor])
    # рисуем численное решение
    fig, ax = plt.subplots(figsize=(8, 7))
    # plt.yscale('log')
    ax.plot(X_cor, u_, 'r-', label='num')
    # рисуем точное решение
    ax.plot(X_cor, u_ex, 'b--', label='exact')
    ax.legend()
    plt.show()


def solve(cells_number: int, left: int, right: int , degree: int):
    cells = np.linspace(left, right, cells_number + 1)

    matrix, vector = set_Global_Matrix(degree, cells_number, cells)
    solution = QR(matrix, vector)

    # Q, R = np.linalg.qr(matrix)
    # solution = np.dot(np.linalg.inv(R), np.dot(Q.T, vector))

    coef_matrix = np.zeros((cells_number, degree + 1))
    for i in range(cells_number):
        coef_matrix[i] = solution[i * (degree + 1): (i + 1) * (degree + 1)]

    cond_matrix = np.linalg.cond(matrix)

    return matrix, vector, cond_matrix, coef_matrix, cells


def main():
    left = 0
    right = 1
    polynomial_degree: int = 5
    solution_list = []
    error = []
    abs_error = []
    points = np.linspace(left, right, 300)
    original = np.array([original_function(i) for i in points])
    time_list = []
    cond_list = []

    i = 5
    while i < 161:
        print(f'K = {i}')

        start = time.time()
        matrix, vector, cond_matrix, coef_matrix, cells = solve(cells_number=i, left=left, right=right,
                                                                degree=polynomial_degree)
        print(f"Time in seconds: {time.time() - start:.2}\n")
        time_list.append(time.time() - start)

        solution_list.append([matrix, vector, cond_matrix, coef_matrix, cells])
        cond_list.append(cond_matrix)

        approx_solution = np.array([approx_function(x, coef_matrix, cells, polynomial_degree) for x in points])

        error.append(np.max(abs(approx_solution - original)) / np.max(abs(original)))
        abs_error.append(np.max(abs(approx_solution - original)))

        if i == 5:
            plot_solution(coef_matrix, cells, polynomial_degree, left, right)
            plt.show()

        i *= 2

    rate = [0]
    rate_abs = [0]
    for i in range(len(error) - 1):
        rate.append(np.log2(error[i] / error[i + 1]))
        rate_abs.append(np.log2(abs_error[i] / abs_error[i + 1]))

    k_list = [5 * 2 ** i for i in range(len(error))]
    table = PrettyTable()
    table.field_names = ["Mesh", "Time", "Error_absolute", "Order abs", "Error_relative", "Order", "Cond_Global_matrix"]
    for i in range(len(error)):
        table.add_row(
            [5 * 2 ** i, "{:2.2f}".format(time_list[i]), "{:2.2e}".format(abs_error[i]), "{:2.2e}".format(rate_abs[i]), "{:2.2e}".format(error[i]),
             "{:2.2e}".format(rate[i]), "{:2.2e}".format(solution_list[i][2])])
        # table.add_row(5 * 2 ** i, f'{time_list[i]:.2}', f'{abs_error[i]:.2}', f'{rate_abs[i]:.2}', f'{error[i]:.2}',
                      # f'{rate[i]:.2}', f'{solution_list[i][2]:.2}')
    # print(table)
    plt.plot(k_list, error, 'r', label='error')
    # plt.plot(abs_error, label='abs error')
    plt.title('Error')
    plt.show()

    df = pd.DataFrame(({'Mesh': k_list,
                        'Time': time_list,
                        'Absolute error': abs_error,
                        'Absolute order': rate_abs,
                        'Relative error': error,
                        'Order': rate,
                        'Global matrix cond': cond_list}))
    print(f'{df.to_string()}\n')
    writer = pd.ExcelWriter(f'results.xlsx')
    df.to_excel(writer)
    writer.save()
    writer.close()


if __name__ == '__main__':
    main()
