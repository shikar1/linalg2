class Matrix:
    def __init__(self, rows: int, columns: int, input_data=None, data=None, indexes=None,
                 ptr_to_first=None) -> None:
        # данные о строках и столбцах
        self.rows: int = rows
        self.columns: int = columns
        # три основных массива, задающих CSR-формат хранения
        self.data: list = [] if data is None else data
        self.indexes: list = [] if indexes is None else indexes
        self.ptr_to_first: list = [0] * (rows + 1) if ptr_to_first is None else ptr_to_first

        # если на вход были переданы значения, формируем три массива
        if input_data is not None:
            for row, col, value in input_data:
                self.indexes.append(col)
                self.data.append(value)
                self.ptr_to_first[row + 1] += 1

            # суммируем с предыдущими, тем самым задавая начало для каждой строки матрицы
            for i in range(1, len(self.ptr_to_first)):
                self.ptr_to_first[i] += self.ptr_to_first[i - 1]

    # метод вывода элемента матрицы
    def get(self, row: int, col: int) -> float | None:
        if self.rows < row or row < 1 or self.columns < col or col < 1:
            return None
        # получаем начало и конец нужной строки
        start_index: int = self.ptr_to_first[row - 1]
        end_index: int = self.ptr_to_first[row]
        # проходимся по нужной строке в массиве индексов, затем берём нужное в массиве значений
        if start_index != end_index:
            for i in range(start_index, end_index):
                if self.indexes[i] == col - 1:
                    return self.data[i]
        # возвращаем 0, если искомая строка полностью нулевая
        return 0

    # получение следа матрицы
    def get_trace(self) -> float | None:
        # проверяем, квадратная ли у нас матрица
        if self.rows != self.columns:
            return None
        trace = 0
        # проходимся по индексам и суммируем элементы на главной диагонали
        for index in range(1, self.rows + 1):
            trace += self.get(index, index)
        return trace

    # вывод матрицы
    def __repr__(self) -> str:
        result: list = []
        for row in range(1, self.rows + 1):
            result.append(" ".join((str(self.get(row, col)) for col in range(1, self.columns + 1))))
        return "\n".join(result)

    # сравнение двух матриц в CSR
    def __eq__(self, other):
        return other.rows == self.rows and other.columns == self.columns and other.data == self.data\
               and other.indexes == self.indexes and other.ptr_to_first == self.ptr_to_first

# транспонирование матрицы
def transpose(matrix: Matrix) -> Matrix:
    transposed_data = []

    for col in range(1, matrix.columns + 1):
        for row in range(1, matrix.rows + 1):
            val = matrix.get(row, col)
            if abs(val) > 1e-10:
                transposed_data.append((col - 1, row - 1, val))

    return Matrix(matrix.columns, matrix.rows, transposed_data)

# ввод матрицы с клавиатуры
def get_csr_matrix_from_input(is_square=False) -> Matrix:
    if not is_square:
        n, m = map(int, input(
            "Введите через пробел количество строк (n) и количество столбцов (m)\n").split())
    else:
        n = int(input("Введите число (n) – количество строк и столбцов\n"))
        m = n
    data: list = []
    print("Далее на разных строчках задайте строки матрице (элементы через пробел)")
    # формируем список кортежей, потом подаём его на вход классу в качестве input_data
    for row in range(n):
        for col, value in enumerate(map(float, input().split())):
            if value:
                data.append((row, col, value))
    return Matrix(n, m, data)


# перевод матрицы в формат двумерного списка, в данной задаче не используется
def get_classic_format(matrix: Matrix) -> list[list[float]]:
    result = []
    for row in range(matrix.rows):
        result.append([])
        for col in range(matrix.columns):
            result[row].append(matrix.get(row + 1, col + 1))
    return result

# сложение матриц
def add(a: Matrix, b: Matrix) -> Matrix | None:
    # проверяем согласованность размеров
    if a.rows != b.rows or a.columns != b.columns:
        return None

    result: list = []
    # проходимся по индексам, ненулевые результаты добавляем в result
    for i in range(a.rows):
        for j in range(b.columns):
            if elements_sum := a.get(i + 1, j + 1) + b.get(i + 1, j + 1):
                result.append((i, j, elements_sum))

    # формируем матрицу-результат в формате CSR
    return Matrix(a.rows, a.columns, result)


# умножение матрицы на скаляр
def scalar_multiply(matrix: Matrix, scalar: float) -> Matrix:
    if scalar:
        # умножаем каждый ненулевой элемент матрицы на скаляр, формируем результат в CSR с обновлённым data
        new_data: list = list(map(lambda x: x * scalar, matrix.data))
        return Matrix(matrix.rows, matrix.columns, None, new_data, matrix.indexes, matrix.ptr_to_first)
    # если скаляр нулевой, возвращаем нулевую матрицу
    return Matrix(matrix.rows, matrix.columns, [])


# умножение матриц
def multiply(a: Matrix, b: Matrix) -> Matrix | None:
    # проверяем согласованность размеров
    if a.columns != b.rows:
        return None

    result = []
    # проходимся по индексам
    for i in range(a.rows):
        for j in range(b.columns):
            # формируем новый элемент, умножая i-ую строку на j-ый столбец
            if product := sum(a.get(i + 1, k + 1) * b.get(k + 1, j + 1) for k in range(a.columns)):
                result.append((i, j, product))

    # формируем матрицу-результат в формате CSR
    return Matrix(a.rows, b.columns, result)

# при получении матрицы в формате CSR происходит перевод на стандартный get_geterminant
def get_determinant_csr(matrix: Matrix) -> float:
    standard_matrix: list[list[float]] = get_classic_format(matrix)
    return get_determinant(standard_matrix)

# подсчёт определителя для матрицы в стандартном виде
def get_determinant(matrix: list[list[float]]) -> float:
    # базовые значения определителей: 2*2, 1*1, пустое
    if not matrix:
        return 0
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant: int = 0
    # используем разложение по строке, перебираем элементы первой строки
    for c in range(len(matrix)):
        # создаём матрицу для определения минора, дальше считаем по формуле разложения
        minor: list[list[float]] = [row[:c] + row[c + 1:] for row in matrix[1:]]
        determinant += (-1) ** c * matrix[0][c] * get_determinant(minor)

    return determinant


# проверка на наличие обратной матрицы
def check_inverse(determinant: float) -> bool:
    return bool(determinant)


def save_matrix(classic_matrix: list[list[float]]) -> Matrix:
    if not classic_matrix:
        return Matrix(0, 0, [])

    rows = len(classic_matrix)
    cols = len(classic_matrix[0]) if rows > 0 else 0

    data = []
    for row in range(rows):
        for col in range(cols):
            value = classic_matrix[row][col]
            if value != 0:
                data.append((row, col, float(value)))  # Индексы с 0!

    return Matrix(rows, cols, data)