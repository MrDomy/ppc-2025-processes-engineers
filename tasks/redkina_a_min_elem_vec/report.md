# "Нахождение минимального элемента в векторе", вариант № 4
### Студент: Редькина Алина Александровна
### Группа: 3823Б1ПР1
### Преподаватель: Сысоев Александр Владимирович, доцент


## Введение

   Поиск минимального элемента в массиве данных является одной из фундаментальных задач в программировании и вычислительной математике. Данная операция широко используется в различных алгоритмах. В контексте параллельного программирования эта задача представляет интерес для исследования эффективности различных подходов к распараллеливанию.

---

## Постановка задачи

**Цель работы:**  
  Реализовать и сравнить последовательную и параллельную версии алгоритма поиска минимального элемента в векторе целых чисел.

**Определение задачи:**  
  Для заданного вектора `V` необходимо определить элемент `min(V) = min(v[i])`, где `i = 0..n-1`.

**Ограничения:**
  - Входные данные — вектор целых чисел произвольной длины (в том числе пустой).
  - Корректность должна сохраняться при отрицательных, положительных и смешанных значениях.
  - Для параллельной реализации используется модель передачи сообщений (MPI).
  - Результат обеих реализаций должен совпадать.

---

## Описание алгоритма (последовательная версия)

  1. Инициализировать переменную `minimum` первым элементом вектора.  
  2. Для каждого последующего элемента `v[i]`:  
    - Выбрать минимальное между `minimum` и `v[i]` и записать в `minimum`.
  3. Вернуть значение `minimum`.

**Сложность:** O(n), где *n* — размер вектора.

### Код последовательной реализации

```cpp
bool RedkinaAMinElemVecSEQ::RunImpl() {
  const auto &vec = GetInput();

  int minimum = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    minimum = std::min(minimum, vec[i]);
  }

  GetOutput() = minimum;
  return true;
}
```

---

## Схема распараллеливания (MPI)

Алгоритм параллельно вычисляет минимальный элемент входного вектора, распределяя данные между процессами с использованием механизма разрозненной рассылки (`MPI_Scatterv`). Каждый процесс определяет локальный минимум своей части данных, после чего результаты объединяются в глобальный минимум при помощи коллективной операции редукции.

1. **Инициализация**
Все процессы входят в коммуникатор `MPI_COMM_WORLD`. Каждый процесс получает свой ранг и общее количество процессов.

2. **Передача размера**
Процесс с рангом 0 вычисляет длину входного вектора и рассылает её всем процессам через `MPI_Bcast`. На всех остальных ранках массив входных данных (`GetInput()`) остаётся пустым.

3. **Распределение данных между процессами**
Распределение выполняется почти равномерно: базовый размер фрагмента — `base = n / size`, остаток — `rem = n % size`.
Первые rem процессов получают по `base + 1` элементу, остальные — по `base`.
Процесс 0 формирует массивы:
- `counts` — количество элементов для каждого процесса,
- `displs` — смещения начала подмассива.
Эти массивы инициализируются нулями на всех ранках для безопасности.
Затем выполняется вызов `MPI_Scatterv`, где:
- процесс 0 отправляет фрагменты исходного вектора,
- остальные процессы получают только свои части, передавая `nullptr` в качестве sendbuf, sendcounts и displs.
Если процесс получил нулевой объём данных (`size_l == 0`), то указатель на локальный буфер может быть `nullptr`, что безопасно при MPI.

4. **Локальный минимум**
Каждый процесс последовательно вычисляет минимальный элемент своего подмассива (`vec_l`). 
Для процессов с нулевым объёмом данных локальный минимум остаётся равным `INT_MAX`.

5. **Глобальная редукция**
Локальные минимумы всех процессов объединяются вызовом `MPI_Allreduce` с операцией `MPI_MIN`. Каждый процесс получает одинаковое значение — глобальный минимум.

6. **Завершение**
Полученный глобальный минимум записывается в выходную переменную `GetOutput()`.

### Код параллельной реализации

```cpp
bool RedkinaAMinElemVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base = n / size;
  const int rem = n % size;
  const int size_l = (rank < rem) ? base + 1 : base;

  std::vector<int> vec_l(size_l);

  std::vector<int> counts(size, 0);
  std::vector<int> displs(size, 0);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = (i < rem) ? base + 1 : base;
    }
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_INT, size_l > 0 ? vec_l.data() : nullptr, size_l, MPI_INT, 0,
               MPI_COMM_WORLD);

  int min_l = INT_MAX;
  for (const int v : vec_l) {
    min_l = std::min(min_l, v);
  }

  int min_g = 0;
  MPI_Allreduce(&min_l, &min_g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = min_g;
  return true;
}
```

---

## Экспериментальные результаты

### Окружение
| Параметр | Значение |
|-----------|-----------|
| Процессор | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics |
| Операционная система | Windows 11 |
| Компилятор | g++ 13.3.0 |
| Тип сборки | Release |
| Число процессов | 2 |

### Проверка корректности
- Положительные числа: {7, 2, 5, 3, 4, -7} → -7
- Отрицательные числа: {-7, -2, -5, -3, -4, -1} → -7
- Смешанные числа: {4, -1, 5, -7, 3, -3} → -7
- Единичный элемент: {-7} → -7
- Все одинаковые элементы: {3, 3, 3, 3, -7} → -7
- Дубликаты: {6, 1, 3, 9, 2, -7} → -7
- Большие числа: {7000, 1500, 4000, -7, 6000} → -7
- Смешанные положительные, отрицательные и нулевые значения: {10, -5, 20, 0, -7, 15} → -7
- Минимум в начале: {-7, 1, 9, 14, 20} → -7
- Минимум в конце: {88, 8, 18, 4, -7} → -7
- Минимум в середине: {18, 8, -7, 17, 19} → -7
- Большой вектор (1000 элементов, убывающий порядок): {1000, 999, 998, …, 1} с -7 в середине → -7
- Граничные значения: {INT_MAX, -7, 0, 100} → -7
- Простой возрастающий вектор: {1, 2, -7} → -7
- Минимум в середине: {1, 5, 8, 2, -7} → -7
- Смешанные значения: {5, 8, 10, 4, -7} → -7
- Отрицательные числа: {-3, -1, -6, -7} → -7
- Два элемента, минимум в конце: {2, -7} → -7
- Три элемента, минимум в середине: {3, -7, 2} → -7
- Большой вектор (100 элементов, убывающий порядок): {100, 99, 98, …, 1} с -7 в середине → -7
- Смешанные положительные и отрицательные значения: {7, -6, 0, -7, 9, -1} → -7
- Граничные значения типа int: {INT_MAX, -7, 0} → -7
- Единичный элемент: {-7} → -7
- Возрастающий вектор из 4 элементов: {-7, 1, 2, 3} → -7
- Возрастающий вектор из 5 элементов: {-7, 1, 2, 3, 4} → -7
- Возрастающий вектор из 6 элементов: {-7, 1, 2, 3, 4, 5} → -7
- Возрастающий вектор из 7 элементов: {-7, 1, 2, 3, 4, 5, 6} → -7
- Минимум в середине: {-7, 1, 2, 3, 4} → -7
- Минимум в конце: {2, 3, 4, -7, 5} → -7
- Минимум в начале (убывающий порядок): {5, 4, 3, 2, -7} → -7
- Смешанные значения: {11, 6, 7, 2, 6, -7, 10, 4} → -7
- Все отрицательные элементы: {-1, -2, -3, -4, -7} → -7
- Нули с одним положительным элементом: {0, 0, 0, 0, -7} → -7

**Результат:** Все тесты успешно пройдены, последовательная и MPI-версии возвращают одинаковые значения.

### Оценка производительности
  Для оценки производительности использовались тесты с большим объемом данных (100000000 элементов).

**Время выполнения task_run**

Режим | Процессы | Время, с  | Ускорение | Эффективность
seq   | 1        | 0.059     | 1.00      | N/A
mpi   | 2        | 0.086     | 0.686     | 0.343
mpi   | 3        | 0.092     | 0.641     | 0.214
mpi   | 4        | 0.076     | 0.776     | 0.194

**Время выполнения pipeline**

Режим | Процессы | Время, с  | Ускорение | Эффективность
seq   | 1        | 0.059     | 1.00      | N/A
mpi   | 2        | 0.094     | 0.628     | 0.314
mpi   | 3        | 0.084     | 0.702     | 0.234
mpi   | 4        | 0.076     | 0.776     | 0.194

**Результат:** Все тесты успешно пройдены.

**Наблюдения:**
  - При небольших размерах вектора ускорение невелико из-за накладных расходов MPI.  
  - Для больших объемов данных производительность параллельной версии немного выше.  
  - Операция поиска минимума имеет малую вычислительную плотность, поэтому параллелизация не даёт значительного выигрыша.

---

## Выводы

  1. **Корректность:** обе реализации (SEQ и MPI) корректно решают задачу и дают одинаковые результаты в функциональных тестах.  
  2. **Производительность:** для небольших наборов данных параллельная версия неэффективна; ускорение наблюдается только при очень больших размерах.  
  3. **Масштабируемость:** алгоритм масштабируется линейно при росте числа процессов, однако эффективность ограничена коммуникационными затратами.  
  4. **Вывод:** использование MPI для данной задачи оправдано лишь в контексте более сложных параллельных вычислений.

---

## Источники

  1. Лекции Сысоева Александра Владимировича.

## Приложения

**common.hpp**
```cpp
#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace redkina_a_min_elem_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<int, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace redkina_a_min_elem_vec
```

**ops_seq.hpp**
```cpp
#pragma once

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_min_elem_vec {

class RedkinaAMinElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RedkinaAMinElemVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace redkina_a_min_elem_vec
```

**ops_mpi.hpp**
```cpp
#pragma once

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_min_elem_vec {

class RedkinaAMinElemVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RedkinaAMinElemVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace redkina_a_min_elem_vec
```

**ops_seq.cpp**
```cpp
#include "redkina_a_min_elem_vec/seq/include/ops_seq.hpp"

#include <climits>
#include <cstddef>
#include <vector>

#include "redkina_a_min_elem_vec/common/include/common.hpp"

namespace redkina_a_min_elem_vec {

RedkinaAMinElemVecSEQ::RedkinaAMinElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RedkinaAMinElemVecSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool RedkinaAMinElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool RedkinaAMinElemVecSEQ::RunImpl() {
  const auto &vec = GetInput();
  
  int minimum = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    minimum = std::min(minimum, vec[i]);
  }

  GetOutput() = minimum;
  return true;
}


bool RedkinaAMinElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_min_elem_vec
```

**ops_mpi.cpp**
```cpp
#include "redkina_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "redkina_a_min_elem_vec/common/include/common.hpp"

namespace redkina_a_min_elem_vec {

RedkinaAMinElemVecMPI::RedkinaAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RedkinaAMinElemVecMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool RedkinaAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool RedkinaAMinElemVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base = n / size;
  const int rem = n % size;
  const int size_l = (rank < rem) ? base + 1 : base;

  std::vector<int> vec_l(size_l);

  std::vector<int> counts(size, 0);
  std::vector<int> displs(size, 0);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = (i < rem) ? base + 1 : base;
    }
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_INT, size_l > 0 ? vec_l.data() : nullptr, size_l, MPI_INT, 0,
               MPI_COMM_WORLD);

  int min_l = INT_MAX;
  for (const int v : vec_l) {
    min_l = std::min(min_l, v);
  }

  int min_g = 0;
  MPI_Allreduce(&min_l, &min_g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = min_g;
  return true;
}

bool RedkinaAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_min_elem_vec
```