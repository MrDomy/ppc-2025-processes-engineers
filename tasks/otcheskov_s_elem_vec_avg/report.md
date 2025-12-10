# Вычисление среднего значения элементов вектора

- Студент: Отческов Семён Андреевич, группа 3823Б1ПР1
- Технологии: SEQ | MPI
- Вариант: 2

## 1. Введение
- Вычисление среднего арифметического элементов вектора является хоть и простой но важной задачей в анализе данных. При работе с большими объемами данных (несколько миллионов или даже миллиардов элементов) скорость вычислений простого алгоритма может оказаться недостаточной даже при использовании оптимизаций компилятора.

- В данной работе представлены два алгоритма вычисления среднего значения элементов вектора: последовательная (базовая) реализация и параллельная реализация с использованием технологии MPI (Message Passing Interface).

- **Цель работы:** сравнение производительности алгоритмов и анализ эффективности распараллеливания вычислительной задачи.

## 2. Постановка задачи
**Формальная постановка:**
- Для вектора V длины N вычислить среднее арифметическое: $avg=\frac{\sum_{i=0}^{N-1}V[i]}{N}$

**Входные данные:**
- Вектор целых чисел произвольной длины N.

**Выходные данные:**
- Вещественное число — среднее значение элементов вектора V.

**Ограничения:**
- Вектор должен содержать хотя бы один элемент.
- Вектор содержит целые числа.

## 3. Описание алгоритма (последовательного)

### 3.1. Этапы выполнения задачи
**1. Валидация данных (`ValidationImpl`):**
- Проверка на пустоту вектора.
- Проверка корректности начального состояния выходного значения.

**2. Предобработка данных (`PreProcessingImpl`):**
- Задача не требует предобработки, поэтому данный этап пропускается.

**3. Вычисления (`RunImpl`):**
- Подсчёт суммы элементов вектора.
- Деление полученой суммы на число элементов в векторе.

**4. Постобработка данных (`PostProcessingImpl`):**
- Аналогична предобработке.

### 3.2. Сложность алгоритма:
- Временная сложность: `O(N)` — однократный проход по каждому элементу вектора.
- Пространственная сложность: `O(N)` — хранение вектора произвольной длины N.

## 4. Схема распараллеливания алгоритма с помощью MPI

### 4.1. Распределение данных
- Блочное распределение:
  - Ранг 0 распределяет данные между всеми процессами с помощью `MPI_Scatterv`
  - Размер базового блока (`batch_size`) элементов определяется как `число элементов вектора / число процессов`.
  - Размер локальной части для процесса (`proc_size`) равно размеру базового блока.
  - Если есть остаток от деления числа элементов, то он добавляется по одному элементу в `proc_size` первым процессам.
  - Каждый процесс получает только свою часть данных в локальный буфер `local_data`

### 4.2. Топология коммуникаций
- Линейная топология (все процессы связаны через `MPI_COMM_WORLD`).
- Роль процессов:
  - **Ранг 0**: координатор — вычисляет распределение, рассылает метаданные, распределяет данные
  - **Все ранги**: рабочие — получают свою часть данных, вычисляют локальную сумму

### 4.3. Паттерны коммуникации
- В MPI реализации используются коллективные блокирующие функции.
- Функция `MPI_Bcast`:
  - Применяется в валидации, чтобы каждый процесс прошёл проверку аналогично процессу ранга 0.
  - Применяется для рассылки массивов числа локальных элементов (`counts`) и смещений в исходном массиве (`displacements`), чтобы использовать `MPI_Scatterv`
- Функция `MPI_Scatterv`:
  - Рассылает части исходного вектора процессам в локальный буффер `local_data`, основываясь на числе элементов конкретного процесса в `counts[proc_rank]` и смещении `displacements[proc_rank]`.
- Функция `MPI_Allreduce`:
  1. Производит операцию суммирования (`MPI_SUM`) локальных сумм со всех процессов.
  2. Итоговую сумму и рассылкает её всем процессам

### 4.4. Распределение вычислений
1. **Инициализация MPI** — получение ранга и числа процессов.
2. **Определение распределения** — вычисление размеров блоков и смещений.
3. **Распределение данных** — рассылка частей вектора на процессы.
4. **Локальные вычисления**— суммирование элементов локального сегмента.
5. **Глобальная редукция** — объединение локальных сумм.
6. **Финальное вычисление** — расчёт среднего значения.

## 5. Особенности реализаций

### 5.1. Структура кода
Реализации классов и методов на языке С++ указаны в [Приложении](#10-приложение).

#### 5.1.1. Файлы
- `./common/include/common.hpp` — общие определения типов данных ([см. Приложение №1](#101-приложение-1--общие-определения)).
- `./seq/include/ops_seq.hpp` — определение класса последовательной версии задачи ([см. Приложение №2.1](#1021-заголовочный-файл)).
- `./seq/include/ops_mpi.hpp` — определение класса параллельной версии задачи ([см. Приложение №2.2](#1021-файл-реализации)).
- `./seq/src/ops_seq.cpp` — реализация последовательной версии задачи ([см. Приложение №3.1](#1031-заголовочный-файл)).
- `./seq/src/ops_mpi.cpp` — реализация параллельной версии задачи ([см. Приложение №3.2](#1032-файл-реализации)).
- `./tests/functional/main.cpp` — реализация функциональных и валидационных тестов ([см. Приложение №4](#104-приложение-4--функциональные-и-валидационные-тесты)).
- `./tests/performance/main.cpp` — реализация производительных тестов ([см. Приложение №5](#105-приложение-5--проиводительные-тесты)).

#### 5.1.2. Ключевые классы
- `OtcheskovSElemVecAvgSEQ` — последовательная версия.
- `OtcheskovSElemVecAvgMPI` — параллельная версия.
- `OtcheskovSElemVecAvgFuncTests` — функциональные тесты.
- `OtcheskovSElemVecAvgFuncTestsValidation` — валидационные тесты.
- `OtcheskovSElemVecAvgPerfTests` — производительные тесты.

#### 5.1.3. Основные методы
- `ValidationImpl` — валидация входных данных и состояния выходных данных.
- `PreProcessingImpl` — препроцессинг, не используется.
- `RunImpl` — основная логика вычислений.
- `PostProcessingImpl` — постпроцессинг, не используется.

### 5.2. Реализация последовательной версии

#### 5.2.1. ValidationImpl
- Проверка на пустоту вектора `GetInput` и начального значения выходного параметра `GetOutput`.
- Выходное значение `GetOutput` инициализируется значением `NAN` для однозначной идентификации начального состояния.
- `NAN` - макрос, который раскрывается в константное выражение типа float, представляющее значение тихого NaN (QNaN).
- Проверка, что `GetOutput` не был изменён до начала задачи, выполняется функцией `std::isnan` из библиотеки `cmath`.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgMPI::ValidationImpll() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}
```

#### 5.2.2. PreProcessingImpl
В препроцессинге нет необходимости, поэтому данный этап пропускается.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgSEQ::PreProcessingImpl() {
  return true;
}
```

#### 5.2.3. RunImpl
- Дополнительная проверка на пустоту входного вектора.
- Сумма элементов массива вычисляется STL-функцией `std::reduce`, расположенной в библиотеке `<numeric>`.
- Функция `std::reduce` принимает пару итераторов, определяющих диапазон элементов.
  - Она складывает числа в произвольном порядке, предоставляя компилятору больше свободы для оптимизации.
- Присвоение `GetOutput` среднего арифметрического, поделив полученную сумму на размер входного вектора.
- Проверяем, что `GetOutput` был изменён в результате предыдущей операции с помощью `!std::isnan`.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  // проверка на пустоту вектора
  if (GetInput().empty()) {
    return false;
  }

  // вычисляем среднее арифметическое элементов вектора
  int64_t sum = std::reduce(GetInput().begin(), GetInput().end(), int64_t{0});
  GetOutput() = static_cast<double>(sum) / static_cast<double>(GetInput().size());
  return !std::isnan(GetOutput());
}
```

#### 5.2.4. PostProcessingImpl
В постпроцессинге нет необходимости, поэтому данный этап пропускается.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return true;
}
```

### 5.3. Реализация параллельной версии
**Этапы:** `PreProcessingImpl`, `PostProcessingImpl` аналогичны последовательной версии.

#### 5.3.1. ValidationImpl
- Выполняется проверка на процессе ранга 0, аналогична реализации в последовательной версии.
- Результат проверки рассылкается всем процессам через `MPI_Bcast`
- Таким образом, проверка едина для всех процессов.

#### 5.3.1. RunImpl
- Распределение данных, как описано в разделе [4.1. Распределение данных](#41-распределение-данных).
- Вычисление локальных сумм.
- С помощью `MPI_Allreduce` выполняется:
  - Сбор локальных сумм из процессов.
  - Сложение полученных сумм.
  - Передача результата всем процессам.

**Реализация на C++:**
- Представлена в разделе [Приложение №3 — Параллельная версия решения задачи
](#1032-файл-реализации)

### 5.4. Использование памяти
- Последовательная версия: `O(N)` — входной вектор произвольной длины `N`.
- Параллельная версия: 
  - Ранг 0: `O(N)` — хранит исходный массив + `O(N/P)` — локальная часть данных исходного массива, распределённого `P` процессам.
  - Остальные ранги: `O(N/P)` — локальная часть данных исходного массива, распределённого `P` процессам.

### 5.5. Допущения и крайние случаи
- Все процессы запускаются в рамках одного MPI-коммуникатора.
- Ранг 0 является корневым процессом для распределения данных и валидации.


## 6. Тестовые инфраструктуры
### 6.1. Windows
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 12400F (6 cores, 12 threads, 2500 MHz) |
| RAM        | 32 GB DDR4 (3200 MHz)                                |
| OS         | Windows 10 (10.0.19045)                              |
| Компилятор | MSVC 19.42.34435, Release Build                      |

### 6.2 WSL
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 12400F (6 cores, 12 threads, 2500 MHz) |
| RAM        | 16 GB DDR4 (3200 MHz)                                |
| OS         | Ubuntu 24.04.3 LTS on Windows 10 x86_64              |
| Компилятор | GCC 13.3.0, Release Build                            |

### 6.3. Общие настройки
- **Переменные окружения:** PPC_NUM_PROC = 2, 4.
- **Данные:** элементы вектора хранятся в `test_vec*.txt` файлах в директории `./data`.


## 7. Результаты и обсуждение

### 7.1. Корректность

Корректность задачи для обоих версий проверена с помощью набора параметризированных тестов Google Test.

#### 7.1.1. Функциональные тесты
- Вектор с положительными числами:
  - Данные: `./data/test_vec1.txt`.
  - Ожидаемое значение: `50.5`.

- Вектор со смешанными положительными и отрицательными числами:
  - Данные: `./data/test_vec2.txt`.
  - Ожидаемое значение: `14.5`.

- Вектор с одним элементом:
  - Данные: `./data/test_vec_one_elem.txt`.
  - Ожидаемое значение: `5.0`.

- Вектор с дробным средним значением элементов:
  - Данные: `./data/test_vec_fraction.txt`.
  - Ожидаемое значение: `4.0/3.0`.

- Вектор с большого размера:
  - Данные: `./data/test_vec_one_million_elems.txt`.
  - Ожидаемое значение: `-2.60988`.

- Вектор с чередующимися противоположными элементами:
  - Данные: `./data/test_vec_alternating_elems.txt`.
  - Ожидаемое значение: `0.0`.

- Вектор с нулевыми элементами:
  - Данные: `./data/test_vec_zeros_elems.txt`.
  - Ожидаемое значение: `0.0`.

#### 7.1.2. Валидационные тесты
- Обработка пустого вектора:
  - Данные: пустой вектор.
  - Цель: проверка корректной обработки некорректных входных данных.

- Проверка сброса выходного значения перед выполнением:
  - Данные: любой вектор.
  - Цель: проверка провала валидации при изменении состояния переменных до запуска задачи.

#### 7.1.3. Механизм проверки
- Все тесты выполняются как для последовательной (SEQ), так и для параллельной (MPI) версии.
- Данные загружаются из файлов через абсолютные пути, получаемые функцией `GetAbsoluteTaskPath()`.
- Выходное и ожидаемое вещественные значения сравниваются с учётом машинной точности:
    ```с++
    std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon()
    ```
    где:
    - `std:fabs()` — вычисляет абсолютное значение разности ожидаемого и полученного значений.
    - `std::numeric_limits<double>::epsilon()` — возвращает машинный эпсилон для типа double.

- Для некорректных сценариев проверяется провал валидации (`ValidationImpl()`).

### 7.2. Производительные тесты

#### 7.2.1. Методология тестирования
- **Данные:** вектор из 1 миллиона элементов дублируется до 512 миллионов. Данные для вектора берутся из файла `./data/test_vec_one_million_elems.txt`.
- **Режимы:**
  - **pipeline** — запуск и измерение времени всех этапов алгоритма (`Validation -> PreProcessing -> Run -> PostProcessing`).
  - **task_run** — запуск всех этапов алгоритма, но измеряется время только на этапе `Run`.
- **Производительность** мерилась только в режиме `task_run`
- **Метрики:** число процессов, абсолютное время выполнения task_run, ускорение, эффективность.

#### 7.2.2. Результаты тестирования на 512 миллионов элементов

**Windows:**
| Режим | Процессов | Время, s | Ускорение | Эффективность |
| ----- | --------- | -------- | --------- | ------------- |
| seq   | 1         | 0.089805 | 1.0000    | N/A           |
| mpi   | 2         | 0.528954 | 0.1698    | 8.5%          |
| mpi   | 4         | 0.427870 | 0.2081    | 5%            |

**WSL:**
| Режим | Процессов | Время, s | Ускорение | Эффективность |
| ----- | --------- | -------- | --------- | ------------- |
| seq   | 1         | 0.613402 | 1.0000    | N/A           |
| mpi   | 2         | 2.407203 | 0.2548    | 12.7%         |
| mpi   | 4         | 1.401476 | 0.4377    | 10.9%         |

**\*GitHub:**
| Режим | Процессов | Время, s | Ускорение | Эффективность |
| ----- | --------- | -------- | --------- | ------------- |
| seq   | 1         | 0.097631 | 1.0000    | N/A           |
| mpi   | 2         | 0.261573 | 0.3733    | 18.7%         |

*\*Результаты собирались на локальном форке из Github Actions*

### 7.3. Анализ результатов

- **Аномально низкая производительность:**
  - На объёме данных в 512 миллионов элементов наблюдается значительное замедление MPI-версии по сравнению с последовательной реализацией.
  - На всех тестовых платформах ускорение составляет менее 1 (0.17-0.44).
  - Эффективность падает до  значений (5-19%)

- **Сравнение инфраструктур:**
  - Наиболее стабильной и быстрой из тестовых инфраструктур является машина на GitHub.
  - WSL также показывает стабильный результат, но является самой медленной инфраструктурой, что, вероятно, связано архитектурными особенностями технологии.

- **Ограничения масштабируемости:**
  - Из-за затрат на коммуникацию между процессами эффективность значительно снижается.


## 8. Заключения

### 8.1. Достигнутые результаты:
1. **Схема распределения данных** — блочная схема распределения, которая равномерно распределяет данные процессам.
2. **Эффективное использования памяти** — текущая реализация требует хранения части исходного вектора в каждом процессе и целого вектора в процессе ранга 0.
3. **Корректность результатов** — полное соответствие последовательной и параллельной версий.
4. **Модульность и тестируемость** — код структурирован и покрыт функциональными и валидационными тестами.

### 8.2. Выявленные проблемы и возможные улучшения:
1. **Улучшение масштабируемости** — текущая реализация демонстрирует аномально низкую эффективность (5-19%) при работе с большими объёмами данных.
2. **Оптимизация коммуникационных операций** — необходимо исследовать причины значительных накладных расходов на этапе распределения данных входного вектора процессам.
3. **Рассмотрение более эффективных реализаций** — рассмотреть реализации, где минимизированы накладные расходны на передачу данных между процессами.

В рамках данной работы успешно решена задача вычисления среднего арифметического элементов вектора, реализованы два решения: последовательное и параллельное с использованием MPI.

## 9. Источники
1. std::reduce // cppreference.com URL: https://en.cppreference.com/w/cpp/algorithm/reduce.html (дата обращения: 12.11.2025).
2. Документация по курсу «Параллельное программирование» // Parallel Programming Course URL: https://learning-process.github.io/parallel_programming_course/ru/index.html (дата обращения: 25.10.2025).
3. The big STL Algorithms tutorial: reduce operations // Sandor Dargo's Blog URL: https://www.sandordargo.com/blog/2021/10/20/stl-alogorithms-tutorial-part-27-reduce-operations (дата обращения: 12.11.2025).
4. "Коллективные и парные взаимодействия" / Сысоев А. В // Лекции по дисциплине «Параллельное программирование для кластерных систем».

## 10. Приложение

### 10.1. Приложение №1 — Общие определения
Файл: `./common/include/common.hpp`.
```cpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

using InType = std::vector<int>;
using OutType = double;
using TestType = std::tuple<std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.2. Приложение №2 — Последовательная версия решения задачи 
#### 10.2.1. Заголовочный файл:
Файл: `./seq/ops_seq.hpp`.
```cpp
#pragma once

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSElemVecAvgSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace otcheskov_s_elem_vec_avg
```

#### 10.2.1. Файл реализации:
Файл: `./seq/ops_seq.cpp`.
```cpp
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgSEQ::OtcheskovSElemVecAvgSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgSEQ::ValidationImpl() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgSEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int sum = std::reduce(GetInput().begin(), GetInput().end(), uint64_t{0});
  GetOutput() = sum / static_cast<double>(GetInput().size());
  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.3. Приложение №3 — Параллельная версия решения задачи

#### 10.3.1. Заголовочный файл
Файл: `./mpi/ops_mpi.hpp`.
```cpp
namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSElemVecAvgMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int proc_rank_{};
  int proc_num_{};
};

}  // namespace otcheskov_s_elem_vec_avg
```

#### 10.3.2. Файл реализации
Файл: `./mpi/ops_mpi.cpp`.
```cpp
namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgMPI::OtcheskovSElemVecAvgMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);
  SetTypeOfTask(GetStaticTypeOfTask());
  if (proc_rank_ == 0) {
    GetInput() = in;
  }
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgMPI::ValidationImpl() {
  bool is_valid = true;
  if (proc_rank_ == 0) {
    is_valid = !GetInput().empty() && std::isnan(GetOutput());
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSElemVecAvgMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgMPI::RunImpl() {
  // передача размера исходного массива
  int total_size = 0;
  if (proc_rank_ == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // распределение данных
  int batch_size = total_size / proc_num_;
  int remainder = total_size % proc_num_;
  int proc_size = batch_size + (proc_rank_ < remainder ? 1 : 0);
  InType local_data(proc_size);
  std::vector<int> displacements;
  std::vector<int> counts;
  if (proc_rank_ == 0) {
    displacements.resize(proc_num_);
    counts.resize(proc_num_);
    int offset = 0;
    for (int i = 0; i < proc_num_; i++) {
      counts[i] = batch_size + (i < remainder ? 1 : 0);
      displacements[i] = offset;
      offset += counts[i];
    }
  }
  MPI_Scatterv(GetInput().data(), counts.data(), displacements.data(), MPI_INT, local_data.data(), proc_size, MPI_INT,
               0, MPI_COMM_WORLD);

  // вычисления среднего элементов вектора
  int64_t local_sum = std::reduce(local_data.begin(), local_data.end(), int64_t{0});
  int64_t total_sum = 0;
  MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = static_cast<double>(total_sum) / static_cast<double>(total_size);

  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg

```

### 10.4. Приложение №4 — функциональные и валидационные тесты
Файл: `./tests/functional/main.cpp`.
```cpp
namespace otcheskov_s_elem_vec_avg {
// функциональные тесты
class OtcheskovSElemVecAvgFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string filename = FormatFileName(std::get<0>(test_param));
    std::string avg_str = FormatAverage(std::get<1>(test_param));
    return filename + "_" + avg_str;
  }

 protected:
  // открывает файл и считывает элементы вектора
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = std::get<0>(params);
    expected_avg_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num{};
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_avg_ = NAN;
  // форматирует название файла для имени теста
  static std::string FormatFileName(const std::string &filename) {
    size_t lastindex = filename.find_last_of('.');
    std::string name = filename;
    if (lastindex != std::string::npos) {
      name = filename.substr(0, lastindex);
    }

    std::string format_name = name;
    for (char &c : format_name) {
      if (std::isalnum(c) == 0 && c != '_') {
        c = '_';
      }
    }
    return format_name;
  }
  // форматирует ожидаемое выходное значение для имени теста
  static std::string FormatAverage(double value) {
    std::string str = RemoveTrailingZeros(value);
    if (value < 0) {
      str = "minus_" + str.substr(1, str.size());
    }

    for (char &c : str) {
      if (c == '.') {
        c = 'p';
      }
    }
    return "num_" + str;
  }
  // Убирает плавающие 0 в ожидаемом выходном значении
  static std::string RemoveTrailingZeros(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(10) << value;

    std::string str_value = ss.str();
    if (str_value.find('.') != std::string::npos) {
      str_value = str_value.substr(0, str_value.find_last_not_of('0') + 1);
      if (str_value.find('.') == str_value.size() - 1) {
        str_value = str_value.substr(0, str_value.size() - 1);
      }
    }
    return str_value;
  }
};

namespace {

TEST_P(OtcheskovSElemVecAvgFuncTests, VectorAverageFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {std::make_tuple("test_vec1.txt", 50.5),
                                            std::make_tuple("test_vec2.txt", 14.5),
                                            std::make_tuple("test_vec_one_elem.txt", 5.0),
                                            std::make_tuple("test_vec_fraction.txt", 4.0 / 3.0),
                                            std::make_tuple("test_vec_one_million_elems.txt", -2.60988),
                                            std::make_tuple("test_vec_alternating_elems.txt", 0.0),
                                            std::make_tuple("test_vec_zeros_elems.txt", 0.0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
    ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = OtcheskovSElemVecAvgFuncTests::PrintFuncTestName<OtcheskovSElemVecAvgFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAverageFuncTests, OtcheskovSElemVecAvgFuncTests, kGtestValues, kFuncTestName);


// Валидацонные тесты
class OtcheskovSElemVecAvgFuncTestsValidation : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    if (std::isnan(output_data)) {
      return true;
    }
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
  // переопределяем функцию запуска из ppc::util::BaseRunFuncTests
  void ExecuteTest(::ppc::util::FuncTestParam<InType, OutType, TestType> test_param) {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kNameTest)>(test_param);

    ValidateTestName(test_name);

    const auto test_env_scope = ppc::util::test::MakePerTestEnvForCurrentGTest(test_name);

    if (IsTestDisabled(test_name)) {
      GTEST_SKIP();
    }

    if (ShouldSkipNonMpiTask(test_name)) {
      std::cerr << "kALL and kMPI tasks are not under mpirun\n";
      GTEST_SKIP();
    }

    task_ =
        std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kTaskGetter)>(test_param)(GetTestInputData());
    const TestType &params = std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kTestParams)>(test_param);
    const std::string param_name = std::get<0>(params);
    // Специально для теста изменяем состояние входных и выходных данных до запуска теста
    if (param_name.find("_changed_output_") != std::string::npos) {
      task_->GetInput() = {1, 1, 1, 1, 1};
      task_->GetOutput() = 1.0;
    }
    ExecuteTaskPipeline();
  }
  // Переопределяем функцию запуска пайплайна из ppc::util::BaseRunFuncTests
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void ExecuteTaskPipeline() {
    EXPECT_FALSE(task_->Validation()); // проверяем провал валидации
    task_->PreProcessing();            // активируем остальные шаги для корректного удаления задачи
    task_->Run();
    task_->PostProcessing();
  }

 private:
  InType input_data_;
  OutType expected_avg_ = NAN;
  ppc::task::TaskPtr<InType, OutType> task_;
};

TEST_P(OtcheskovSElemVecAvgFuncTestsValidation, VectorAverageFuncTestsValidation) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kValidationTestParam = {std::make_tuple("test_empty_vec", NAN),
                                                      std::make_tuple("test_changed_output_before_run", NAN)};

const auto kValidationTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(
                                                         kValidationTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
                                                     ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(
                                                         kValidationTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kValidationGtestValues = ppc::util::ExpandToValues(kValidationTestTasksList);

const auto kValidationFuncTestName =
    OtcheskovSElemVecAvgFuncTestsValidation::PrintFuncTestName<OtcheskovSElemVecAvgFuncTestsValidation>;

INSTANTIATE_TEST_SUITE_P(VectorAverageFuncTestsValidation, OtcheskovSElemVecAvgFuncTestsValidation,
                         kValidationGtestValues, kValidationFuncTestName);

}  // namespace

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.5. Приложение №5 — проиводительные тесты
Файл: `./tests/performance/main.cpp`.
```cpp
namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_avg_ = NAN;

  void SetUp() override {
    std::string filename = "test_vec_one_million_elems.txt";
    expected_avg_ = -2.60988;

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num{};
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();

    // 2 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 4 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 8 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 16 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 32 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 64 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 128 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 256 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 512 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OtcheskovSElemVecAvgPerfTests, VectorAveragePerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, OtcheskovSElemVecAvgMPI, OtcheskovSElemVecAvgSEQ>(
    PPC_SETTINGS_otcheskov_s_elem_vec_avg);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSElemVecAvgPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VectorAveragePerfTests, OtcheskovSElemVecAvgPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_elem_vec_avg

```