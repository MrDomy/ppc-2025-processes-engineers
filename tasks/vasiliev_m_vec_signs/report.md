# Отчет по лабораторной работе
## Нахождение числа чередований знаков значений соседних элементов вектора

- Студент: Васильев Михаил Петрович, группа 3823Б1ПР2
- Технология: SEQ | MPI
- Вариант: 5

# 1. Введение
В современных условиях обработки больших массивов данных особое значение имеет эффективное использование вычислительных ресурсов. Параллельные вычисления позволяют значительно уменьшить время выполнения алгоритмов за счет распределения вычислительной нагрузки между несколькими процессорами или процессами.

Одним из наиболее широко используемых инструментов параллельного программирования является технология MPI (Message Passing Interface), обеспечивающая взаимодействие процессов посредством обмена сообщениями в распределённых системах.

В рамках данной работы исследуется задача на нахождение числа чередований знаков значений соседних элементов вектора. Для решения задачи были реализованы две версии алгоритма: последовательная (SEQ) и параллельная (MPI). Основная цель работы заключается в сравнении их производительности и анализе эффективности параллелизации для данной задачи.

---

# 2. Постановка задачи
Дан вектор размера `n`, где `n` — целое положительное число (int). Необходимо найти количество чередований знаков соседних элементов вектора.

Например, в векторе `1, 4, -3, 2, -1, 3` будет `4` чередования. При этом, элемент `0` не будет влиять на изменение знака и, следовательно, не инкрементировать число чередований.

---

# 3. Последовательная версия (SEQ)

SEQ и MPI версия алгоритмов состоят из 4-х этапов:
  `ValidationImpl()` - Проверка входных данных.
  `PreProcessingImpl()` - Подготовка к выполнению алгоритма.
  `RunImpl()` - Основной алгоритм.
  `PostProcessingImpl()` - Проверка конечного результата.

Далее работа данных методов в SEQ версии алгоритма:

**1. `ValidationImpl()`**
- Проверяется, что входной вектор не пуст.

**2. `PreProcessingImpl()`**
- Обнуление выходного результата.

**3. `RunImpl()`**
1. Создание переменной `alters`, для содержания количества чередований знаков.
2. Через цикл проверяются соседние элементы. Если один из них больше нуля, а другой меньше - переменная `alters` инкрементируется.

**4. `PostProcessingImpl()`**
- Проверяется, что итоговое число чередований больше или равно нулю.

---

# 4. Схема параллелизации
Идея параллелизации заключается в том, что вектор можно разбить на независимые части, что позволяет распределить обработку каждой части отдельным процессом параллельно.

Каждый процесс высчитывает количество чередований в своей рассматриваемой части вектора, затем идет проверка на чередование знаков между этими частями и, в итоге, результаты суммируются в главном процессе операцией `MPI_Allreduce`.


## Схема
Главным процессом (rank 0) инициализируется MPI и, в зависимости от указанного количества процессов в параметрах, создается данное число отдельных процессов, занимающие свою часть вектора (а если процессы не могут равномерно занять части, то первые из них будут иметь больше элементов на обработку, чем остальные).

Таким образом (при равномерном распределении): `Процесс_0` займет первые `n` элементов вектора для обработки, `Процесс_1` займет `с n+1 до 2n` элементы, и так далее.

Затем каждый процесс обрабатывает свои отдельные части и получает локальный результат чередований.

Далее, дополнительно, происходит проверка на чередование между данными частями вектора, не покрытые процессами.

После этого все локальные результаты суммируются в главный процесс (rank 0), что будет передаваться как конечный результат работы алгоритма.


## Распределение нагрузки
При выделении `k` процессов и обработки вектора размером `n`, количество элементов распределяются следующим образом:

```cpp
int chunk = n / k;          // Изначальное количество элементов на процесс
remain = n % k;              // Остаточное количество неиспользованных элементов вектора
```

(То есть первые `remain` процессов получают на один элемент больше, что обеспечивает равномерное распределение нагрузки)

```cpp
vector<int> counts(k);
  for (int i = 0; i < remain; i++) {
    counts[i]++;
  }  // counts[i] хранит, сколько элементов получит процесс i

vector<int> displs(k);
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }  // displs[i] — с какого элемента вектора начнётся часть процесса i
```

**Пример распределения для n=10, k=3:**
- Процесс 0: элементы 0-3 (4 элемента)
- Процесс 1: элементы 4-6 (3 элемента)
- Процесс 2: элементы 7-9 (3 элемента)


## Коммуникации между процессами

MPI версия алгоритма использует следующие коммуникации:

1. **MPI_Bcast** — рассылает процессам их отдельное количество элементов вектора (counts) и смещение в нем (displs):
   ```cpp
   MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
   ```
2. **MPI_Scatterv** — раздача данных (частей вектора) всем процессам (учитывая возможное неравномерное разбитие):
   ```cpp
   MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT,
              local_data.data(), counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
   ```
3. **MPI_Recv** — каждый процесс (кроме первого) принимает последний элемент предыдущего процесса (учитывание чередований между частями вектора):
   ```cpp
   MPI_Recv(&prev_last, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   ```
4. **MPI_Send** — каждый процесс (кроме последнего) отправляет последний элемент своей части вектора следующему процессу (учитывание чередований между частями вектора):
   ```cpp
   MPI_Send(&send_val, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
   ```
5. **MPI_Allreduce** — суммирует значения всех локальных чередований и перенаправляет этот результат всем процессам (результат необходим всем для корректного завершения выполнения вне зависимости от `rank`):
   ```cpp
   MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   ```

---

# 5. Программная реализация (MPI версия алгоритма)
## Структура решения
MPI-версия реализована классом `VasilievMVecSignsMPI`, наследуемый от `BaseTask`. Структура реализована на `Pipeline`, имеющий четыре последовательных этапа:

1. **Validation** — валидация входных данных
2. **PreProcessing** — предварительная обработка
3. **Run** — основной алгоритм/вычисления
4. **PostProcessing** — постобработка/проверка результатов

## Структура класса
```cpp
namespace vasiliev_m_vec_signs {
  using InType = std::vector<int>;
  using OutType = int;
  using BaseTask = ppc::task::Task<InType, OutType>;

  class VasilievMVecSignsMPI : public BaseTask {
   public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kMPI;
    }
    explicit VasilievMVecSignsMPI(const InType &in);
    static bool SignChangeCheck(int a, int b); // отдельная функция для проверки двух соседних элементов
    static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);  // вычисление кол-ва элементов на процесс и смещения в векторе
  
   private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;
  };
}
```
## Реализация методов
### Конструктор
```cpp
VasilievMVecSignsMPI::VasilievMVecSignsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}
```
Конструктор инициализирует тип задачи (в данном случае MPI), устанавливает входные данные и тип выходных.

### Валидация данных
```cpp
bool VasilievMVecSignsMPI::ValidationImpl() {
  return !GetInput().empty();
}
```
Проверка на корректность входных данных (вектор не является пустым).

### Предварительная обработка
```cpp
bool VasilievMVecSignsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}
```
Зануление выходных данных.

#### Основной алгоритм
```cpp
bool VasilievMVecSignsMPI::RunImpl() {
  int rank = 0; // информация о процессах
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &vec = GetInput();
  int n = static_cast<int>(vec.size());

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  // вычисления counts и displs происходят только на root-ранге
  if (rank == 0) {
    CalcCountsAndDispls(n, size, counts, displs);  // распределение частей вектора между процессами
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);  // рассылка counts и displs всем процессам
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // раздача выбранного количества элементов процессам
  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT,
              local_data.data(), counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  int local_count = 0;

  // нахождение числа чередований в локальной части вектора
  if (!local_data.empty()) {
    for (size_t i = 0; i < local_data.size() - 1; i++) {
      if (SignChangeCheck(local_data[i], local_data[i + 1])) {
        local_count++;
      }
    }
  }

  // проверка чередований между частями вектора
  int first_elem = local_data.empty() ? 0 : local_data.front();
  int last_elem = local_data.empty() ? 0 : local_data.back();

  int prev_last = 0;
  if (rank > 0) {
    MPI_Recv(&prev_last, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!local_data.empty() && SignChangeCheck(prev_last, first_elem)) {
      local_count++;
    }
  }

  if (rank < size - 1) {
    MPI_Send(&last_elem, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }

  // суммирование количества всех локальных чередований в global_count и перенаправление результата всем процессам
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;

  return true;
}
```

### Постобработка результатов

```cpp
bool VasilievMVecSignsMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}
```
Проверка на неотрицательный результат (общее число чередований >= 0).

### Вспомогательные функции
```cpp
bool VasilievMVecSignsMPI::SignChangeCheck(int a, int b) {
  return (a > 0 && b < 0) || (a < 0 && b > 0);
}
```
Проверка на чередование знаков соседних элементов вектора.

```cpp
void VasilievMVecSignsMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  int chunk = n / size;
  int remain = n % size;

  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < remain ? 1 : 0);
  }

  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}
```
Вычисление количества элементов на процесс и смещения в векторе.

### Особые случаи
В данном алгоритме предполагается, что нули в векторе не влияют на чередование (например, при векторе: `-4, 0, -2`; число чередований будет равняться `0`), поэтому сравнение соседних элементов в векторе - строгое.

### Преимущества MPI-реализации
- **Параллелизация** — каждый процесс хранит и обрабатывает только свою часть вектора, следовательно сложность обработки сокращается в `k` раз.
- **Масштабируемость** — алгоритм будет работать все более эффективно на все более большем векторе и/или при увеличении числа процессов.
- **Отсутствие гонок данных** — каждый процесс работает только со своими отдельными частями вектора, не затрагивая данные других процессов.

---

# 6. Результаты экспериментов
## Типы экспериментов/тестов
Для проверки корректной работы алгоритмов и ее производительности были составлены:
- Функциональные тесы - для проверки результатов алгоритмов
- Тесты производительности - для измерения производительности/времени выполнения работы алгоритмом
Для написания тестов были использованы `google tests`.

## Алгоритм считывания тестовых значений
Все тестовые векторы записаны в отдельном файле формата `.txt`.

Схема считывания векторов в тестах следующая: подразумевается, что один вектор занимает одну строку, следовательно считывание данных происходит построчно.
Элементы вектора разделяются пробелами (без запятых) и сигнализируется их окончание точкой с запятой. После этого знака указывается истинное количество чередований знаков в данном векторе.
(Напр.: `1 -2 3 -4 ; 3`)

Алгоритм считывания тестовых значений позволяет переводить векторы из `string` формата в обрабатываемый формат `vector<int>`, для дальнейшей работы в вычислении количества чередований (SEQ или MPI).

Соответственный алгоритм:
```cpp
std::stringstream ss(line);  // создание потока для строки, читающий числа до `;`
      std::vector<int> vec;
      int val;
      while (ss >> val) {
        vec.push_back(val);  // добавление числа в вектор
        ss >> std::ws;
        if (ss.peek() == ';') {  // встреча `;` - окончание цикла
          ss.get();
          break;
        }
      }  // по окончании цикла вектор string вида был преобразован в тип vector<int>

      int expected = 0;
      ss >> expected;  // считывание ожидаемого результата
```

## Функциональные тесты
Для проверки корректности алгоритмов было написано 5 тестовых векторов с заготовленным результатом количества чередований в них.

`1 -2 3 -4 ; 3`
`5 6 -7 8 -9 ; 3` - первые два теста проверяют базовую работу (в первом чередования идут через каждый элемент; во втором - добавлен элемент вначале, не влияющий на результат)

`5 6 7 8 9 9 2 ; 0` - чередований знаков нет (ожидаемый результат - 0)
`5 0 -7 1 0 9 ; 1` - проверка обработки нулей в алгоритме (их пропуска)

5 тест - увеличенный вектор размером в `100` элементов, для дополнительной проверки корректности работы.

### Результаты
**Все 5 функциональных тестов были успешно пройдены** для обеих версий (SEQ и MPI) при различном количестве процессов: 1, 2, 4, 6, 8 (работа проводилась на машине с 6 доступными ядрами).

- Были верно подсчитаны тестовые векторы
- Результаты обоих версий алгоритмов (SEQ, MPI) были идентичны
- Корректна распределена нагрузка при параллельном выполнении
- Правильная обработка нулей в векторе
- Верная работа алгоритма при большем количестве процессов, чем частей вектора или физических ядер 

## Тесты производительности
Для наиболее явной проверки производительности были реализованы тесты на векторе размером `110 млн.` элементов с двумя режимами измерений:

1. **task_run** — измерение времени выполнения метода `RunImpl()`
2. **pipeline** — измерение времени выполнения всего конвейера (`ValidationImpl()` + `PreProcessingImpl()` + `RunImpl()` + `PostProcessingImpl()`)

### Ускорение при параллелизации
При увеличении количества процессов для работы параллельного алгоритма, ускорение не будет являться линейным из-за накладных расходов:
- Инициализация MPI
- Рассылка разных данных процессам
- Ожидание завершение всех процессов (из-за возможного неравномерного распределения элементов вектора на процессы)
- Сбор результатов

В дополнение к этому, из-за вышеперечисленных пунктов параллельный алгоритм на малых данных будет выполняться дольше, чем последовательная версия (затраты на параллелизацию больше, чем выигрыш от ее преимуществ). Следовательно, ускорение будет заметно только на большом векторе.

---

# 7. Результаты и выводы
## Корректность реализации
Все написанные тесты были выполнены верно при применении последовательной (SEQ) и параллельной (MPI) версии алгоритма (результаты работ алгоритмов равны предзаписанным ответам). Результаты работ обоих алгоритмов совпадают.

При вычислении тестового вектора, цель которого - проверка производительности, алгоритмы так же корректно выполнили задачу и предоставили явные показатели ускорения и эффективности.

## Инфраструктура для тестов
### Виртуальная машина (VirtualBox)
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 9400F (6 cores, 6 threads, 2900 MHz)   |
| RAM        | 10 GB DDR4 (2660 MHz)                                |
| OS         | Ubuntu 24.04.3 LTS                                   |
| Compiler   | GCC 13.3.0, Release Build                            |

## Производительность
Изначально размер тестируемого вектора представляет `10 млн.` элементов. Для более справедливой оценки, размер, перед тестом, увеличивается путем копирования всех существующих элементов и вставки их как новых в цикле. При тестировании итогового вектора размером `110 млн.` элементов с `~45 млн.` чередований, показатели производительности (при разных количествах процессов) были следующими:

`(из-за специфики использования виртуальной машины, доступных аппаратных ресурсов и способа генерации тестового вектора, увеличение еще в несколько раз размерности вектора приводит к несправедливым результатам из-за bottleneck-а аппаратуры и синхронизации/передачи больших буферов данных)`

| Версия алг-ма        | Кол-во процессов | Время, с | Ускорение | Эффективность |
|----------------------|------------------|----------|-----------|---------------|
| SEQ                  | 1                | 0.527    | 1.00      | N/A           |
| MPI                  | 2                | 0.392    | 1.34      | 67.0%         |
| MPI                  | 4                | 0.239    | 2.21      | 55.3%         |
| MPI                  | 6                | 0.218    | 2.42      | 40.3%         |

Из таблицы видно, что при увеличении количества процессов время выполнения алгоритмов сокращается, а его работа ускоряется с хорошей масштабируемостью.

При использовании 2 процессов, время работы было сокращено в 1,3 раза, при использовании 4 процессов - более, чем в 2 раза, а при использовании 6 - в 2,42 раза. Эффективность, при увеличении процессов уменьшается, вследствие синхронизаций и обменов сообщений.

(При использовании 8 процеесов и флага `--oversubscribe` время работы увеличится и будет вести себя непредсказуемо, из-за использованием нескольких процессов одного физического ядра).

Но следует учесть, что при тестировании на векторах размером меньше `~1 млн.` элементов, накладные затраты параллелизации перевешивают выигрыш от параллельного вычисления и в таких случаях следует использовать последовательную версию алгоритма (или меньшее количество процессов при MPI версии для достижения наименьшего возможного времени, в зависимости от размеров вектора).

## Выводы
- Функциональны тесты и тесты для производительности выполняются корректно.
- MPI версия алгоритма эффективна для векторов больших размеров (5 млн и более).
- SEQ версия алгоритма эффективна для векторов меньших размеров (из-за накладных расходов)
- Хорошее масштабирование при увеличении числа процессов.
- Корректная работа параллельной версии алгоритма при количестве процессов меньше, чем доступных ядер; такого же количества; и больше.
- Распределение нагрузки имеет ключевую роль в увеличении эффективности работы при параллелизации.

---

# 8. Заключение
В рамках данной работы была успешно решена задача нахождение числа чередований знаков значений соседних элементов вектора с использованием технологии MPI.
Были разработаны и реализованы последовательная (SEQ) и параллельная (MPI) версии алгоритма.
Было выявлено на каких размерах данных и какого ускорения возможно достигнуть при использовании параллелизации.

В итоге, результаты показали, что применение MPI позволяет повысить производительность при обработке больших объёмов данных, а применяемый подход может эффективно использоваться для разработки как в задачах обработки векторов, так и для более масштабных вычислительных алгоритмов.

---

# 9. Список литературы

1. MPI Guide. **Using MPI with C++: a basic guide (2024)** [Электронный ресурс]. — Режим доступа: https://www.paulnorvig.com/guides/using-mpi-with-c.html

2. Snir M., Otto S. **MPI: The Complete Reference**. - MIT Press Cambridge, 1995 - 217 p.

3. MPI Forum. **MPI: A Message-Passing Interface Standard. Version 3.1** [Электронный ресурс]. — Режим доступа: https://www.mpi-forum.org/docs/

4. Google Test Documentation. **GoogleTest User’s Guide** [Электронный ресурс]. — Режим доступа: https://google.github.io/googletest/

---

# 10. Приложение
## Общие обозначения `common.hpp`

```cpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace vasiliev_m_vec_signs {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vasiliev_m_vec_signs
```

## Header SEQ версии `ops_seq.hpp`

```cpp
#pragma once

#include "task/include/task.hpp"
#include "vasiliev_m_vec_signs/common/include/common.hpp"

namespace vasiliev_m_vec_signs {

class VasilievMVecSignsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VasilievMVecSignsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_vec_signs

```

## Реализация методов SEQ версии `ops_seq.cpp`

```cpp
#include "vasiliev_m_vec_signs/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "vasiliev_m_vec_signs/common/include/common.hpp"

namespace vasiliev_m_vec_signs {

VasilievMVecSignsSEQ::VasilievMVecSignsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMVecSignsSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMVecSignsSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool VasilievMVecSignsSEQ::RunImpl() {
  auto &vec = GetInput();
  int alters = 0;

  for (size_t i = 0; i < (vec.size() - 1); i++) {
    if ((vec[i] > 0 && vec[i + 1] < 0) || (vec[i] < 0 && vec[i + 1] > 0)) {
      alters++;
    }
  }

  GetOutput() = alters;
  return true;
}

bool VasilievMVecSignsSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace vasiliev_m_vec_signs

```

## Header MPI версии `ops_mpi.hpp`

```cpp
#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vasiliev_m_vec_signs/common/include/common.hpp"

namespace vasiliev_m_vec_signs {

class VasilievMVecSignsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VasilievMVecSignsMPI(const InType &in);
  static bool SignChangeCheck(int a, int b);
  static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_vec_signs

```

## Реализация методов MPI версии `ops_mpi.cpp`

```cpp
#include "vasiliev_m_vec_signs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "vasiliev_m_vec_signs/common/include/common.hpp"

namespace vasiliev_m_vec_signs {

VasilievMVecSignsMPI::VasilievMVecSignsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMVecSignsMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMVecSignsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool VasilievMVecSignsMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &vec = GetInput();
  int n = static_cast<int>(vec.size());

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    CalcCountsAndDispls(n, size, counts, displs);
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  int local_count = 0;

  if (!local_data.empty()) {
    for (size_t i = 0; i < local_data.size() - 1; i++) {
      if (SignChangeCheck(local_data[i], local_data[i + 1])) {
        local_count++;
      }
    }
  }

  int first_elem = local_data.empty() ? 0 : local_data.front();
  int last_elem = local_data.empty() ? 0 : local_data.back();

  int prev_last = 0;
  if (rank > 0) {
    MPI_Recv(&prev_last, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!local_data.empty() && SignChangeCheck(prev_last, first_elem)) {
      local_count++;
    }
  }

  if (rank < size - 1) {
    MPI_Send(&last_elem, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;

  return true;
}

void VasilievMVecSignsMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  int chunk = n / size;
  int remain = n % size;

  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < remain ? 1 : 0);
  }

  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}

bool VasilievMVecSignsMPI::SignChangeCheck(int a, int b) {
  return (a > 0 && b < 0) || (a < 0 && b > 0);
}

bool VasilievMVecSignsMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace vasiliev_m_vec_signs

```

## Тесты проверки функционала `functional/main.cpp`

```cpp
#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_vec_signs/common/include/common.hpp"
#include "vasiliev_m_vec_signs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_vec_signs/seq/include/ops_seq.hpp"

namespace vasiliev_m_vec_signs {

class VasilievMVecSignsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_vec_signs, "test_vectors.txt");
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    test_vectors_.clear();
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      std::stringstream ss(line);
      std::vector<int> vec;
      int val = 0;
      while (ss >> val) {
        vec.push_back(val);
        ss >> std::ws;
        if (ss.peek() == ';') {
          ss.get();
          break;
        }
      }

      int expected = 0;
      ss >> expected;
      test_vectors_.emplace_back(vec, expected);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int index = std::get<0>(params);
    input_data_ = test_vectors_[index].first;
    expected_output_ = test_vectors_[index].second;
    return input_data_;
  }

 private:
  std::vector<std::pair<std::vector<int>, int>> test_vectors_;
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(VasilievMVecSignsFuncTests, AlternationsInVector) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(0, "case1"), std::make_tuple(1, "case2"),
                                            std::make_tuple(2, "case3"), std::make_tuple(3, "case4"),
                                            std::make_tuple(4, "case5")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<VasilievMVecSignsMPI, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_vec_signs),
                   ppc::util::AddFuncTask<VasilievMVecSignsSEQ, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_vec_signs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VasilievMVecSignsFuncTests::PrintFuncTestName<VasilievMVecSignsFuncTests>;

INSTANTIATE_TEST_SUITE_P(SignAlternationsTests, VasilievMVecSignsFuncTests, kGtestValues, kPerfTestName);

TEST(SignChangeCheckTests, PosToNeg) {
  EXPECT_TRUE(vasiliev_m_vec_signs::VasilievMVecSignsMPI::SignChangeCheck(1, -1));
}

TEST(SignChangeCheckTests, NegToPos) {
  EXPECT_TRUE(vasiliev_m_vec_signs::VasilievMVecSignsMPI::SignChangeCheck(-5, 3));
}

TEST(SignChangeCheckTests, NoChangePos) {
  EXPECT_FALSE(vasiliev_m_vec_signs::VasilievMVecSignsMPI::SignChangeCheck(5, 8));
}

TEST(SignChangeCheckTests, NoChangeNeg) {
  EXPECT_FALSE(vasiliev_m_vec_signs::VasilievMVecSignsMPI::SignChangeCheck(-4, -1));
}

TEST(SignChangeCheckTests, Zero) {
  EXPECT_FALSE(vasiliev_m_vec_signs::VasilievMVecSignsMPI::SignChangeCheck(4, 0));
}

}  // namespace

}  // namespace vasiliev_m_vec_signs

```

## Тесты проверки производительности `performance/main.cpp`

```cpp
#include <gtest/gtest.h>

#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_vec_signs/common/include/common.hpp"
#include "vasiliev_m_vec_signs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_vec_signs/seq/include/ops_seq.hpp"

namespace vasiliev_m_vec_signs {

class VasilievMVecSignsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::vector<std::pair<std::vector<int>, int>> test_vectors_;
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_vec_signs, "perf_test_vector.txt");
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    test_vectors_.clear();
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      std::stringstream ss(line);
      std::vector<int> vec;
      int val = 0;
      while (ss >> val) {
        vec.push_back(val);
        ss >> std::ws;
        if (ss.peek() == ';') {
          ss.get();
          break;
        }
      }

      int expected = 0;
      ss >> expected;
      
      std::vector<int> incr_vec = vec;

      for (int i = 0; i < 10; i++) {
        incr_vec.insert(incr_vec.end(), vec.begin(), vec.end());
      }

      int expected_incr = (expected * 11) + 10;

      test_vectors_.emplace_back(incr_vec, expected_incr);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    input_data_ = test_vectors_[0].first;
    expected_output_ = test_vectors_[0].second;
    return input_data_;
  }
};

TEST_P(VasilievMVecSignsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VasilievMVecSignsMPI, VasilievMVecSignsSEQ>(PPC_SETTINGS_vasiliev_m_vec_signs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VasilievMVecSignsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VasilievMVecSignsPerfTests, kGtestValues, kPerfTestName);

}  // namespace vasiliev_m_vec_signs

```


