import numpy as np
import matplotlib.pyplot as plt

# Эталонные паттерны (1 – активен, 0 – нет)
patterns = {
    'L': np.array([[1,0,0],[1,0,0],[1,1,1]]),
    'T': np.array([[1,1,1],[0,1,0],[0,1,0]]),
    'X': np.array([[1,0,1],[0,1,0],[1,0,1]]),
    'C': np.array([[1,1,1],[1,0,0],[1,1,1]]),
    'O': np.array([[1,1,1],[1,0,1],[1,1,1]]),
    'П': np.array([[1,1,1],[1,0,1],[1,0,1]])
}

# Суммируем все паттерны
sum_mat = np.sum(list(patterns.values()), axis=0)  # матрица 3x3, значения от 0 до 6

# Рисуем одну картинку
plt.figure(figsize=(4, 4))
plt.imshow(sum_mat, cmap='gray', vmin=0, vmax=6)
plt.colorbar(label='Количество паттернов')
plt.title('Наложение 6 эталонов (3x3)')
plt.axis('off')
plt.show()