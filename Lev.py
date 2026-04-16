import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.gridspec as gridspec

# Настройки для черно-белого отображения
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 11
plt.rcParams['hatch.linewidth'] = 1.5

class SingleLetterBWComparison:
    def __init__(self, threshold=5):
        self.grid_size = 3
        self.n_neurons = 9
        self.threshold = threshold
        
        # Все эталоны
        self.etalons = {
            'L': [10, 0, 0, 10, 0, 0, 10, 10, 10],
            'T': [10, 10, 10, 0, 10, 0, 0, 10, 0],
            'X': [10, 0, 10, 0, 10, 0, 10, 0, 10],
            'C': [10, 10, 10, 10, 0, 0, 10, 10, 10]
        }
        
        # Выбираем букву для анализа
        self.selected_letter = 'L'  # МОЖНО ИЗМЕНИТЬ НА 'T', 'X', 'C'
        self.etalon_vector = self.etalons[self.selected_letter]
        
        # Все тестовые случаи для выбранной буквы
        self.passing_cases = [
            # Полный эталон
            {
                'vector': [10, 0, 0, 10, 0, 0, 10, 10, 10],
                'name': 'Полный L',
                'type': 'full',
                'description': 'Полное совпадение'
            },
            # Фрагменты
            {
                'vector': [10, 0, 0, 10, 0, 0, 10, 0, 0],
                'name': 'Вертикаль L',
                'type': 'fragment',
                'description': 'Левая вертикаль'
            },
            {
                'vector': [0, 0, 0, 0, 0, 0, 10, 10, 10],
                'name': 'Горизонталь L',
                'type': 'fragment',
                'description': 'Нижняя горизонталь'
            },
            {
                'vector': [0, 0, 0, 10, 0, 0, 10, 10, 0],
                'name': 'Угол L',
                'type': 'fragment',
                'description': 'Угловой фрагмент'
            },
            {
                'vector': [10, 0, 0, 0, 0, 0, 0, 0, 0],
                'name': 'Один нейрон',
                'type': 'fragment',
                'description': 'N1 из эталона'
            },
            # Пустой
            {
                'vector': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                'name': 'Пустой',
                'type': 'empty',
                'description': 'Пустой паттерн'
            }
        ]
        
        self.blocked_cases = [
            # С лишними нейронами
            {
                'vector': [10, 10, 0, 10, 0, 0, 10, 10, 10],
                'name': 'Лишний N2',
                'type': 'extra',
                'mismatch': [1],
                'description': 'Добавлен N2'
            },
            {
                'vector': [10, 0, 0, 10, 10, 0, 10, 10, 10],
                'name': 'Лишний N5',
                'type': 'extra',
                'mismatch': [4],
                'description': 'Добавлен N5'
            },
            {
                'vector': [10, 0, 10, 10, 0, 0, 10, 10, 10],
                'name': 'Лишний N3',
                'type': 'extra',
                'mismatch': [2],
                'description': 'Добавлен N3'
            },
            {
                'vector': [10, 0, 0, 10, 0, 10, 10, 10, 10],
                'name': 'Лишние N6,N9',
                'type': 'extra',
                'mismatch': [5, 8],
                'description': 'Добавлены N6,N9'
            },
            
            # Неправильные формы
            {
                'vector': [0, 0, 0, 0, 10, 0, 0, 10, 0],
                'name': 'Чужая верт.',
                'type': 'wrong',
                'mismatch': [4, 7],
                'description': 'Вертикаль в центре'
            },
            {
                'vector': [0, 0, 10, 0, 0, 0, 10, 10, 0],
                'name': 'Чужой угол',
                'type': 'wrong',
                'mismatch': [2, 6, 7],
                'description': 'Угол справа'
            },
            
            # Другие буквы
            {
                'vector': [10, 10, 10, 0, 10, 0, 0, 10, 0],
                'name': 'Буква T',
                'type': 'other',
                'mismatch': [0, 1, 2, 4, 7],
                'description': 'Другой паттерн'
            },
            {
                'vector': [10, 0, 10, 0, 10, 0, 10, 0, 10],
                'name': 'Буква X',
                'type': 'other',
                'mismatch': [0, 2, 4, 6, 8],
                'description': 'Другой паттерн'
            },
            
            # Шум
            {
                'vector': [10, 10, 0, 0, 10, 10, 0, 0, 10],
                'name': 'Шум 1',
                'type': 'noise',
                'mismatch': [0, 1, 4, 5, 8],
                'description': 'Случайный'
            },
            {
                'vector': [0, 10, 0, 10, 0, 10, 0, 10, 0],
                'name': 'Шум 2',
                'type': 'noise',
                'mismatch': [1, 3, 5, 7],
                'description': 'Случайный'
            }
        ]
    
    def check_pattern(self, test_vector):
        """Проверяет, является ли тест подмножеством эталона"""
        etalon_activity = [1 if v >= self.threshold else 0 for v in self.etalon_vector]
        test_activity = [1 if v >= self.threshold else 0 for v in test_vector]
        
        mismatches = []
        for i in range(9):
            if test_activity[i] == 1 and etalon_activity[i] == 0:
                mismatches.append(i)
        
        passes = len(mismatches) == 0
        return passes, mismatches
    
    def draw_pattern(self, ax, vector, title, description='', 
                    highlight_mismatches=None, is_passing=True, is_etalon=False):
        """Рисует один паттерн в черно-белом стиле"""
        activity = [1 if v >= self.threshold else 0 for v in vector]
        
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(-0.5, 2.3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Заголовок
        ax.text(1, 2.5, title, 
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color='black')
        
        # Описание
        if description:
            ax.text(1, -0.25, description,
                   ha='center', va='center',
                   fontsize=9, color='black', style='italic')
        
        # Легкая сетка для ориентации
        for i in range(4):
            ax.plot([-0.2, 2.2], [i*0.75 - 0.2, i*0.75 - 0.2], 
                   'k-', linewidth=0.3, alpha=0.2)
            ax.plot([i*0.75 - 0.2, i*0.75 - 0.2], [-0.2, 2.2], 
                   'k-', linewidth=0.3, alpha=0.2)
        
        # Рисуем нейроны (увеличенный размер)
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                is_active = activity[idx] == 1
                
                x, y = j, 2 - i
                
                # Рамка нейрона
                rect = Rectangle((x - 0.35, y - 0.35), 0.7, 0.7,
                                facecolor='white',
                                edgecolor='black',
                                linewidth=1.5,
                                alpha=1.0)
                ax.add_patch(rect)
                
                if is_etalon and is_active:
                    # Эталон - залитый черный квадрат
                    fill = Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                                   facecolor='black',
                                   edgecolor='black',
                                   linewidth=1)
                    ax.add_patch(fill)
                    
                elif highlight_mismatches and idx in highlight_mismatches:
                    # Лишний нейрон - жирный крест и штриховка
                    ax.plot([x-0.25, x+0.25], [y-0.25, y+0.25], 'k-', linewidth=2.5)
                    ax.plot([x-0.25, x+0.25], [y+0.25, y-0.25], 'k-', linewidth=2.5)
                    # Штриховка фона
                    hatch = Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                                    facecolor='none',
                                    edgecolor='none',
                                    hatch='///',
                                    alpha=0.4)
                    ax.add_patch(hatch)
                    
                elif is_active:
                    if is_passing:
                        # Проходящий - большой черный круг
                        circle = Circle((x, y), 0.25,
                                      facecolor='black',
                                      edgecolor='black',
                                      linewidth=1)
                        ax.add_patch(circle)
                    else:
                        # Блокируемый - круг с крестом
                        circle = Circle((x, y), 0.25,
                                      facecolor='black',
                                      edgecolor='black',
                                      linewidth=1)
                        ax.add_patch(circle)
                        ax.text(x, y, '✗',
                               ha='center', va='center',
                               fontsize=14, fontweight='bold',
                               color='white')
                else:
                    # Неактивный - пустой круг
                    circle = Circle((x, y), 0.25,
                                  facecolor='white',
                                  edgecolor='black',
                                  linewidth=1.5)
                    ax.add_patch(circle)
        
        # Маленькие номера нейронов для справки
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                x, y = j, 2 - i
                ax.text(x, y-0.5, f'{idx+1}',
                       ha='center', va='center',
                       fontsize=7, color='gray', alpha=0.7)
    
    def create_figure(self):
        """Создает черно-белый рисунок для выбранной буквы"""
        
        n_passing = len(self.passing_cases)
        n_blocked = len(self.blocked_cases)
        n_rows = max(n_passing, n_blocked)
        
        # Создаем фигуру (убираем место под легенду)
        fig = plt.figure(figsize=(16, 1.2 * n_rows + 3))
        
        # Основной заголовок
        plt.suptitle(f'РАСПОЗНАВАНИЕ БУКВЫ "{self.selected_letter}": ПРОХОДЯЩИЕ VS БЛОКИРУЕМЫЕ', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Правило (компактное)
        plt.figtext(0.5, 0.94, 
                   f'Правило: паттерн проходит ⇔ все его активные нейроны есть в эталоне {self.selected_letter}',
                   ha='center', fontsize=10, style='italic')
        
        # Создаем сетку: строки = n_rows + 2, колонки = 3
        gs = gridspec.GridSpec(n_rows + 2, 3, 
                              height_ratios=[0.8, 0.5] + [1] * n_rows,
                              width_ratios=[1, 1.2, 1],
                              hspace=0.5, wspace=0.3)
        
        # --- ЭТАЛОН в центре верха (увеличенный) ---
        ax_etalon = plt.subplot(gs[0, 1])
        self.draw_pattern(ax_etalon, self.etalon_vector, 
                         f'ЭТАЛОН: БУКВА "{self.selected_letter}"', 
                         is_etalon=True, is_passing=True)
        
        # --- ЗАГОЛОВКИ КОЛОНОК (компактные) ---
        # Левая колонка - ПРОХОДЯТ
        ax_title_left = plt.subplot(gs[1, 0])
        ax_title_left.axis('off')
        ax_title_left.text(0.5, 0.3, '✓ ПРОХОДЯЩИЕ', 
                          ha='center', va='center',
                          fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor='white', edgecolor='black', linewidth=2))
        
        # Правая колонка - БЛОКИРУЮТСЯ
        ax_title_right = plt.subplot(gs[1, 2])
        ax_title_right.axis('off')
        ax_title_right.text(0.5, 0.3, '✗ БЛОКИРУЕМЫЕ', 
                           ha='center', va='center',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor='white', edgecolor='black', linewidth=2))
        
        # --- ЗАПОЛНЯЕМ ЛЕВУЮ КОЛОНКУ (проходящие) ---
        for idx, case in enumerate(self.passing_cases):
            row = idx + 2
            ax = plt.subplot(gs[row, 0])
            
            self.draw_pattern(ax, case['vector'], case['name'], 
                            case['description'],
                            highlight_mismatches=None,
                            is_passing=True)
        
        # --- ЗАПОЛНЯЕМ ПРАВУЮ КОЛОНКУ (блокируемые) ---
        for idx, case in enumerate(self.blocked_cases):
            row = idx + 2
            ax = plt.subplot(gs[row, 2])
            
            mismatches = case.get('mismatch', [])
            
            self.draw_pattern(ax, case['vector'], case['name'], 
                            case['description'],
                            highlight_mismatches=mismatches,
                            is_passing=False)
            
            # Добавляем краткое пояснение для лишних нейронов
            if mismatches:
                mismatch_text = f"лишние: {','.join([str(i+1) for i in mismatches])}"
                ax.text(1, -0.45, mismatch_text,
                       ha='center', va='center',
                       fontsize=7, color='black', style='normal')
        
        # Скрываем неиспользуемые ячейки
        for row in range(len(self.passing_cases) + 2, n_rows + 2):
            ax = plt.subplot(gs[row, 0])
            ax.axis('off')
        
        for row in range(len(self.blocked_cases) + 2, n_rows + 2):
            ax = plt.subplot(gs[row, 2])
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Выводим краткую статистику
        self.print_statistics()
    
    def print_statistics(self):
        """Выводит краткую статистику"""
        print("\n" + "="*60)
        print(f"ИТОГО ДЛЯ БУКВЫ '{self.selected_letter}':")
        print(f"   Проходящих: {len(self.passing_cases)}")
        print(f"   Блокируемых: {len(self.blocked_cases)}")
        print("="*60)

# =============================================
# ЗАПУСК В GOOGLE COLAB
# =============================================

print("⚫" * 30)
print("  ЧЕРНО-БЕЛОЕ СРАВНЕНИЕ ДЛЯ ОДНОЙ БУКВЫ")
print("⚫" * 30)

# Создаем визуализатор
viz = SingleLetterBWComparison()

# Выбор буквы (раскомментируйте нужную строку)
# viz.selected_letter = 'T'
# viz.selected_letter = 'X'
# viz.selected_letter = 'C'
viz.etalon_vector = viz.etalons[viz.selected_letter]

print(f"\nАнализируем букву: {viz.selected_letter}")
print("(Чтобы изменить букву, измените selected_letter в коде)")

# Создаем рисунок
viz.create_figure()

print("\n✅ Черно-белый рисунок готов!")
print("   • Убрана таблица условных обозначений")
print("   • Увеличен размер рисунков")
print("   • Номера нейронов показаны маленькими цифрами")