import taichi as ti
import taichi.math as tm
from gui import BaseShader
from colors import black
from core import hash22
from sdf import sd_circle

"""
Декомпозиция анимации "Падающие окружности с  пульсацией самих окружностей и  внутренних кругов в них"

1. Элементы анимации:
   a. Вертикальные линии/колонки:
      - Анимация представляет собой набор вертикальных "рельс", по которым движутся окружности вниз
      - Каждая колонка (определяемая column_id = tm.floor(uv_scaled.x)) имеет свою уникальную скорость падения и начальное смещение, это нужно для асинхронного движения
      - Скорость колонки current_column_speed вычисляется на основе хэша hash22(tm.vec2(column_id, 1.0)).x, это как раз и отвечает за то, что они с разной скоростью падают вниз
      - Начальное смещение column_start_offset , чтоб старт у всех был разный

   b. Падающие окружности:
      - Это как раз сами кольца, движущиеся вниз по своим колонкам
      - Расположение окружности в виртуальной сетке (current_id = tm.vec2(column_id, tm.floor(virtual_y_in_grid))) определяется с учетом времени t, скорости колонки и начального смещения
      - Каждая окружность со своим хэш h = hash22(current_id), используемый для определения ее свойств

   c. Пульсация внешнего кольца:
      - Радиус внешнего кольца animated_radius_outer анимируется с помощью синусоидальной функции:
        self.ring_radius + tm.sin(t * self.pulse_frequency + h.y * tm.pi * 2.0) * self.pulse_amplitude
      - Фаза пульсации зависит от h.y, что делает пульсацию уникальной для каждой окружности
      - Функция пульсации на graphtoy.com: sin(x * A + B) * C
        Ссылка: [https://graphtoy.com/?f1(x,t)=sin(x*7)*0.04]
   d. Пульсирующие внутренние круги:
      - Некоторые окружности (с вероятностью self.inner_circle_probability) содержат круг внутри себя
      - Радиус внутреннего круга `animated_radius_inner` также пульсирует, но с другой частотой и амплитудой:
        base_inner_radius + tm.sin(t * self.inner_pulse_frequency + h.x * tm.pi * 2.0) * self.inner_pulse_amplitude
      - Фаза пульсации внутреннего круга зависит от h.x
      - Функция пульсации
        Ссылка: [https://graphtoy.com/?f1(x,t)=sin(x*8)*0.03]
   e. Цвета:
      - Цвета внешнего и внутреннего кругов генерируются на основе хэшей h.x и h.y, это и дает уникальные цвета для каждой окружности
      - Цвет внутреннего круга смешивается с цветом внешнего

2. Взаимосвязи:
   - Все элементы анимации привязаны к  "виртуальной сетки", каждая ячейка - место для окружности
   - Скорость падения и начальное смещение определяются на уровне колонки
   - Свойства отдельных окружностей (размер, наличие внутреннего круга) определяются уникальным хэшем current_id этой окружности
   - Пульсация внешнего кольца и внутреннего круга происходит независимо 

3. Реализация:
   - BaseShader: для базовой функциональности шейдера 
   - main_image: Для реализации логики отрисовки каждого пикселя
   - Масштабирование: uv_scaled = uv * self.grid_scale преобразует координаты пикселя в координаты сетки
   -column_id: используется как идентификатор колонки
   -hash22: Функция хэширования используется для генерации псевдослучайных значений для скорости колонок, начальных смещений и окружностей
   -sd_circle: Функция  используется для расчета расстояния до окружности
   -smoothstep: Функция используется для сглаживания краев окружностей
   - tm.mix: Функция для смешивания цветов
   - Параметры класса: Все настраиваемые параметры анимации (`grid_scale`, `speed`, `ring_radius`, `pulse_amplitude` и т.д.) вынесены в `__init__`
"""


class FallingRingsShader(BaseShader):
    """
    Класс FallingRingsShader реализует анимацию падающих окружностей
    Каждая окружность плывет строго вниз по вертикальным линиям, меняет свой размер
    каждая вертикальная линия имеет свою скорость, и некоторые окружности
    имеют внутренние круги
    унаследование от BaseShader
    """

    def __init__(self, title: str, res: tuple[int, int] = (800, 800)):
        """
        Инициализирует шейдер

        Args:
            title (str): Заголовок окна шейдера
            res (tuple[int, int]): Разрешение окна
        """
        super().__init__(title, res=res)
        self.grid_scale = tm.vec2(20.0, 20.0)  # Размер сетки
        self.speed = 1.0  # Обычная скорость падения
        self.speed_variation = 0.9  # Максимальное отклонение от обычной скорости
        self.ring_radius = 0.35  # радиус внешней окружности
        self.line_width = 0.03  # Толщина линии окружности
        self.softness = 0.01  # Мягкость краев

        self.pulse_amplitude = 0.04  # насколько сильно меняется радиус
        self.pulse_frequency = 7.0  # как быстро меняется размер

        # ---- Параметры для внутреннего КРУГА
        self.inner_circle_radius_factor = 0.4  # Радиус внутреннего круга
        self.inner_circle_probability = 0.4  # Вероятность появления внутреннего круга
        self.inner_color_mix_factor = 0.5

        # ---- ПАРАМЕТРЫ ДЛЯ КРУГА ----
        self.inner_pulse_amplitude = 0.03  # Амплитуда пульсации внутреннего круга
        self.inner_pulse_frequency = 8.0  # Частота пульсации внутреннего круга
        # -------------------------------------------------------

    @ti.func
    def main_image(self, uv: tm.vec2, t: ti.f32) -> tm.vec3:
        """
        Основная функция шейдера, которая вычисляет цвет для каждого пикселя


        Args:
            uv (tm.vec2): Нормализованные координаты пикселя
            t (ti.f32): Текущее время анимации

        Returns:
            tm.vec3: Цветовой вектор для данного пикселя.
        """

        uv_scaled = uv * self.grid_scale
        column_id = tm.floor(uv_scaled.x)
        column_speed_hash = hash22(tm.vec2(column_id, 1.0)).x
        speed_multiplier = 1.0 - self.speed_variation + column_speed_hash * (2.0 * self.speed_variation)
        current_column_speed = self.speed * speed_multiplier
        column_start_offset = hash22(tm.vec2(column_id, 0.0)).x * 100.0
        virtual_y_in_grid = uv_scaled.y + t * current_column_speed + column_start_offset
        current_id = tm.vec2(column_id, tm.floor(virtual_y_in_grid))
        p = tm.vec2(tm.fract(uv_scaled.x) - 0.5, tm.fract(virtual_y_in_grid) - 0.5)
        h = hash22(current_id)

        # --- Расчет для ВНЕШНЕЙ окружности ---
        animated_radius_outer = self.ring_radius + tm.sin(
            t * self.pulse_frequency + h.y * tm.pi * 2.0) * self.pulse_amplitude
        animated_radius_outer = tm.max(animated_radius_outer, self.line_width * 1.5)
        outer_color = tm.vec3(tm.fract(h.x * 2.3), tm.fract(h.y * 3.4), tm.fract((h.x + h.y) * 4.5))
        outer_color = tm.mix(outer_color, tm.vec3(1.0) - outer_color, 0.2)

        d_outer = abs(sd_circle(p, animated_radius_outer)) - self.line_width
        alpha_outer = tm.smoothstep(0.0, self.softness, -d_outer)
        final_color = black
        final_color = tm.mix(final_color, outer_color, alpha_outer)

        # --- Расчет для ВНУТРЕННЕГО КРУГА  ---
        if tm.fract(h.x + h.y * 7.1) < self.inner_circle_probability:
            base_inner_radius = self.ring_radius * self.inner_circle_radius_factor

            # Анимированный радиус внутреннего круга
            animated_radius_inner = base_inner_radius + tm.sin(
                t * self.inner_pulse_frequency + h.x * tm.pi * 2.0) * self.inner_pulse_amplitude
            animated_radius_inner = tm.max(animated_radius_inner, 0.005)
            inner_color = tm.vec3(tm.fract(h.y * 5.6), tm.fract(h.x * 6.7), tm.fract((h.x - h.y) * 7.8))
            inner_color = tm.mix(inner_color, tm.vec3(1.0) - inner_color, 0.2)
            inner_color = tm.mix(inner_color, outer_color, self.inner_color_mix_factor)


            d_inner = sd_circle(p, animated_radius_inner)
            alpha_inner = tm.smoothstep(0.0, self.softness, -d_inner)
            final_color = tm.mix(final_color, inner_color, alpha_inner)

        return final_color


if __name__ == "__main__":
    ti.init(arch=ti.opengl)
    shader = FallingRingsShader("Падающие окружности с независимой пульсацией внутренних кругов")
    shader.main_loop()