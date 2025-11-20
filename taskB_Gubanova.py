import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, unrolling_limit=0)

# --- Параметры окна---
aspect_ratio = 1 / 1
res_x = 1280
res_y = int(res_x / aspect_ratio)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# --- Параметры анимации светлячков ---
NUM_FIREFLIES = 50  # Количество светлячков
FIREFLY_RADIUS = 0.008  # Базовый радиус светлячка
PULSE_SPEED = 5.0  # Скорость пульсации размера
MAX_PULSE_FACTOR = 1.5  # Максимальный коэффициент увеличения размера
MIN_PULSE_FACTOR = 0.5  # Минимальный коэффициент уменьшения размера
FIREFLY_SPEED = 0.05  # Базовая скорость движения светлячков
ATTRACTION_RADIUS = 0.2  # Радиус, в котором светлячки притягиваются/отталкиваются друг от друга
ATTRACTION_STRENGTH = 0.0005  # Сила притяжения/отталкивания между светлячками
TRAIL_FADE_SPEED = 0.95  # Скорость затухания следа от светлячков

# Определяем максимальный возможный радиус светлячка в пикселях
MAX_FIREFLY_PIXEL_RADIUS = int(FIREFLY_RADIUS * MAX_PULSE_FACTOR * res_x * 1.5) + 2

# --- Поля для хранения состояния светлячков ---
# firefly_pos: Позиция светлячка (x, y)
firefly_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_FIREFLIES)
# firefly_vel: Вектор скорости светлячка (vx, vy)
firefly_vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_FIREFLIES)
# firefly_color: Базовый цвет светлячка (r, g, b)
firefly_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_FIREFLIES)
# firefly_offset: Смещение
firefly_offset = ti.field(dtype=ti.f32, shape=NUM_FIREFLIES)


@ti.kernel
def init_fireflies():
    """
    Начальные позиции, скорости, цвета и смещения для всех светлячков.
    Каждый светлячок получает случайную начальную позицию, случайную нормализованную скорость
    (для случайного направления движения), случайный базовый цвет и случайное смещение
    """
    for i in range(NUM_FIREFLIES):

        firefly_pos[i] = tm.vec2(ti.random(), ti.random())
        # Случайное направление, нормализованное и умноженное на базовую скорость.
        firefly_vel[i] = tm.normalize(tm.vec2(ti.random(), ti.random()) * 2.0 - 1.0) * FIREFLY_SPEED
        # Случайный базовый цвет RGB для каждого светлячка.
        firefly_color[i] = tm.vec3(ti.random(), ti.random(), ti.random())
        # Случайное смещение для фазы пульсации
        firefly_offset[i] = ti.random() * tm.pi * 2.0


@ti.func
def sdf_circle(p, center, radius):
    """
    Возвращает расстояние от точки p до ближайшей точки на окружности с центром `center` и радиусом `radius`.
    Отрицательное значение означает, что точка p находится внутри круга.
    Позволяет эффективно определять, находится ли пиксель внутри светлячка
    """
    return tm.length(p - center) - radius


@ti.func
def smoothstep(edge0, edge1, x):
    """
    Создает плавный переход значения от 0 к 1, когда `x` переходит от `edge0` к `edge1`.
    Используется для сглаживания краев светлячков, создавая эффект мягкого свечения.
    """
    t = tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def smootherstep(edge0, edge1, x):
    """
    Предоставляет более плавный (кубический) переход от 0 к 1 между `edge0` и `edge1`.
    Используется для более естественного изменения силы взаимодействия между светлячками.
    """
    t = tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def mix_colors(color1, color2, alpha):
    """

    Смешивает два цвета `color1` и `color2` с заданным `alpha`-коэффициентом.
    Используется для изменения цвета светлячков во время пульсации и для общего смешивания.

    """
    return color1 * (1.0 - alpha) + color2 * alpha


@ti.kernel
def update_fireflies(t: ti.f32):
    """
    Обновляет позиции и скорости светлячков на каждом шаге времени t

    Включает следующие элементы:
    - Взаимодействие между светлячками:Притяжение/отталкивание в зависимости от расстояния
      Реализовано с использованием функции вида f(x) (зависимость от расстояния)
    - Случайное блуждание
    - Нормализация скорости
    - Отскок от границ
    """
    for i in range(NUM_FIREFLIES):
        current_pos = firefly_pos[i]
        attraction_force = tm.vec2(0.0, 0.0)
        for j in range(NUM_FIREFLIES):
            if i != j:
                other_pos = firefly_pos[j]
                dist_vec = other_pos - current_pos
                dist = tm.length(dist_vec)

                if dist < ATTRACTION_RADIUS and dist > 1e-5:
                    force_magnitude = smootherstep(0.0, ATTRACTION_RADIUS, dist) * ATTRACTION_STRENGTH

                    if dist < FIREFLY_RADIUS * 2.0:
                        force_magnitude = -ATTRACTION_STRENGTH * 2.0 / dist

                    attraction_force += tm.normalize(dist_vec) * force_magnitude

        firefly_vel[i] += attraction_force
        random_dir_change = (tm.vec2(ti.random(), ti.random()) * 2.0 - 1.0) * 0.005
        firefly_vel[i] += random_dir_change
        firefly_vel[i] = tm.normalize(firefly_vel[i]) * FIREFLY_SPEED
        firefly_pos[i] += firefly_vel[i]
        if firefly_pos[i].x < 0.0:
            firefly_vel[i].x *= -1.0
            firefly_pos[i].x = 0.0
        elif firefly_pos[i].x > 1.0:
            firefly_vel[i].x *= -1.0
            firefly_pos[i].x = 1.0

        if firefly_pos[i].y < 0.0:
            firefly_vel[i].y *= -1.0
            firefly_pos[i].y = 0.0
        elif firefly_pos[i].y > 1.0:
            firefly_vel[i].y *= -1.0
            firefly_pos[i].y = 1.0


@ti.kernel
def render(t: ti.f32):
    """
    Рендерит каждый пиксель на экране.

    """

    for i, j in pixels:
        pixels[i, j] *= TRAIL_FADE_SPEED

    for i_ff in range(NUM_FIREFLIES):
        ff_pos = firefly_pos[i_ff]
        ff_color = firefly_color[i_ff]
        pulse_factor = (tm.sin(t * PULSE_SPEED + firefly_offset[i_ff]) * 0.5 + 0.5)
        current_radius = FIREFLY_RADIUS * mix_colors(MIN_PULSE_FACTOR, MAX_PULSE_FACTOR, pulse_factor)
        pixel_x = int(ff_pos.x * res_x)
        pixel_y = int(ff_pos.y * res_y)

        for x_offset in range(-MAX_FIREFLY_PIXEL_RADIUS, MAX_FIREFLY_PIXEL_RADIUS + 1):
            for y_offset in range(-MAX_FIREFLY_PIXEL_RADIUS, MAX_FIREFLY_PIXEL_RADIUS + 1):
                current_pixel_x = pixel_x + x_offset
                current_pixel_y = pixel_y + y_offset

                if 0 <= current_pixel_x < res_x and 0 <= current_pixel_y < res_y:
                    p = tm.vec2(current_pixel_x / res_x, current_pixel_y / res_y)
                    dist = sdf_circle(p, ff_pos, current_radius)
                    if dist < current_radius * 1.0:
                        alpha_smooth = 1.0 - smoothstep(0.0, current_radius * 0.5, dist)
                        pulsing_color = mix_colors(ff_color, tm.vec3(1.0, 1.0, 1.0), pulse_factor * 0.7)
                        pixels[current_pixel_x, current_pixel_y] += pulsing_color * alpha_smooth * 0.5


# --- Основной цикл программы ---

init_fireflies()
gui = ti.GUI("Танцующие светлячки в летнюю ночь", res=(res_x, res_y))
frame_count = 0
while gui.running:
    t = gui.frame / 60.0
    update_fireflies(t)
    render(t)
    gui.set_image(pixels)
    gui.show()

    frame_count += 1