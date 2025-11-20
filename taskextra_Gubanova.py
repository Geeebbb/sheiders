import taichi as ti
import math

ti.init(arch=ti.vulkan)
# Разрешение экрана
width, height = 800, 600
# Поле для хранения цвета каждого пикселя
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
# Глобальное время для анимации
t = ti.field(dtype=ti.f32, shape=())
t[None] = 0.0
# Количество элементов в системе
num_planets = 10

# --- Описание геометрии объектов ---
@ti.func
def sdf_sphere(p: ti.math.vec2, center: ti.math.vec2, radius: ti.f32) -> ti.f32:
    """Signed Distance Function для сферы"""
    return (p - center).norm() - radius

@ti.func
def sdf_ring(p: ti.math.vec2, center: ti.math.vec2, radius: ti.f32, thickness: ti.f32) -> ti.f32:
    """Signed Distance Function для кольцевидной планеты)"""
    d = (p - center).norm()
    return ti.abs(d - radius) - thickness

# --- f(x, t): Динамика орбиты планет ---
@ti.func
def orbit_path(x: ti.f32, t: ti.f32, planet_id: ti.i32) -> ti.f32:
    """Позиция планеты на орбите как функция от x и t"""
    freq = 0.5 + 0.1 * planet_id
    return 0.1 * ti.sin(freq * x + t) + 0.2 * planet_id

# --- Линейные преобразования пространства ---
@ti.func
def linear_transform(p: ti.math.vec2, angle: ti.f32) -> ti.math.vec2:
    """Линейное вращение вектора p на заданный угол"""
    c, s = ti.cos(angle), ti.sin(angle)
    return ti.math.mat2([[c, -s], [s, c]]) @ p

# --- Нелинейные преобразования пространства ---
@ti.func
def nonlinear_transform(p: ti.math.vec2, t: ti.f32) -> ti.math.vec2:
    """Нелинейное искажение координат на основе синуса and косинуса"""
    return p + 0.05 * ti.math.vec2(ti.sin(p.y + t), ti.cos(p.x + t))

# --- Сглаживание ---

@ti.func
def smooth_min(a: ti.f32, b: ti.f32, k: ti.f32) -> ti.f32:
    """Плавное объединение двух SDF"""
    h = ti.max(k - ti.abs(a - b), 0.0) / k
    return ti.min(a, b) - h * h * k * 0.25

@ti.func
def clamp(x: ti.f32, a: ti.f32, b: ti.f32) -> ti.f32:
    """Ограничение значения x в диапазоне [a, b]"""
    return ti.min(ti.max(x, a), b)

@ti.func
def smoothstep(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    """Плавный переход между двумя значениями"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# --- Смешивание цветов ---

@ti.func
def mix_colors(col1: ti.math.vec3, col2: ti.math.vec3, t: ti.f32) -> ti.math.vec3:
    """Смешивание двух цветов с коэффициентом t"""
    return col1 * (1.0 - t) + col2 * t

# --- Основная функция рендеринга ---

@ti.kernel
def render():
    """Рендеринг кадра сцены с планетами и орбитами"""
    for i, j in pixels:

        uv = ti.math.vec2(i / width, j / height) * 2.0 - 1.0
        uv.x *= width / height
        time = t[None]

        d = 1e10  # Начальное значение расстояния
        col = ti.math.vec3(0.0)  # Начальный цвет

        # Фон
        bg_col = mix_colors(
            ti.math.vec3(0.0, 0.0, 0.1),
            ti.math.vec3(0.1, 0.0, 0.2),
            ti.sin(time * 0.5)
        )
        col = bg_col

        # Рисуем все планеты
        for planet_id in range(num_planets):
            angle = time * (0.5 + 0.1 * planet_id)
            radius = 0.2 + 0.1 * planet_id
            center = ti.math.vec2(ti.cos(angle), ti.sin(angle)) * radius


            p = uv
            p = linear_transform(p, time * 0.1)
            p = nonlinear_transform(p, time)
            sdf = sdf_sphere(p, center, 0.05) if planet_id % 2 == 0 else sdf_ring(p, center, 0.05, 0.01)
            orbit_d = ti.abs(p.y - orbit_path(p.x, time, planet_id))
            sdf = smooth_min(sdf, orbit_d, 0.1)

            if sdf < d:
                d = sdf
                planet_col = ti.math.vec3(
                    0.5 + 0.5 * ti.sin(time + planet_id),
                    0.5 + 0.5 * ti.cos(time + planet_id * 0.5),
                    0.5
                )
                col = mix_colors(planet_col, bg_col, smoothstep(-0.01, 0.01, d))

        pixels[i, j] = col

# --- Основной цикл программы ---

def main():
    """Главная функция, запускающая графическое окно"""
    gui = ti.GUI("Planets", res=(width, height))
    while gui.running:
        t[None] += 0.03
        render()
        gui.set_image(pixels)
        gui.show()
        if gui.get_event(ti.ui.PRESS) and gui.event.key == ti.ui.ESCAPE:
            break
    gui.close()

if __name__ == "__main__":
    main()
