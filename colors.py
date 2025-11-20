import taichi as ti
import taichi.math as tm

# Базовые цветовые константы
red = tm.vec3(1., 0., 0.)  # красный
green = tm.vec3(0., 1., 0.)  #  зеленый
blue = tm.vec3(0., 0., 1.)  #  синий
black = tm.vec3(0.)  # Черный
white = tm.vec3(1.)  # Белый


@ti.func
def hue_gradient(t: ti.f32) -> tm.vec3:
    """
    Генерирует плавный радужный градиент, проходящий через все цвета палитры.
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB цвета в заданной позиции градиента
    """
    p = ti.abs(tm.fract(t + tm.vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0)
    return tm.clamp(p - 1.0, 0.0, 1.0)


@ti.func
def tech_gradient(t: ti.f32) -> tm.vec3:
    """
    градиент с резкими цветовыми переходами
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB цвета
    """
    return ti.pow(tm.vec3(t + 0.01), tm.vec3(120.0, 10.0, 180.0))


@ti.func
def fire_gradient(t: ti.f32) -> tm.vec3:
    """
    Градиент, имитирующий цвета пламени
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB цвета огня
    """
    return ti.max(
        ti.pow(tm.vec3(ti.min(t * 1.02, 1.0)), tm.vec3(1.7, 25.0, 100.0)),
        tm.vec3(0.06 * pow(max(1.0 - abs(t - 0.35), 0.0), 5.0))
    )


@ti.func
def desert_gradient(t: ti.f32) -> tm.vec3:
    """
    Градиент
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0

    Возвращает:
        tm.vec3: Вектор RGB цвета
    """
    s = ti.sqrt(tm.clamp(1.0 - (t - 0.4) / 0.6, 0.0, 1.0))
    sky = ti.sqrt(tm.mix(tm.vec3(1.0), tm.vec3(0.0, 0.8, 1.0), tm.smoothstep(t, 0.4, 0.9)) * tm.vec3(s, s, 1.0))
    land = tm.mix(tm.vec3(0.7, 0.3, 0.0), tm.vec3(0.85, 0.75 + ti.max(0.8 - t * 20.0, 0.0), 0.5), (t / 0.4) ** 2)
    return tm.clamp(sky if t > 0.4 else land, 0.0, 1.0) * tm.clamp(1.5 * (1.0 - ti.abs(t - 0.4)), 0.0, 1.0)


@ti.func
def electric_gradient(t: ti.f32) -> tm.vec3:
    """
    Электрический градиент
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0

    Возвращает:
        tm.vec3: Вектор RGB
    """
    return tm.clamp(tm.vec3(t * 8.0 - 6.3, tm.smoothstep(t, 0.6, 0.9) ** 2, ti.pow(t, 3.0) * 1.7), 0.0, 1.0)


@ti.func
def neon_gradient(t: ti.f32) -> tm.vec3:
    """
    Неоновый градиент с яркими насыщенными цветами
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB
    """
    return tm.clamp(tm.vec3(t * 1.3 + 0.1, (abs(0.43 - t) * 1.7) ** 2, (1.0 - t) * 1.7), 0.0, 1.0)


@ti.func
def heatmap_gradient(t: ti.f32) -> tm.vec3:
    """
    Градиент тепловой карты
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB
    """
    return tm.clamp(
        (pow(t, 1.5) * 0.8 + 0.2) * tm.vec3(
            tm.smoothstep(t, 0.0, 0.35) + t * 0.5,
            tm.smoothstep(t, 0.5, 1.0),
            ti.max(1.0 - t * 1.7, t * 7.0 - 6.0)
        ),
        0.0, 1.0
    )


@ti.func
def rainbow_gradient(t: ti.f32) -> tm.vec3:
    """
    радужный градиент
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB радуги
    """
    c = 1.0 - ti.pow(ti.abs(tm.vec3(t) - tm.vec3(0.65, 0.5, 0.2)) * tm.vec3(3.0, 3.0, 5.0), tm.vec3(1.5, 1.3, 1.7))
    c.r = max((0.15 - (ti.abs(t - 0.04) * 5.0) ** 2), c.r)
    c.g = tm.smoothstep(t, 0.04, 0.45) if t < 0.5 else c.g
    return tm.clamp(c, 0.0, 1.0)


@ti.func
def brightness_gradient(t: ti.f32) -> tm.vec3:
    """
    Простой градиент яркости от черного к белому
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB
    """
    return tm.vec3(t * t)


@ti.func
def grayscale_gradient(t: ti.f32) -> tm.vec3:
    """
    Линейный градиент яркости от черного к белому
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0

    Возвращает:
        tm.vec3: Вектор RGB
    """
    return tm.vec3(t)


@ti.func
def stripe_gradient(t: ti.f32) -> tm.vec3:
    """
    Полосатый градиент с чередующимися светлыми и темными полосами
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0
    Возвращает:
        tm.vec3: Вектор RGB
    """
    return tm.vec3(ti.floor(t * 32.0) % 2.0) * 0.2 + 0.8


@ti.func
def ansi_gradient(t: ti.f32) -> tm.vec3:
    """
    Градиент
    Параметры:
        t (ti.f32): Позиция в градиенте от 0.0 до 1.0

    Возвращает:
        tm.vec3: Вектор RGB
    """
    return ti.floor(t * tm.vec3(8.0, 4.0, 2.0)) % 2.0


@ti.func
def show_all_gradientm(fragCoord: tm.vec2) -> tm.vec3:
    """
    Отображает все градиенты вертикально в одном окне
    Параметры:
        fragCoord (tm.vec2): Координаты текущего пикселя
    Возвращает:
        tm.vec3: Вектор RGB цвета для текущего пикселя
    """
    num_palettes = 12.0
    x = fragCoord.x / resf.x
    i = num_palettes * fragCoord.y / resf.y

    col = tm.vec3(0.0)
    if fragCoord.y % (resf.y / num_palettes) < ti.max(resf.y / 100.0, 3.0):
        col = tm.vec3(0.0)
    elif i > 11.0:
        col = hue_gradient(x)
    elif i > 10.0:
        col = tech_gradient(x)
    elif i > 9.0:
        col = fire_gradient(x)
    elif i > 8.0:
        col = desert_gradient(x)
    elif i > 7.0:
        col = electric_gradient(x)
    elif i > 6.0:
        col = neon_gradient(x)
    elif i > 5.0:
        col = heatmap_gradient(x)
    elif i > 4.0:
        col = rainbow_gradient(x)
    elif i > 3.0:
        col = brightness_gradient(x)
    elif i > 2.0:
        col = grayscale_gradient(x)
    elif i > 1.0:
        col = stripe_gradient(x)
    else:
        col = ansi_gradient(x)

    return tm.clamp(col ** (1 / 2.2), 0., 1.)


@ti.kernel
def render(t: ti.f32):
    """
    Основная функция рендеринга, выполняемая на GPU.
    Вычисляет цвет каждого пикселя параллельно.
    Параметры:
        t (ti.f32): Время в секундах с момента запуска программы
    """
    for fragCoord in ti.grouped(pixels):
        col = show_all_gradientm(fragCoord)
        pixels[fragCoord] = col


if __name__ == "__main__":
    import time

    # Инициализация Taichi с использованием GPU
    ti.init(arch=ti.gpu)

    asp = 16 / 9
    h = 600
    w = int(asp * h)
    res = w, h
    resf = tm.vec2(float(w), float(h))

    pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
    gui = ti.GUI("Генератор цветовых градиентов", res=res, fast_gui=True)
    start = time.time()
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
        t = time.time() - start
        render(t)
        gui.set_image(pixels)
        gui.show()
    gui.close()