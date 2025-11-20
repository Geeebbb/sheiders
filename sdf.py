import taichi as ti
import taichi.math as tm

@ti.func
def sd_circle(p, r):
    """
    Вычисляет Signed Distance Function (SDF) от точки до окружности.

    :param p: (vec2) Координаты точки.
    :param r: (float) Радиус окружности.
    :return: (float) Расстояние до границы окружности (отрицательное — внутри).
    """
    return p.norm() - r


@ti.func
def sd_segment(p, a, b):
    """
    Вычисляет расстояние от точки до отрезка.

    :param p: (vec2) Точка, от которой измеряется расстояние.
    :param a: (vec2) Начальная точка отрезка.
    :param b: (vec2) Конечная точка отрезка.
    :return: (float) Расстояние от точки до ближайшей точки отрезка.
    """
    pa = p - a
    ba = b - a
    h = tm.clamp((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return (pa - ba * h).norm()


@ti.func
def sd_box(p, b):
    """
    Вычисляет SDF от точки до прямоугольника с центром в начале координат.

    :param p: (vec2) Точка, от которой измеряется расстояние.
    :param b: (vec2) Половина размеров прямоугольника
    :return: (float) Signed расстояние до границы прямоугольника.
    """
    d = abs(p) - b
    return ti.max(d, 0.).norm() + ti.min(ti.max(d.x, d.y), 0.0)


@ti.func
def sd_roundbox(p, b, r):
    """
    Вычисляет SDF от точки до скруглённого прямоугольника.

    :param p: (vec2) Точка, от которой измеряется расстояние.
    :param b: (vec2) Половинные размеры прямоугольника.
    :param r: (vec4) Радиусы скруглений:
              .xy — правый верх/низ,
              .zw — левый верх/низ.
    :return: (float) Расстояние до границы скруглённого прямоугольника.
    """
    rr = r.zw
    if p.x > 0.:
        rr = r.xy
    if p.y < 0.:
        rr.x = rr.y
    q = ti.abs(p) - b + rr[0]
    return ti.min(ti.max(q[0], q[1]), 0.) + ti.max(q, 0.0).norm() - rr[0]


@ti.func
def sd_trapezoid(p, r1, r2, he):
    """
    Вычисляет SDF от точки до равнобедренной трапеции.

    :param p: (vec2) Точка, от которой измеряется расстояние.
    :param r1: (float) Половина нижнего основания.
    :param r2: (float) Половина верхнего основания.
    :param he: (float) Высота трапеции.
    :return: (float) Signed расстояние до границы трапеции.
    """
    k1 = tm.vec2(r2, he)
    k2 = tm.vec2(r2 - r1, 2. * he)
    pp = tm.vec2(abs(p[0]), p[1])
    ca = tm.vec2(pp[0] - ti.min(pp[0], r1 if pp[1] < 0. else r2), ti.abs(pp[1]) - he)
    cb = pp - k1 + k2 * tm.clamp(((k1 - pp) @ k2) / (k2 @ k2), 0., 1.)
    s = -1. if cb[0] < 0. and ca[1] < 0. else 1.
    return s * ti.sqrt(ti.min(ca @ ca, cb @ cb))


@ti.func
def sd_arc(p, sc, ra, rb):
    """
    Вычисляет SDF от точки до дуги окружности с заданной толщиной.

    :param p: (vec2) Точка, от которой измеряется расстояние.
    :param sc: (vec2) Направляющий вектор (определяет дугу).
    :param ra: (float) Радиус окружности (от центра до дуги).
    :param rb: (float) Толщина дуги.
    :return: (float) Signed расстояние до границы дуги.
    """
    p.x = abs(p.x)
    return (tm.length(p - sc * ra) if sc.y * p.x > sc.x * p.y else abs(tm.length(p) - ra)) - rb
