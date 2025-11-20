import taichi as ti
import taichi.math as tm
import time


@ti.data_oriented
class BaseShader:
    """
    Базовый класс для создания шейдеров с использованием Taichi
    Предоставляет основную структуру для инициализации окна,
    расчета пикселей и отрисовки изображения.
    """

    def __init__(self,
                 title: str,
                 res: tuple[int, int] | None = None,
                 gamma: float = 2.2
                 ):
        """
        Инициализирует базовый шейдер.

        Args:
            title (str): Заголовок окна приложения.
            res (tuple[int, int] | None): Разрешение окна в пикселях (ширина, высота).
                                         Если None, используется разрешение по умолчанию
            gamma (float): Значение гамма-коррекции для вывода изображения.

        """
        self.title = title
        self.res = res if res is not None else (1000, 563)
        self.resf = tm.vec2(*self.res)
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.res)
        self.gamma = gamma

    @ti.kernel
    def init(self):
        """
        Метод инициализации, который выполняется один раз перед началом основного цикла.

        """
        pass

    @ti.kernel
    def calculate(self, t: ti.f32):
        """
        Метод для выполнения расчетов, не связанных напрямую с отрисовкой каждого пикселя.
        Вызывается каждый кадр перед методом `render`.


        Args:
            t (ti.f32): Текущее время с момента запуска шейдера.
        """
        pass

    @ti.func
    def main_image(self, uv: tm.vec2, t: ti.f32) -> tm.vec3:
        """
        Основная функция шейдера, которая вычисляет цвет для каждого пикселя.


        Args:
            uv (tm.vec2): Нормализованные координаты пикселя в пространстве [-0.5*aspect_ratio, 0.5*aspect_ratio] по x
                          и [-0.5, 0.5] по y, где (0,0) - центр экрана.
            t (ti.f32): Текущее время с момента запуска шейдера.

        Returns:
            tm.vec3: Цветовой вектор RGB для данного пикселя.
        """
        col = tm.vec3(0.)
        col.rg = uv + 0.5
        return col

    @ti.kernel
    def render(self, t: ti.f32):
        """
        Ядро Taichi, отвечающее за рендеринг изображения.
        Перебирает каждый пиксель на экране, вычисляет его цвет,
        применяет гамма-коррекцию и сохраняет результат в поле `self.pixels`.

        Args:
            t (ti.f32): Текущее время с момента запуска шейдера.
        """
        for fragCoord in ti.grouped(self.pixels):
            uv = (fragCoord - 0.5 * self.resf) / self.resf.y
            col = self.main_image(uv, t)
            if self.gamma > 0.0:
                col = tm.clamp(col ** (1 / self.gamma), 0., 1.)
            self.pixels[fragCoord] = col

    def main_loop(self):
        """
        Запускает основной цикл приложения, который обрабатывает события,
        обновляет время, вызывает методы `calculate` и `render`,
        и отображает результат на экране.
        """
        gui = ti.GUI(self.title, res=self.res, fast_gui=True)
        start = time.time()

        self.init()
        while gui.running:
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == ti.GUI.ESCAPE:
                    break

            t = time.time() - start
            self.calculate(t)
            self.render(t)
            gui.set_image(self.pixels)
            gui.show()

        gui.close()


class TwoPassShader(BaseShader):
    """
    Класс, реализующий двухпроходный шейдер. Наследуется от BaseShader.

    """

    def __init__(self,
                 title: str,
                 res: tuple[int, int] | None = None,
                 gamma: float = 2.2
                 ):
        """
        Инициализирует двухпроходный шейдер.

        Args:
            title (str): Заголовок окна приложения.
            res (tuple[int, int] | None): Разрешение окна в пикселях (ширина, высота).
                                         Если None, используется разрешение по умолчанию.
            gamma (float): Значение гамма-коррекции для вывода изображения.
        """
        super().__init__(title, res=res, gamma=gamma)
        self.buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

    @ti.kernel
    def render_pass1(self, t: ti.f32):
        """
        Первый проход рендеринга.
        Вычисляет цвета для каждого пикселя и сохраняет их в промежуточный буфер `self.buffer`.
        Это тот же расчет, что и в `BaseShader.render`, но с выводом в другой буфер.

        Args:
            t (ti.f32): Текущее время с момента запуска шейдера.
        """
        for fragCoord in ti.grouped(self.buffer):
            uv = (fragCoord - 0.5 * self.resf) / self.resf.y
            col = self.main_image(uv, t)
            if self.gamma > 0.0:
                col = tm.clamp(col ** (1 / self.gamma), 0., 1.)
            self.buffer[fragCoord] = col

    @ti.kernel
    def render_pass2(self, t: ti.f32):
        """
        Второй проход рендеринга.
        Считывает данные из промежуточного буфера `self.buffer` и записывает их в конечный
        буфер пикселей `self.pixels`, который затем отображается на экране.

        Args:
            t (ti.f32): Текущее время с момента запуска шейдера. (В данном случае не используется напрямую,
                        но передается для совместимости сигнатур).
        """
        for fragCoord in ti.grouped(self.pixels):
            # Берем цвет из буфера pass1, округляя координаты до кратных 16,
            # чтобы создать эффект блочной пикселизации.
            col = self.buffer[fragCoord // 16 * 16]
            self.pixels[fragCoord] = col

    def render(self, t: ti.f32):
        """
        Переопределенный метод рендеринга, который вызывает оба прохода:
        сначала `render_pass1`, затем `render_pass2`.

        Args:
            t (ti.f32): Текущее время с момента запуска шейдера.
        """
        self.render_pass1(t)
        self.render_pass2(t)

    def main_loop(self):
        """
        Переопределенный основной цикл для двухпроходного шейдера.
        """
        gui = ti.GUI(self.title, res=self.res, fast_gui=True)
        start = time.time()
        show_buffer = False

        while gui.running:
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == ti.GUI.ESCAPE:
                    break
                elif gui.event.key == ti.GUI.RETURN:
                    show_buffer = not show_buffer

            t = time.time() - start
            self.render(t)
            if show_buffer:
                gui.set_image(self.buffer)
            else:
                gui.set_image(self.pixels)
            gui.show()

        gui.close()


if __name__ == "__main__":
    ti.init(arch=ti.opengl)
    shader = BaseShader("Базовый шейдер")
    shader.main_loop()
