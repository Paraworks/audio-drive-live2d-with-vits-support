import time
import math
import random
import logging
import functools
import numpy as np
import yaml
import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from rimo_utils import matrix
from rimo_utils import 计时
import psd_tools
import 现实
import argparse
Vtuber尺寸 = 720, 720

parser = argparse.ArgumentParser(
    description=
    '一些必须的文件')
parser.add_argument('--psd',
                    type=str,
                    help='拆分好的psd文件',
                    default = '../res/莉沫酱较简单版.psd')
parser.add_argument('--y1',
                    type=str,
                    help='必须要用到的两个文件之一.',
                    default='../res/变形.yaml')
parser.add_argument('--y2',
                    type=str,
                    help='必须要用到的两个文件之一',
                    default = '../res/信息.yaml')
args = parser.parse_args()


def 相位转移(x):
    if x is None:
        return x
    if type(x) is str:
        return 相位转移(eval(x))
    if type(x) in [int, float]:
        return np.array([[x, x], [x, x]])
    else:
        return np.array(x)


class 图层类:
    def __init__(self, 名字, bbox, z, 物理, npdata):
        self.名字 = 名字
        self.npdata = npdata
        self.纹理编号, 纹理座标 = self.生成opengl纹理()
        self.变形 = []
        self.物理 = 相位转移(物理)
        深度 = 相位转移(z)
        assert len(深度.shape) == 2
        self.shape = 深度.shape

        q, w = 纹理座标
        a, b, c, d = bbox
        [[p1, p2],
         [p4, p3]] = np.array([
             [[a, b, 0, 1, 0, 0, 0, 1], [a, d, 0, 1, w, 0, 0, 1]],
             [[c, b, 0, 1, 0, q, 0, 1], [c, d, 0, 1, w, q, 0, 1]],
         ])
        x, y = self.shape
        self.顶点组 = np.zeros(shape=[x, y, 8])
        for i in range(x):
            for j in range(y):
                self.顶点组[i, j] = p1 + (p4-p1)*i/(x-1) + (p2-p1)*j/(y-1)
                self.顶点组[i, j, 2] = 深度[i, j]

    def 生成opengl纹理(self):
        w, h = self.npdata.shape[:2]
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        纹理 = np.zeros([d, d, 4], dtype=self.npdata.dtype)
        纹理[:, :, :3] = 255
        纹理[:w, :h] = self.npdata
        纹理座标 = (w / d, h / d)

        width, height = 纹理.shape[:2]
        纹理编号 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, 纹理编号)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, 纹理)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)

        return 纹理编号, 纹理座标

    def 顶点组导出(self):
        return self.顶点组.copy()


class vtuber:
    def __init__(self, psd路径, 切取范围=(1024, 1024), 信息路径='信息.yaml', 变形路径='变形.yaml'):
        psd = psd_tools.PSDImage.open(psd路径)
        with open(args.y2, encoding='utf8') as f:
            信息 = yaml.safe_load(f)
        with open(args.y1, encoding='utf8') as f:
            self.变形组 = yaml.safe_load(f)
        
        def 再装填():
            while True:
                time.sleep(1)
                try:
                    with open("C:/Programming/Vtuber_Tutorial/7/变形.yaml", encoding='utf8') as f:
                        self.变形组 = yaml.safe_load(f)
                except Exception as e:
                    logging.exception(e)
        import threading
        t = threading.Thread(target=再装填)
        t.setDaemon(True)
        t.start()

        self.所有图层 = []
        self.psd尺寸 = psd.size
        self.切取范围 = 切取范围

        def dfs(图层, path=''):
            if 图层.is_group():
                for i in 图层:
                    dfs(i, path + 图层.name + '/')
            else:
                名字 = path+图层.name
                if 名字 not in 信息:
                    logging.warning(f'图层「{名字}」找不到信息，丢了！')
                    return
                a, b, c, d = 图层.bbox
                npdata = 图层.numpy()
                npdata[:, :, :3] = npdata[:, :, :3][:, :, ::-1]
                self.所有图层.append(图层类(
                    名字=名字,
                    z=信息[名字]['深度'],
                    物理=信息[名字].get('物理'),
                    bbox=(b, a, d, c),
                    npdata=npdata
                ))
        for 图层 in psd:
            dfs(图层)
        self.截图 = None
        self.启用截图 = False
        self._记忆 = {}
    
    def 获取截图(self, 反转颜色=True):
        while True:
            self.启用截图 = True
            if self.截图:
                img = np.frombuffer(self.截图, dtype=np.uint8).reshape((*Vtuber尺寸, 4)).copy()
                if 反转颜色:
                    img[:, :, :3] = img[:, :, :3][:, :, ::-1]
                img = img[::-1]
                return img
            time.sleep(0.01)

    def 附加变形(self, 变形名, 图层名, a, b, f):
        变形 = self.变形组[变形名]
        if 图层名 not in 变形:
            return a, b
        if '位置' in 变形[图层名]:
            d = 变形[图层名]['位置']
            if type(d) is str:
                d = eval(d)
            d = np.array(d)
            a[:, :2] += d.reshape(a.shape[0], 2) * f
        return a, b

    def 多重附加变形(self, 变形组, 图层名, a, b):
        for 变形名, 强度 in 变形组:
            a, b = self.附加变形(变形名, 图层名, a, b, 强度)
        return a, b

    def 动(self, 图层, t):
        if 图层.物理 is None:
            return t
        res = t
        q, 上次时间 = self._记忆.get(id(图层), (None, 0))
        现在时间 = time.time()
        if q is not None:
            时间差 = min(0.1, 现在时间-上次时间)
            物理缩小 = 0.05
            w = 图层.物理.reshape(t.shape[0], 1)
            w = w * 物理缩小 + 1 * (1-物理缩小)
            ww = -((1-w)**时间差)+1
            v = t - q
            res = q + v * ww
        self._记忆[id(图层)] = res, 现在时间
        return res

    def opengl绘图循环(self, window, 数据源, line_box=False):
        def 没有状态但是却能均匀变化的随机数(范围=(0, 1), 速度=1):
            now = time.time()*速度
            a, b = int(now), int(now)+1
            random.seed(a)
            f0 = random.random()
            random.seed(b)
            f1 = random.random()
            f = f0 * (b-now) + f1 * (now-a)
            return 范围[0] + (范围[1]-范围[0])*f
        
        def 锚击(x, a, b):
            x = sorted([x, a, b])[1]
            return (x-a)/(b-a)

        @functools.lru_cache(maxsize=16)
        def model(xz, zy, xy, 脸大小, x偏移, y偏移):
            model_p = \
                matrix.translate(0, 0, -0.9) @ \
                matrix.rotate_ax(xz, axis=(0, 2)) @ \
                matrix.rotate_ax(zy, axis=(2, 1)) @ \
                matrix.translate(0, 0.9, 0.9) @ \
                matrix.rotate_ax(xy, axis=(0, 1)) @ \
                matrix.translate(0, -0.9, 0) @ \
                matrix.perspective(999)
            f = 750/(800-脸大小)
            extra = matrix.translate(x偏移*0.6, -y偏移*0.8, 0) @ \
                    matrix.scale(f, f, 1)
            return model_p, extra

        model_g = \
            matrix.scale(2 / self.切取范围[0], 2 / self.切取范围[1], 1) @ \
            matrix.translate(-1, -1, 0) @ \
            matrix.rotate_ax(-math.pi / 2, axis=(0, 1))

        def draw(图层):
            源 = 图层.顶点组导出()
            x, y, _ = 源.shape

            所有顶点 = 源.reshape(x*y, 8)

            a, b = 所有顶点[:, :4], 所有顶点[:, 4:]
            a = a @ model_g
            z = a[:, 2:3]
            z -= 0.1
            a[:, :2] *= z
            眼睛左右 = 横旋转量*4 + 没有状态但是却能均匀变化的随机数((-0.2, 0.2), 速度=1.6)
            眼睛上下 = 竖旋转量*7 + 没有状态但是却能均匀变化的随机数((-0.1, 0.1), 速度=2)
            闭眼强度 = 锚击(左眼大小+右眼大小, -0.001, -0.008)
            眉上度 = 锚击(左眉高+右眉高, -0.03, 0.01) - 闭眼强度*0.1
            闭嘴强度 = 锚击(嘴大小, 0.05, 0) * 1.1 - 0.1
            a, b = self.多重附加变形([
                ['永远', 1],
                ['眉上', 眉上度],
                ['左眼远离', 眼睛左右],
                ['右眼远离', -眼睛左右],
                ['左眼上', 眼睛上下],
                ['右眼上', 眼睛上下],
                ['左眼闭', 闭眼强度],
                ['右眼闭', 闭眼强度],
                ['闭嘴', 闭嘴强度],
            ], 图层.名字, a, b)

            xz = 横旋转量 / 1.2
            zy = 竖旋转量 / 1.4
            xy = Z旋转量 / 5
            if not 图层.名字.startswith('头/'):
                xz /= 8
                zy = 0

            model_p, extra = model(xz, zy, xy, 脸大小, x偏移, y偏移)
            a = a @ model_p
            a = self.动(图层, a)
            a = a @ extra

            b *= z

            所有顶点 = np.concatenate([a, b], axis=1).reshape([x, y, 8])

            glBegin(GL_QUADS)
            for i in range(x-1):
                for j in range(y-1):
                    for p in [所有顶点[i, j], 所有顶点[i, j+1], 所有顶点[i+1, j+1], 所有顶点[i+1, j]]:
                        glTexCoord4f(*p[4:])
                        glVertex4f(*p[:4])
            glEnd()

        while not glfw.window_should_close(window):
            with 计时.帧率计('绘图'):
                glfw.poll_events()
                glClearColor(0, 0, 0, 0)
                glClear(GL_COLOR_BUFFER_BIT)
                横旋转量, 竖旋转量, Z旋转量, y偏移, x偏移, 嘴大小, 脸大小, 左眼大小, 右眼大小, 左眉高, 右眉高 = 数据源()
                for 图层 in self.所有图层:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, 图层.纹理编号)
                    glColor4f(1, 1, 1, 1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    draw(图层)
                    if line_box:
                        glDisable(GL_TEXTURE_2D)
                        glColor4f(0.3, 0.3, 1, 0.2)
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                        draw(图层)
                glfw.swap_buffers(window)
                if self.启用截图:
                    glReadBuffer(GL_FRONT)
                    self.截图 = glReadPixels(0, 0, *Vtuber尺寸, GL_RGBA, GL_UNSIGNED_BYTE)


缓冲特征 = None


def 特征缓冲(缓冲比例=0.8):
    global 缓冲特征
    新特征 = 现实.获取特征组()
    if 缓冲特征 is None:
        缓冲特征 = 新特征
    else:
        缓冲特征 = 缓冲特征 * 缓冲比例 + 新特征 * (1 - 缓冲比例)
    return 缓冲特征


def init_window():
    def 超融合():
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
        glfw.window_hint(glfw.FLOATING, True)
    glfw.init()
    超融合()
    glfw.window_hint(glfw.SAMPLES, 4)
    # glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*Vtuber尺寸, 'Vtuber', None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - Vtuber尺寸[0], monitor_size.height - Vtuber尺寸[1])
    glViewport(0, 0, *Vtuber尺寸)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_FRONT)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    return window


if __name__ == "__main__":
    现实.启动()
    window = init_window()
    莉沫酱 = vtuber(args.psd)
    import sys
    sys.path.append('..')
    from utils_live2d import 虚拟摄像头 
    虚拟摄像头.start(莉沫酱, (1280, 720))
    莉沫酱.opengl绘图循环(window, 数据源=特征缓冲)
