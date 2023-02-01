import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import re
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import unicodedata
from scipy.io.wavfile import write
import logging
import threading
import multiprocessing
import cv2
import dlib
import numpy as np
from rimo_utils import 计时
import argparse
import paddle
from ppgan.apps.wav2lip_predictor import Wav2LipPredictor
import winsound
detector = dlib.get_frontal_face_detector()
parser = argparse.ArgumentParser(
    description=
    'Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--audio',
                    type=str,
                    help='中途生成的语音存储路径',
                    default = '../res/audio.wav')
parser.add_argument('--face',
                    type=str,
                    help='中之人的照片，用于talking face',
                    default = '../res/stdface.jpg')
parser.add_argument('--outfile',
                    type=str,
                    help='视频路径，动作捕捉的替代品',
                    default='../res/result_voice.mp4')
parser.add_argument('--predictor',
                    type=str,
                    help='Filepath of res/shape_predictor_68_face_landmarks.dat',
                    default = '../res/shape_predictor_68_face_landmarks.dat')
parser.add_argument('--config',
                    type=str,
                    help='Config file of vits model.',
                    default='../res/config.json')
parser.add_argument('--model',
                    type=str,
                    help='Checkpoint file of vits model.',
                    default='../res/G_205000.pth')
parser.add_argument('--texture',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/conversation.txt')
parser.add_argument('--waitVideo',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/Masahiro.mp4')
paddle.set_device('gpu')
args = parser.parse_args()                        
pipe = multiprocessing.Pipe()
dev = torch.device("cuda:0")
hps_ms = utils.get_hparams_from_file(args.config)
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model).to(dev)
_ = net_g_ms.eval()
_ = utils.load_checkpoint(args.model, net_g_ms, None)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

def 多边形面积(a):
    a = np.array(a)
    x = a[:, 0]
    y = a[:, 1]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def 人脸定位(img):
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))


predictor = dlib.shape_predictor(args.predictor)


def 提取关键点(img, 脸位置):
    landmark_shape = predictor(img, 脸位置)
    关键点 = []
    for i in range(68):
        pos = landmark_shape.part(i)
        关键点.append(np.array([pos.x, pos.y], dtype=np.float32))
    return np.array(关键点)


def 计算旋转量(关键点):
    def 中心(索引数组):
        return sum([关键点[i] for i in 索引数组]) / len(索引数组)
    左眉 = [18, 19, 20, 21]
    右眉 = [22, 23, 24, 25]
    下巴 = [6, 7, 8, 9, 10]
    鼻子 = [29, 30]
    眉中心, 下巴中心, 鼻子中心 = 中心(左眉 + 右眉), 中心(下巴), 中心(鼻子)
    中线 = 眉中心 - 下巴中心
    斜边 = 眉中心 - 鼻子中心
    中线长 = np.linalg.norm(中线)
    横旋转量 = np.cross(中线, 斜边) / 中线长**2
    竖旋转量 = 中线 @ 斜边 / 中线长**2
    Z旋转量 = np.cross(中线, [0, 1]) / 中线长
    return np.array([横旋转量, 竖旋转量, Z旋转量])


def 计算嘴大小(关键点):
    边缘 = 关键点[0:17]
    嘴边缘 = 关键点[48:60]
    嘴大小 = 多边形面积(嘴边缘) / 多边形面积(边缘)
    return np.array([嘴大小])


def 计算相对位置(img, 脸位置):
    x = (脸位置.top() + 脸位置.bottom())/2/img.shape[0]
    y = (脸位置.left() + 脸位置.right())/2/img.shape[1]
    y = 1 - y
    相对位置 = np.array([x, y])
    return 相对位置


def 计算脸大小(关键点):
    边缘 = 关键点[0:17]
    t = 多边形面积(边缘)**0.5
    return np.array([t])


def 计算眼睛大小(关键点):
    边缘 = 关键点[0:17]
    左 = 多边形面积(关键点[36:42]) / 多边形面积(边缘)
    右 = 多边形面积(关键点[42:48]) / 多边形面积(边缘)
    return np.array([左, 右])


def 计算眉毛高度(关键点): 
    边缘 = 关键点[0:17]
    左 = 多边形面积([*关键点[18:22]]+[关键点[38], 关键点[37]]) / 多边形面积(边缘)
    右 = 多边形面积([*关键点[22:26]]+[关键点[44], 关键点[43]]) / 多边形面积(边缘)
    return np.array([左, 右])


def 提取图片特征(img):
    脸位置 = 人脸定位(img)
    if not 脸位置:
        return None
    相对位置 = 计算相对位置(img, 脸位置)
    关键点 = 提取关键点(img, 脸位置)
    旋转量组 = 计算旋转量(关键点)
    脸大小 = 计算脸大小(关键点)
    眼睛大小 = 计算眼睛大小(关键点)
    嘴大小 = 计算嘴大小(关键点)
    眉毛高度 = 计算眉毛高度(关键点)
    
    img //= 2
    img[脸位置.top():脸位置.bottom(), 脸位置.left():脸位置.right()] *= 2 
    for i, (px, py) in enumerate(关键点):
        cv2.putText(img, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    
    return np.concatenate([旋转量组, 相对位置, 嘴大小, 脸大小, 眼睛大小, 眉毛高度])

原点特征组 = 提取图片特征(cv2.imread(args.face))
特征组 = 原点特征组 - 原点特征组

def 获取特征组():
    global 特征组
    return 特征组


def 转移():
    global 特征组
    logging.warning('转移线程启动了！')
    while True:
        特征组 = pipe[1].recv()

'''
默认使用模型中的第二个说话人，如需要请修改speaker_id,noise_scale,noise_scale_w等参数
'''
def 语音核心():
    with open(args.texture, "r", encoding="utf-8") as f1:
        text = f1.read()
    speaker_id = 1
    stn_tst = get_text(text,hps_ms)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(dev)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(dev)
        sid = torch.LongTensor([speaker_id]).to(dev)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.267, noise_scale_w=0.4, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(args.audio,22050, audio)
    predictor = Wav2LipPredictor(checkpoint_path=None,
                                 static=False,
                                 fps=25,
                                 pads=[0, 10, 0, 0],
                                 face_det_batch_size=16,
                                 wav2lip_batch_size=128,
                                 resize_factor=1,
                                 crop=[0, -1, 0, -1],
                                 box=[-1, -1, -1, -1],
                                 rotate=False,
                                 nosmooth=False,
                                 face_detector='sfd',
                                 face_enhancement=False)
    predictor.run(args.face, args.audio,args.outfile)
    with open(args.texture, "w", encoding="utf-8") as f1:
        text = f1.write('跳过这句话')

def 播放语音():
    filename = args.audio
    winsound.PlaySound(filename, winsound.SND_FILENAME)


def 语音循环(pipe):
    while True:
        with open(args.texture, "r", encoding="utf-8") as f1:
            text = f1.read()
        if text == "跳过这句话":
            待机循环(pipe)
        else:
            语音核心()
            捕捉循环(pipe)


def 待机循环(pipe):
    global 原点特征组
    global 特征组
    cap = cv2.VideoCapture(args.waitVideo)
    while True:
        try:
            with 计时.帧率计('提特征'):
                ret, img = cap.read()
                新特征组 = 提取图片特征(img)
                #cv2.imshow('', img[:, ::-1])
                cv2.waitKey(30)
                if 新特征组 is not None:
                    特征组 = 新特征组 - 原点特征组
                pipe.send(特征组)
        except:
            break
    cap.release()
    cv2.destroyAllWindows()

def 捕捉循环(pipe):
    global 原点特征组
    global 特征组
    cap = cv2.VideoCapture(args.outfile)
    a = threading.Thread(target = 播放语音)
    a.start()
    while True:
        try:
            with 计时.帧率计('提特征'):
                ret, img = cap.read()
                新特征组 = 提取图片特征(img)
                #cv2.imshow('', img[:, ::-1])
                cv2.waitKey(30)
                if 新特征组 is not None:
                    特征组 = 新特征组 - 原点特征组
                pipe.send(特征组)
        except:
            break
    cap.release()
    cv2.destroyAllWindows()

def 启动():
    t = threading.Thread(target=转移)
    t.setDaemon(True)
    t.start()
    logging.warning('捕捉进程启动中……')
    p = multiprocessing.Process(target=语音循环, args=(pipe[0],))
    p.daemon = True
    p.start()


if __name__ == '__main__':
    启动()
    np.set_printoptions(precision=3, suppress=True)
    while True:
        time.sleep(0.1)
        # print(获取特征组())
