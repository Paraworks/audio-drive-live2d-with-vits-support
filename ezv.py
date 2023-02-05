import matplotlib.pyplot as plt
import IPython.display as ipd
import dlib
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import multiprocessing
import threading
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import argparse
import paddle
from ppgan.apps.wav2lip_predictor import Wav2LipPredictor
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
parser.add_argument('--config',
                    type=str,
                    help='Config file of vits model.',
                    default='../res/config.json')
parser.add_argument('--model',
                    type=str,
                    help='Checkpoint file of vits model.',
                    default='../res/model.pth')
parser.add_argument('--texture',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/status.txt')
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

def ttv(text):
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
def main():
    while True:
        tts_text = input(":")
        text = '[ZH]' + tts_text + '[ZH]'
        ttv(text)
if __name__ == '__main__':
    main()
