import struct

import cv2
import torch
import pyvirtualcam
import numpy as np
import mediapipe as mp
from PIL import Image

import tha2.poser.modes.mode_20_wx
from models import TalkingAnimeLight, TalkingAnime3
from pose import get_pose
from utils import preprocessing_image, postprocessing_image

import errno
import json
import os
import queue
import socket
import time
import math
from pynput.mouse import Button, Controller
import re
from collections import OrderedDict
from multiprocessing import Value, Process, Queue

from pyanime4k import ac

from tha2.mocap.ifacialmocap_constants import *

from args import args

from tha3.util import torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image

import collections
import winsound
import threading

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)


class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0


device = torch.device('cuda') if torch.cuda.is_available() and not args.skip_model else torch.device('cpu')


def create_default_blender_data():
    data = {}

    for blendshape_name in BLENDSHAPE_NAMES:
        data[blendshape_name] = 0.0

    data[HEAD_BONE_X] = 0.0
    data[HEAD_BONE_Y] = 0.0
    data[HEAD_BONE_Z] = 0.0
    data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[LEFT_EYE_BONE_X] = 0.0
    data[LEFT_EYE_BONE_Y] = 0.0
    data[LEFT_EYE_BONE_Z] = 0.0
    data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[RIGHT_EYE_BONE_X] = 0.0
    data[RIGHT_EYE_BONE_Y] = 0.0
    data[RIGHT_EYE_BONE_Z] = 0.0
    data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    return data


class OSFClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = args.osf.split(':')[0]
        self.port = int(args.osf.split(':')[1])
        self.ifm_fps_number = Value('f', 0.0)
        self.perf_time = 0

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(("", self.port))
        self.socket.settimeout(0.1)
        ifm_fps = FPS()
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK or err == 'timed out':
                    continue
                else:
                    raise e

            # socket_string = socket_bytes.decode("utf-8")
            osf_raw = (struct.unpack('=di2f2fB1f4f3f3f68f136f210f14f', socket_bytes))
            # print(osf_raw[432:])
            data = {}
            OpenSeeDataIndex = [
                'time',
                'id',
                'cameraResolutionW',
                'cameraResolutionH',
                'rightEyeOpen',
                'leftEyeOpen',
                'got3DPoints',
                'fit3DError',
                'rawQuaternionX',
                'rawQuaternionY',
                'rawQuaternionZ',
                'rawQuaternionW',
                'rawEulerX',
                'rawEulerY',
                'rawEulerZ',
                'translationY',
                'translationX',
                'translationZ',
            ]
            for i in range(len(OpenSeeDataIndex)):
                data[OpenSeeDataIndex[i]] = osf_raw[i]
            data['translationY'] *= -1
            data['translationZ'] *= -1
            data['rotationY'] = data['rawEulerY']-10
            data['rotationX'] = (-data['rawEulerX'] + 360)%360-180
            data['rotationZ'] = (data['rawEulerZ'] - 90)
            OpenSeeFeatureIndex = [
                'EyeLeft',
                'EyeRight',
                'EyebrowSteepnessLeft',
                'EyebrowUpDownLeft',
                'EyebrowQuirkLeft',
                'EyebrowSteepnessRight',
                'EyebrowUpDownRight',
                'EyebrowQuirkRight',
                'MouthCornerUpDownLeft',
                'MouthCornerInOutLeft',
                'MouthCornerUpDownRight',
                'MouthCornerInOutRight',
                'MouthOpen',
                'MouthWide'
            ]

            for i in range(68):
                data['confidence' + str(i)] = osf_raw[i + 18]
            for i in range(68):
                data['pointsX' + str(i)] = osf_raw[i * 2 + 18 + 68]
                data['pointsY' + str(i)] = osf_raw[i * 2 + 18 + 68 + 1]
            for i in range(70):
                data['points3DX' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2]
                data['points3DY' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2 + 1]
                data['points3DZ' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2 + 2]

            for i in range(len(OpenSeeFeatureIndex)):
                data[OpenSeeFeatureIndex[i]] = osf_raw[i + 432]
            # print(data['rotationX'],data['rotationY'],data['rotationZ'])

            a = np.array([
                data['points3DX66'] - data['points3DX68'] + data['points3DX67'] - data['points3DX69'],
                data['points3DY66'] - data['points3DY68'] + data['points3DY67'] - data['points3DY69'],
                data['points3DZ66'] - data['points3DZ68'] + data['points3DZ67'] - data['points3DZ69']
            ])
            a = (a / np.linalg.norm(a))
            data['eyeRotationX'] = a[0]
            data['eyeRotationY'] = a[1]
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()


ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()


class IFMClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = args.ifm.split(':')[0]
        self.port = int(args.ifm.split(':')[1])
        self.ifm_fps_number = Value('f', 0.0)
        self.perf_time = 0

    def run(self):

        udpClntSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        data = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719"

        data = data.encode('utf-8')

        udpClntSock.sendto(data, (self.address, self.port))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(("", self.port))
        self.socket.settimeout(0.1)
        ifm_fps = FPS()
        pre_socket_string = ''
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK or err == 'timed out':
                    continue
                else:
                    raise e
            socket_string = socket_bytes.decode("utf-8")
            if args.debug and pre_socket_string != socket_string:
                self.ifm_fps_number.value = ifm_fps()
                pre_socket_string = socket_string
            # print(socket_string)
            # blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(socket_string)

            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        for item in blender_data.split('|'):
            if item.find('#') != -1:
                k, arr = item.split('#')
                arr = [float(n) for n in arr.split(',')]
                data[k.replace("_L", "Left").replace("_R", "Right")] = arr
            elif item.find('-') != -1:
                k, v = item.split("-")
                data[k.replace("_L", "Left").replace("_R", "Right")] = float(v) / 100

        to_rad = 57.3
        data[HEAD_BONE_X] = data["=head"][0] / to_rad
        data[HEAD_BONE_Y] = data["=head"][1] / to_rad
        data[HEAD_BONE_Z] = data["=head"][2] / to_rad
        data[HEAD_BONE_QUAT] = [data["=head"][3], data["=head"][4], data["=head"][5], 1]
        # print(data[HEAD_BONE_QUAT][2],min(data[EYE_BLINK_LEFT],data[EYE_BLINK_RIGHT]))
        data[RIGHT_EYE_BONE_X] = data["rightEye"][0] / to_rad
        data[RIGHT_EYE_BONE_Y] = data["rightEye"][1] / to_rad
        data[RIGHT_EYE_BONE_Z] = data["rightEye"][2] / to_rad
        data[LEFT_EYE_BONE_X] = data["leftEye"][0] / to_rad
        data[LEFT_EYE_BONE_Y] = data["leftEye"][1] / to_rad
        data[LEFT_EYE_BONE_Z] = data["leftEye"][2] / to_rad

        return data


class MouseClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def run(self):
        mouse = Controller()
        posLimit = [int(x) for x in args.mouse_input.split(',')]
        prev = {
            'eye_l_h_temp': 0,
            'eye_r_h_temp': 0,
            'mouth_ratio': 0,
            'eye_y_ratio': 0,
            'eye_x_ratio': 0,
            'x_angle': 0,
            'y_angle': 0,
            'z_angle': 0,
        }
        while True:
            pos = mouse.position
            # print(pos)
            eye_limit = [0.8, 0.5]
            head_eye_reduce = 0.6
            head_slowness = 0.2
            mouse_data = {
                'eye_l_h_temp': 0,
                'eye_r_h_temp': 0,
                'mouth_ratio': 0,
                'eye_y_ratio': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]) * eye_limit[1],
                'eye_x_ratio': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]) * eye_limit[0],
                'x_angle': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]),
                'y_angle': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]),
                'z_angle': 0,
            }
            mouse_data['x_angle'] = np.interp(head_slowness, [0, 1], [prev['x_angle'], mouse_data['x_angle']])
            mouse_data['y_angle'] = np.interp(head_slowness, [0, 1], [prev['y_angle'], mouse_data['y_angle']])
            mouse_data['eye_y_ratio'] -= mouse_data['x_angle'] * eye_limit[1] * head_eye_reduce
            mouse_data['eye_x_ratio'] -= mouse_data['y_angle'] * eye_limit[0] * head_eye_reduce
            if args.bongo:
                mouse_data['y_angle'] += 0.05
                mouse_data['x_angle'] += 0.05
            prev = mouse_data
            self.queue.put_nowait(mouse_data)
            time.sleep(1 / 60)


class ModelClientProcess(Process):
    def __init__(self, input_image):
        super().__init__()
        self.should_terminate = Value('b', False)
        self.updated = Value('b', False)
        self.data = None
        self.input_image = input_image
        self.output_queue = Queue()
        self.input_queue = Queue()
        self.model_fps_number = Value('f', 0.0)
        self.gpu_fps_number = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)

    def run(self):
        model = None
        if not args.skip_model:
            model = TalkingAnime3().to(device)
            model = model.eval()
            model = model
            print("Pretrained Model Loaded")

        eyebrow_vector = torch.empty(1, 12, dtype=torch.half if args.model.endswith('half') else torch.float)
        mouth_eye_vector = torch.empty(1, 27, dtype=torch.half if args.model.endswith('half') else torch.float)
        pose_vector = torch.empty(1, 6, dtype=torch.half if args.model.endswith('half') else torch.float)

        input_image = self.input_image.to(device)
        eyebrow_vector = eyebrow_vector.to(device)
        mouth_eye_vector = mouth_eye_vector.to(device)
        pose_vector = pose_vector.to(device)

        model_cache = OrderedDict()
        tot = 0
        hit = 0
        hit_in_a_row = 0
        model_fps = FPS()
        gpu_fps = FPS()
        while True:
            model_input = None
            try:
                while not self.input_queue.empty():
                    model_input = self.input_queue.get_nowait()
            except queue.Empty:
                continue
            if model_input is None: continue
            simplify_arr = [1000] * ifm_converter.pose_size
            if args.simplify >= 1:
                simplify_arr = [200] * ifm_converter.pose_size
                simplify_arr[ifm_converter.eye_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_surprised_left_index] = 30
                simplify_arr[ifm_converter.eye_surprised_right_index] = 30
                simplify_arr[ifm_converter.iris_rotation_x_index] = 25
                simplify_arr[ifm_converter.iris_rotation_y_index] = 25
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
            if args.simplify >= 2:
                simplify_arr[ifm_converter.head_x_index] = 100
                simplify_arr[ifm_converter.head_y_index] = 100
                simplify_arr[ifm_converter.eye_surprised_left_index] = 10
                simplify_arr[ifm_converter.eye_surprised_right_index] = 10
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_left_index]
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_right_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_wink_right_index] / 2
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_right_index] / 2

                uosum = model_input[ifm_converter.mouth_uuu_index] + \
                        model_input[ifm_converter.mouth_ooo_index]
                model_input[ifm_converter.mouth_ooo_index] = uosum
                model_input[ifm_converter.mouth_uuu_index] = 0
                is_open = (model_input[ifm_converter.mouth_aaa_index] + model_input[
                    ifm_converter.mouth_iii_index] + uosum) > 0
                model_input[ifm_converter.mouth_lowered_corner_left_index] = 0
                model_input[ifm_converter.mouth_lowered_corner_right_index] = 0
                model_input[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
                model_input[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
            if args.simplify >= 3:
                simplify_arr[ifm_converter.iris_rotation_x_index] = 20
                simplify_arr[ifm_converter.iris_rotation_y_index] = 20
                simplify_arr[ifm_converter.eye_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_wink_right_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
            if args.simplify >= 4:
                simplify_arr[ifm_converter.head_x_index] = 50
                simplify_arr[ifm_converter.head_y_index] = 50
                simplify_arr[ifm_converter.neck_z_index] = 100
                model_input[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
                model_input[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
                simplify_arr[ifm_converter.iris_rotation_x_index] = 10
                simplify_arr[ifm_converter.iris_rotation_y_index] = 10
                simplify_arr[ifm_converter.eye_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_surprised_left_index] = 8
                simplify_arr[ifm_converter.eye_surprised_right_index] = 8
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_wink_right_index]
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2

                model_input[ifm_converter.eye_surprised_left_index] += model_input[
                    ifm_converter.eye_surprised_right_index]
                model_input[ifm_converter.eye_surprised_right_index] = model_input[
                                                                           ifm_converter.eye_surprised_left_index] / 2
                model_input[ifm_converter.eye_surprised_left_index] = model_input[
                                                                          ifm_converter.eye_surprised_left_index] / 2

                model_input[ifm_converter.eye_happy_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.mouth_aaa_index] = min(
                    model_input[ifm_converter.mouth_aaa_index] +
                    model_input[ifm_converter.mouth_ooo_index] / 2 +
                    model_input[ifm_converter.mouth_iii_index] / 2 +
                    model_input[ifm_converter.mouth_uuu_index] / 2, 1
                )
                model_input[ifm_converter.mouth_ooo_index] = 0
                model_input[ifm_converter.mouth_iii_index] = 0
                model_input[ifm_converter.mouth_uuu_index] = 0
            for i in range(4, args.simplify):
                simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]
            for i in range(0, len(simplify_arr)):
                if simplify_arr[i] > 0:
                    model_input[i] = round(model_input[i] * simplify_arr[i]) / simplify_arr[i]
            input_hash = hash(tuple(model_input))
            cached = model_cache.get(input_hash)
            tot += 1
            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            if cached is not None and hit_in_a_row < self.model_fps_number.value:
                self.output_queue.put_nowait(cached)
                model_cache.move_to_end(input_hash)
                hit += 1
                hit_in_a_row += 1
            else:
                hit_in_a_row = 0
                if args.perf == 'model':
                    tic = time.perf_counter()
                if args.eyebrow:
                    for i in range(12):
                        eyebrow_vector[0, i] = model_input[i]
                        eyebrow_vector_c[i] = model_input[i]
                for i in range(27):
                    mouth_eye_vector[0, i] = model_input[i + 12]
                    mouth_eye_vector_c[i] = model_input[i + 12]
                for i in range(6):
                    pose_vector[0, i] = model_input[i + 27 + 12]
                if model is None:
                    output_image = input_image
                else:
                    output_image = model(input_image, mouth_eye_vector, pose_vector, eyebrow_vector, mouth_eye_vector_c,
                                         eyebrow_vector_c,
                                         self.gpu_cache_hit_ratio)
                if args.perf == 'model':
                    torch.cuda.synchronize()
                    print("model", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()
                postprocessed_image = output_image[0].float()
                if args.perf == 'model':
                    print("cpu()", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()
                postprocessed_image = convert_linear_to_srgb((postprocessed_image + 1.0) / 2.0)
                c, h, w = postprocessed_image.shape
                postprocessed_image = 255.0 * torch.transpose(postprocessed_image.reshape(c, h * w), 0, 1).reshape(h, w,
                                                                                                                   c)
                postprocessed_image = postprocessed_image.byte().detach().cpu().numpy()
                if args.perf == 'model':
                    print("postprocess", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()

                self.output_queue.put_nowait(postprocessed_image)
                if args.debug:
                    self.gpu_fps_number.value = gpu_fps()
                if args.max_cache_len > 0:
                    model_cache[input_hash] = postprocessed_image
                    if len(model_cache) > args.max_cache_len:
                        model_cache.popitem(last=False)
            if args.debug:
                self.model_fps_number.value = model_fps()
                self.cache_hit_ratio.value = hit / tot
def display(video_path):
    with open('../res/status.txt', "w", encoding="utf-8") as f1:
        f1.write('没活了')
    cap = cv2.VideoCapture(video_path)
    isopen = cap.isOpened()
    filename = "../res/audio.wav"
    if isopen:
        winsound.PlaySound(filename, winsound.SND_FILENAME)
        with open('../res/Masahiro.mp4', "rb") as f2:
                mp4 = f2.read()
    with open('../res/result_voice.mp4', "wb") as f3:
        f3.write(mp4)
@torch.no_grad()
def main():
    img = Image.open(f"data/images/{args.character}.png")
    img = img.convert('RGBA')
    IMG_WIDTH = 512
    wRatio = img.size[0] / IMG_WIDTH
    img = img.resize((IMG_WIDTH, int(img.size[1] / wRatio)))
    for i, px in enumerate(img.getdata()):
        if px[3] <= 0:
            y = i // IMG_WIDTH
            x = i % IMG_WIDTH
            img.putpixel((x, y), (0, 0, 0, 0))
    input_image = preprocessing_image(img.crop((0, 0, IMG_WIDTH, IMG_WIDTH)))
    if args.model.endswith('half'):
        input_image = torch.from_numpy(input_image).half() * 2.0 - 1
    else:
        input_image = torch.from_numpy(input_image).float() * 2.0 - 1
    input_image = input_image.unsqueeze(0)
    extra_image = None
    if img.size[1] > IMG_WIDTH:
        extra_image = np.array(img.crop((0, IMG_WIDTH, img.size[0], img.size[1])))
    output_fps = FPS()
    video_path = "../res/result_voice.mp4"
    cap = cv2.VideoCapture(video_path)
    read_frame = 0
#    ret, frame = cap.read()
    frame_counter = 0
    facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    '''
    if args.anime4k:
        parameters = ac.Parameters()
        # enable HDN for ACNet
        parameters.HDN = True

        # a = ac.AC(
        #     managerList=ac.ManagerList([ac.CUDAManager(dID=0)]),
        #     type=ac.ProcessorType.Cuda_ACNet,
        # )

        a = ac.AC(
            managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=0, dID=0)]),
            type=ac.ProcessorType.OpenCL_ACNet,
        )
        a.set_arguments(parameters)
        print("Anime4K Loaded")
'''
    position_vector = [0, 0, 0, 1]
    position_vector_0 = None
    pose_vector_0 = None

    pose_queue = []
    blender_data={}
    if(args.ifm):
        blender_data = create_default_blender_data()
    mouse_data = {
        'eye_l_h_temp': 0,
        'eye_r_h_temp': 0,
        'mouth_ratio': 0,
        'eye_y_ratio': 0,
        'eye_x_ratio': 0,
        'x_angle': 0,
        'y_angle': 0,
        'z_angle': 0,
    }

    model_output = None
    model_process = ModelClientProcess(input_image)
    model_process.daemon = True
    model_process.start()
    while True:
        # ret, frame = cap.read()
        # input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = facemesh.process(input_frame)
        try:
            ret, frame = cap.read()
            frame_counter += 1
            if frame_counter == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                frame_counter = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = facemesh.process(input_frame)

            if results.multi_face_landmarks is None:
                continue

            facial_landmarks = results.multi_face_landmarks[0].landmark

            if args.debug:
                pose, debug_image = get_pose(facial_landmarks, frame)
            else:
                pose = get_pose(facial_landmarks)

            if len(pose_queue) < 3:
                pose_queue.append(pose)
                pose_queue.append(pose)
                pose_queue.append(pose)
            else:
                pose_queue.pop(0)
                pose_queue.append(pose)

            np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.6, 0.3, 0.1])

            eye_l_h_temp = np_pose[0]
            eye_r_h_temp = np_pose[1]
            mouth_ratio = np_pose[2]
            eye_y_ratio = np_pose[3]
            eye_x_ratio = np_pose[4]
            x_angle = np_pose[5]
            y_angle = np_pose[6]
            z_angle = np_pose[7]

            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            pose_vector_c = [0.0] * 6

            mouth_eye_vector_c[2] = eye_l_h_temp
            mouth_eye_vector_c[3] = eye_r_h_temp

            mouth_eye_vector_c[14] = mouth_ratio * 1.5

            mouth_eye_vector_c[25] = eye_y_ratio
            mouth_eye_vector_c[26] = eye_x_ratio

            pose_vector_c[0] = (x_angle - 1.5) * 1.6
            pose_vector_c[1] = y_angle * 2.0  # temp weight
            pose_vector_c[2] = (z_angle + 1.5) * 2  # temp weight
        except:
            cap = cv2.VideoCapture('../res/result_voice.mp4')
        pose_vector_c[3] = pose_vector_c[1]
        pose_vector_c[4] = pose_vector_c[2]

        model_input_arr = eyebrow_vector_c
        model_input_arr.extend(mouth_eye_vector_c)
        model_input_arr.extend(pose_vector_c)

        model_process.input_queue.put_nowait(model_input_arr)
        has_model_output = 0
        try:
            new_model_output = model_output
            while not model_process.output_queue.empty():
                has_model_output += 1
                new_model_output = model_process.output_queue.get_nowait()
            model_output = new_model_output
        except queue.Empty:
            pass
        if model_output is None:
            time.sleep(1)
            continue
        # print(has_model_output)
        # should_output=should_output or has_model_output
        # if not should_output:
        #     continue

        postprocessed_image = model_output

        if args.perf == 'main':
            print('===')
            print("input", time.perf_counter() - tic)
            tic = time.perf_counter()

        if extra_image is not None:
            postprocessed_image = cv2.vconcat([postprocessed_image, extra_image])
        #在这里
        def judage():
            with open('../res/status.txt', "r", encoding="utf-8") as f1:
                text = f1.read()
            if text != '没活了':
                display(video_path)
        jd = threading.Thread(target = judage)
        jd.start()
        k_scale = 1
        rotate_angle = 0
        dx = 0
        dy = 0
        if args.extend_movement:
            k_scale = position_vector[2] * math.sqrt(args.extend_movement) + 1
            rotate_angle = -position_vector[0] * 10 * args.extend_movement
            dx = position_vector[0] * 400 * k_scale * args.extend_movement
            dy = -position_vector[1] * 600 * k_scale * args.extend_movement
        if args.bongo:
            rotate_angle -= 5
        rm = cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_WIDTH / 2), rotate_angle, k_scale)
        rm[0, 2] += dx + args.output_w / 2 - IMG_WIDTH / 2
        rm[1, 2] += dy + args.output_h / 2 - IMG_WIDTH / 2
        postprocessed_image = cv2.warpAffine(
            postprocessed_image,
            rm,
            (args.output_w, args.output_h))

        if args.perf == 'main':
            print("extendmovement", (time.perf_counter() - tic) * 1000)
            tic = time.perf_counter()

        output_fps_number = output_fps()

        if args.anime4k:
            alpha_channel = postprocessed_image[:, :, 3]
            alpha_channel = cv2.resize(alpha_channel, None, fx=2, fy=2)

            # a.load_image_from_numpy(cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2RGB), input_type=ac.AC_INPUT_RGB)
            # img = cv2.imread("character/test41.png")
            img1 = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGR)
            # a.load_image_from_numpy(img, input_type=ac.AC_INPUT_BGR)
            a.load_image_from_numpy(img1, input_type=ac.AC_INPUT_BGR)
            a.process()
            postprocessed_image = a.save_image_to_numpy()
            postprocessed_image = cv2.merge((postprocessed_image, alpha_channel))
            postprocessed_image = cv2.cvtColor(postprocessed_image, cv2.COLOR_BGRA2RGBA)
            if args.perf == 'main':
                print("anime4k", (time.perf_counter() - tic) * 1000)
                tic = time.perf_counter()
        if args.alpha_split:
            alpha_image = cv2.merge(
                [postprocessed_image[:, :, 3], postprocessed_image[:, :, 3], postprocessed_image[:, :, 3]])
            alpha_image = cv2.cvtColor(alpha_image, cv2.COLOR_RGB2RGBA)
            postprocessed_image = cv2.hconcat([postprocessed_image, alpha_image])

        if args.debug:
            output_frame = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGRA)
            # resized_frame = cv2.resize(output_frame, (np.min(debug_image.shape[:2]), np.min(debug_image.shape[:2])))
            # output_frame = np.concatenate([debug_image, resized_frame], axis=1)
            cv2.putText(output_frame, str('OUT_FPS:%.1f' % output_fps_number), (0, 16), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f / %.1f' % (model_process.model_fps_number.value, model_process.gpu_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            else:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f' % (model_process.model_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.ifm is not None:
                cv2.putText(output_frame, str('IFM_FPS:%.1f' % client_process.ifm_fps_number.value), (0, 48),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str('MEMCACHED:%.1f%%' % (model_process.cache_hit_ratio.value * 100)),
                            (0, 64),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_gpu_cache_len > 0:
                cv2.putText(output_frame, str('GPUCACHED:%.1f%%' % (model_process.gpu_cache_hit_ratio.value * 100)),
                            (0, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("frame", output_frame)
            # cv2.imshow("camera", debug_image)
            cv2.waitKey(1)
        if args.output_webcam:
            # result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            # result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
            #     cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            result_image = postprocessed_image
            if args.output_webcam == 'obs':
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
            cam.send(result_image)
            cam.sleep_until_next_frame()
        if args.perf == 'main':
            print("output", (time.perf_counter() - tic) * 1000)
            tic = time.perf_counter()


if __name__ == '__main__':
    main()