# audio-drive-live2d-with-vits-support
# 语音驱动的vtb尝试
# 环境要求
# PaddleGAN Opencv dlib OpenGL vits所需的环境

参考 https://github.com/PaddlePaddle/PaddleGAN

https://github.com/RimoChan/Vtuber_Tutorial

具体环境安装比较麻烦，需要你安装vs2022和cmake，缺少的包需额外安装

#文件配置
```
#参考现实.py
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

#参考wavlip.py

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

```
# live2d启动
```
#把wav2lip.py与 现实.py丢进vits项目中
cd vits_with_live2d
python wav2lip.py

```
# 启动绿皮chatbot或tts
'''
python launcher.py
#python tts,py
'''
