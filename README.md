# audio-drive-live2d-with-vits-support
# 本地化语音驱动皮套尝试
# 环境要求 opengl&&opencv
Windows 安装Vs2022和Cmake
PaddleGAN Opencv dlib OpenGL vits所需的环境
```
#paddlepaddle whl文件夹:  www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
#安装dlib conda install -c https://conda.anaconda.org/conda-forge dlib
#Paddlegun推荐仓库安装 https://github.com/PaddlePaddle/PaddleGAN
#参考requirements.txt
```

#基础文件配置
```
#参考vits_onnx/app.py Easyvtuber+vits的onnx导出模型，详细看我上一个仓库
def get_args():
    parser = argparse.ArgumentParser(description='inference')
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
    parser.add_argument('--texture',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/status.txt')
    parser.add_argument('--waitVideo',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/Masahiro.mp4')
    parser.add_argument('--onnx_model', 
                    type=str,
                    help = 'onnx checkpoint',
                    default = './moe/model.onnx')
    parser.add_argument('--cfg', 
                    type=str,
                    help = 'onnx config',
                    default="./moe/config.json")
    parser.add_argument('--outdir', 
                    type=str,
                    help='ouput directory',
                    default="./moe")
    parser.add_argument('--key',
                    type=str,
                    help='openai的key',
                    default = "your_key")
    args = parser.parse_args()
    return args

#参考ezv.py 如果采用EasyVtuber+vits则使用该文件
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
                    help='待机视频.',
                    default='../res/Masahiro.mp4')
                    
#参考现实.py 如使用手工皮套则用这个文件和 wavlip.py
#手工live2d样例仓库 https://github.com/RimoChan/Vtuber_Tutorial
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
                    help='待机视频.',
                    default='../res/Masahiro.mp4')

#参考wavlip.py

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
# Easy Vtuber 启动
[先根据大佬的说明文档搭建环境，不需要安装摄像头相关应用。](
https://github.com/yuyuyzl/EasyVtuber)然后使用本仓库的[main.py](https://github.com/Paraworks/audio-drive-live2d-with-vits-support/blob/main/EasyVtuber/main.py)替换原本的main.py
,我只对其做了一点的修改来支持视频读取。在不修改main.py,wav2lip.py,app.py等文件夹parser的情况下，
确保你自己的EasyVtuber文件夹与res及vits或者vits_onnx文件夹位于同一目录下，格式与本仓库相同。
```
cd EasyVtuber
python main.py --character png文件的名字
#vits_onnx 版本
cd vits_onnx
python app.py
#启动一个新的终端，把ezv.py丢进vits目录，启动语音端。目前只自带tts，未绑定chatbot
cd vits
python ezv.py
```
# 手工版live2d启动
```
#把wav2lip.py与 现实.py丢进vits项目中
#创建utrils_live2d 文件夹，放入本项目中的同名文件夹下的内容
cd vits_with_vtb
python wav2lip.py
```
# 启动绿皮chatbot或tts，你也可以认为这是机械神教的铁人
```
python launcher.py
#python tts,py
```
