'''
import cv2
import winsound
import threading
global suatu
cap = cv2.VideoCapture("C:/Programming/vtuber/res/result_voice.mp4")
frame_counter = 0
def display():
    filename = 'C:/Programming/vtuber/res/audio.wav'
    if suatu == 1:
       winsound.PlaySound(filename, winsound.SND_FILENAME)
suatu = 1
while (cap.isOpened()):
   a = threading.Thread(target = display)
   a.start()
   ret, frame = cap.read()
   frame_counter += 1
   if frame_counter == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
      frame_counter = 0
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      suatu = 0

   cv2.imshow("frame", frame)
   key = cv2.waitKey(30)
   # ESC
   if key == 27:
      break
cap.release()
cv2.destroyAllWindows()
'''

# 使用cv2读取显示视频
 
# 引入math
import math
# 引入opencv
import cv2
from ffpyplayer.player import MediaPlayer
# opencv获取本地视频
 
def play_video(video_path, audio_play=True):
    cap = cv2.VideoCapture(video_path)
    if audio_play:
        player = MediaPlayer(video_path)
    # 打开文件状态
    isopen = cap.isOpened()
    if not isopen:
        print("Err: Video is failure. Exiting ...")
 
    # 视频时长总帧数
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 获取视频宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频高度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 播放帧间隔毫秒数
    wait = int(1000 / fps) if fps else 1
    # 帧数计数器
    read_frame = 0
 
    # 循环读取视频帧
    while (isopen):
        # 读取帧图像
        ret, frame = cap.read()
        # 读取错误处理
        if not ret:
 
            if read_frame < total_frame:
                # 读取错误
                print("Err: Can't receive frame. Exiting ...")
            else:
                # 正常结束
                print("Info: Stream is End")
            break
 
        # 帧数计数器+1
        read_frame = read_frame + 1
        cv2.putText(frame, "[{}/{}]".format(str(read_frame), str(int(total_frame))), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 9), 2)
        dst = cv2.resize(frame, (1920//2, 1080//2), interpolation=cv2.INTER_CUBIC)  # 窗口大小
        # 计算当前播放时码
        timecode_h = int(read_frame / fps / 60 / 60)
        timecode_m = int(read_frame / fps / 60)
        timecode_s = read_frame / fps % 60
        s = math.modf(timecode_s)
        timecode_s = int(timecode_s)
        timecode_f = int(s[0] * fps)
        print("{:0>2d}:{:0>2d}:{:0>2d}.{:0>2d}".format(timecode_h, timecode_m, timecode_s, timecode_f))
 
        # 显示帧图像
        cv2.imshow('image', dst)
 
        # 播放间隔
        wk = cv2.waitKey(wait)
 
        # 按键值  & 0xFF是一个二进制AND操作 返回一个不是单字节的代码
        keycode = wk & 0xff
 
        # 空格键暂停
        if keycode == ord(" "):
            cv2.waitKey(0)
 
        # q键退出
        if keycode == ord('q'):
            print("Info: By user Cancal ...")
            break
 
    # 释放实例
    cap.release()
 
    # 销毁窗口
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    play_video("C:/Programming/vtuber/res/result_voice.mp4", audio_play=True)