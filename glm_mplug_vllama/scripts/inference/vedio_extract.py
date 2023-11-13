"""
@Time       : 2023/10/27 11:16
@File       : vedio_extract.py
@Description: 
"""
import cv2
import os


def extract_frames(video_path, output_path, frame_rate):
    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)

    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"获取视频的帧率:{fps}")

    # 计算每隔多少帧抽取一帧
    frame_interval = int(fps * frame_rate)

    # 初始化帧计数器
    frame_count = 0
    num = 0
    while True:
        # 读取视频的一帧
        ret, frame = video.read()

        # 如果无法读取到帧，则退出循环
        if not ret:
            break

        # 如果帧计数器是frame_interval的倍数，则保存该帧
        if frame_count % frame_interval == 0:
            # 构造保存路径
            save_path = os.path.join(output_path, f"frame_{frame_count}.jpg")

            # 保存帧为图片
            cv2.imwrite(save_path, frame)

            num = frame_count / (fps * frame_rate)  # 视频fps为25，设定为3.3s抽帧，相乘后82.5取的82，具体看自身情况
            print(f"当前处理到第{num}张")

        # 增加帧计数器
        frame_count += 1

    # 释放视频对象
    video.release()


# 调用函数进行抽帧处理
video_path = "test.mp4"  # 修改此处为输入路径
output_path = "extract_output"  # 修改此处为输出路径
frame_rate = 3.3  # 每多少秒抽帧，此处设定为3.3秒抽取一张图片
extract_frames(video_path, output_path, frame_rate)
