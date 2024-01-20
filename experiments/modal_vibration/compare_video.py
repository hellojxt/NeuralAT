import cv2
import numpy as np


import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def add_subtitles_and_save_first_frame(
    video_path,
    output_path,
    subtitle_text,
    font_size,
    relative_position,
    preview=False,
):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font_path = "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 计算字幕的绝对位置
    position = (int(relative_position[0] * width), int(relative_position[1] * height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将OpenCV帧转换为Pillow图像
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # 在指定位置添加字幕
        draw.text(position, subtitle_text, font=font, fill=(255, 255, 255))

        # 将Pillow图像转换回OpenCV格式
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 如果是第一帧，保存为预览图像
        if preview:
            cv2.imwrite("test.png", frame)
            cap.release()
            out.release()
            return

        # 写入带字幕的帧
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()


# 使用示例
add_subtitles_and_save_first_frame(
    "path/to/your/video.mp4",
    "path/to/output/video.mp4",
    "test",
    40,
    (0.5, 0.8),
)
