import cv2
from ultra_infer import ModelFormat, Runtime, RuntimeOption
import time
import numpy as np 
def build_infer():

    ui_option = RuntimeOption()
    ui_option.use_rknpu2()
    ui_option.set_model_path(
        "/home/orangepi/rknn-cpp-python/model/lenet5_32.rknn", "", ModelFormat.RKNN  
    )
    ui_runtime = Runtime(ui_option)

    ui_runtime.init_mat_rkmpp("/home/orangepi/rknpu-cpp-python/input.mp4","192.168.0.66","12345678")
    index = 0
    while True:
        # # 假设 output[0] 是 NumPy 数组（形状 (1080, 1920, 3)）
        # output = ui_runtime.get_mat("rtsp://192.168.1.15:8550/streamch1")
        path = f"./output/gsj_{index}.jpg"
        outputs = ui_runtime.get_mat()
        cv2.imwrite(path,outputs[0])
        index = index + 1


build_infer()
