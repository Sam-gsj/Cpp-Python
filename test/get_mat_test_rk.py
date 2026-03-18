import cv2
from ultra_infer import ModelFormat, Runtime, RuntimeOption
import time
import numpy as np 
import os
def build_infer():

    ui_option = RuntimeOption()
    ui_option.use_rknpu2()
    ui_option.set_model_path(
        "./source/lenet5_32.rknn", "", ModelFormat.RKNN
    )
    ui_runtime = Runtime(ui_option)
    num_inputs = ui_runtime.num_inputs()
    _input_names = [ui_runtime.get_input_info(i).name for i in range(num_inputs)]
    inputs = {}

    # ui_runtime.init_mat("rtsp://192.168.1.15:8550/streamch1")
    ui_runtime.init_mat_rkmpp("./source/QQ2026318-14610.mp4")
    index = 0
    while True:
        outputs = ui_runtime.get_mat()
        inputs[_input_names[0]] = outputs[0]

        outputs = ui_runtime.infer_gsj(inputs) 
        if not os.path.exists("./result"):
            os.makedirs("./result")
        cv2.imwrite(f"./result/output_{index}.jpg", outputs[3])
        index+=1
        print(f"Success inference: {index}")

build_infer()
