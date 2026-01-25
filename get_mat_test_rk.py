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
    num_inputs = ui_runtime.num_inputs()
    _input_names = [ui_runtime.get_input_info(i).name for i in range(num_inputs)]
    img = cv2.imread("/home/orangepi/rknn-cpp-Multithreading-main/70119185d6f33efa199f489c67bed8a3.jpg", cv2.COLOR_BGR2RGB)
    # print(img)
    inputs = {}
    
    inputs[_input_names[0]] = img
    start = time.time()
    # ui_runtime.init_mat("rtsp://192.168.1.15:8550/streamch1")
    # ui_runtime.init_mat("/home/orangepi/rknpu-cpp-python/input.mp4")
    index = 0
    while True:
        # # 假设 output[0] 是 NumPy 数组（形状 (1080, 1920, 3)）
        # output = ui_runtime.get_mat("rtsp://192.168.1.15:8550/streamch1")
        outputs = ui_runtime.get_mat_rkmpp("rtsp://192.168.1.15:8550/streamch1")
        print("****************")
        print(len(outputs))
        # time.sleep(1)
        # 若原数组已是 BGR 格式，直接使用：img_bgr = img_np
    
        # cv2.imshow("RTSP Stream", img_bgr)
        
        # # 4. 按 'q' 退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break



build_infer()
