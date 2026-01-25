import sys
import time
from PyQt5 import QtWidgets
import numpy as np
import os

# from src.info_data_v3 import save_data
# from src.info_data_v4 import move2pitch

os.environ["QT_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import easydict
import cv2
from find_block_v2 import draw_rect_max
from slope_v2 import calculate_slope_for_image, create_mask, apply_mask_to_image

from init_npu import InitNpu
from image_classification_v2_npu import img_classify

from info_data_v4_4 import com_info_v1, save_data  #, get_center_str
from DEM_strm_v2 import dem_strm
# from image_classification_v2 import img_classify

from list_v3 import all_best
from PC_thread_422port_v2 import Pro_Con_QT_Worker, PC_Resource

from ultra_infer import ModelFormat, Runtime, RuntimeOption

# from init_npu import InitNpu
# from image_classification_v2_npu import img_classify


def initial_cemara(com_info):
    info_all = com_info.info_all

    config_1 = {
            'serial_port': '/dev/ttyS3',
            'baud_rate': 115200,
        }
    config_1 = easydict.EasyDict(config_1)

    mt_resource_1 = PC_Resource(config_1)
    com_info.mt_resource = mt_resource_1

    work_1 = Pro_Con_QT_Worker()
    work_1.setup(mt_resource_1)
    work_1.start()

    config_2 = {
            'serial_port': '/dev/ttyS0',
            'baud_rate': 115200,
        }
    config_2 = easydict.EasyDict(config_2)
    mt_resource_2 = PC_Resource(config_2)
    work_2 = Pro_Con_QT_Worker()
    work_2.setup(mt_resource_2)
    # info_all.mt_resource = mt_resource
    work_2.start()

    config_3 = {
        'serial_port': '/dev/ttyS9',
        'baud_rate': 115200,
    }
    config_3 = easydict.EasyDict(config_3)
    mt_resource_3 = PC_Resource(config_3)
    work_3 = Pro_Con_QT_Worker()
    work_3.setup(mt_resource_3)
    work_3.start()

    info_save_path = info_all.inout_info.info_save_path
    if not os.path.exists(info_save_path):
        os.makedirs(info_save_path)

    for start in range(10):
        com_info.read_com_info()
        save_path = os.path.join(info_save_path, f'info_start_{start + 1}')
        save_data(info_all, save_path)

        if not com_info.recieved_singal:
            time.sleep(0.5)
        else:
            print("Inital Failed")
            break

    # move2pitch(info_all, work_1)

    return work_1, True


def build_infer():
    app = QtWidgets.QApplication(sys.argv)
    
    # 激光
    str_servo_laser = "55 AA 02 1A FA 01 02 05 0A 00 32 0F A0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 72 F0"

    print("signal initial")
    com_info = com_info_v1()
    info_all = com_info.info_all
 

    info_save_path = info_all.inout_info.info_save_path
    if not os.path.exists(info_save_path):
        os.makedirs(info_save_path)
    print(info_save_path)

    category_folder = os.path.join(info_all.model_info.cnn_save_path, "frame")
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    print(category_folder)

    work_1, boolean = initial_cemara(com_info)   

    print("signal initial finished")

    # DEM高程数据计算坡度，进行地形全局预判
    com_info.read_com_info()
    info_all.print_info()
    save_path = os.path.join(info_save_path, f'info_dem_start')
    save_data(info_all, save_path)

    start_time = time.time()
    dem_strm(info_all)
    end_time = time.time()
    dem_time = end_time - start_time
    print("dem模块花费的时间：", dem_time)

    ui_option = RuntimeOption()
    ui_option.use_rknpu2()
    ui_option.set_model_path(
        "/home/orangepi/rknpu-cpp-python/model/lenet5_32.rknn", "", ModelFormat.RKNN
    )
    ui_runtime = Runtime(ui_option)
    num_inputs = ui_runtime.num_inputs()
    _input_names = [ui_runtime.get_input_info(i).name for i in range(num_inputs)]
    img = cv2.imread("/home/orangepi/rknpu-cpp-python/frame_1.jpg", cv2.COLOR_BGR2RGB)
    # print(img)
    inputs = {}
    
    inputs[_input_names[0]] = img
    start = time.time()
    # ui_runtime.init_mat("rtsp://192.168.1.15:8550/streamch1")
    # ui_runtime.init_mat("/home/orangepi/rknpu-cpp-python/input.mp4","192.168.3.66",password="12345678")
    # ui_runtime.init_mat_rkmpp("/home/orangepi/rknpu-cpp-python/input.mp4","192.168.3.66",password="12345678")
    # ui_runtime.init_mat_rkmpp("rtsp://192.168.1.15:8550/streamch1", "192.168.1.66")
    # ui_runtime.init_mat_rkmpp("/home/orangepi/rknpu-cpp-python/input.mp4","192.168.3.66",password="12345678")
    # time.sleep(2)
    # ui_runtime.stop()
    # time.sleep(10)
    # ui_runtime.init_mat_rkmpp("/home/orangepi/rknpu-cpp-python/input.mp4","192.168.3.66",password="12345678")
    # index = 0
    
    frame_count = 0

    while True:        

        for i in range(149):
            # 假设 output[0] 是 NumPy 数组（形状 (1080, 1920, 3)）
            output = ui_runtime.get_mat()
            img_np = output[0]  # 直接获取 NumPy 数组
            inputs[_input_names[0]] = img_np
            output = ui_runtime.infer_gsj(inputs)
            for i , item in enumerate(output):
                file_name = f"gsj_{i}.jpg"
                cv2.imwrite(file_name, item)
            # 检查是否为有效数组
            if img_np is None or img_np.size == 0:
                print("获取图像失败")
                continue
            
            # 1. 确认数组形状和类型（调试用）
            print("图像形状:", img_np.shape)  # 应输出 (1080, 1920, 3)
            print("数据类型:", img_np.dtype)  # 应输出 uint8
            
            # 2. 通道顺序转换（关键：根据实际格式调整）
            # 若原数组是 RGB 格式，转为 BGR（OpenCV 显示需要）
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/home/orangepi/rknpu-cpp-python/output/{frame_count}.jpg", img_bgr)
            # index+=1
            # # time.sleep(1)
            # # 若原数组已是 BGR 格式，直接使用：img_bgr = img_np
        
            # cv2.imshow("RTSP Stream", img_bgr)
            
            # # 4. 按 'q' 退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # pic_path = f"/home/orangepi/rknpu-cpp-python/output/{frame_count}.jpg"
            # print(pic_path)
            # frame = cv2.imread(pic_path)
            frame = img_bgr
            
            start_picture = time.time()
            frame_count += 1
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # if not ret:  # 如果没有更多的帧
            #     break
            print("New frame count: ", frame_count)
            save_path = os.path.join(category_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(save_path, frame)
            print("save_path: ", save_path)

            print("hello 0")
            com_info.read_com_info()
            save_path = os.path.join(info_save_path, f'info_laser_0_{i + 1}')
            save_data(info_all, save_path)

            print("hello 1")
            work_1.producer.send_write_info(str_servo_laser)
            time.sleep(0.5)
            com_info.read_com_info()
            save_path = os.path.join(info_save_path, f'info_laser_2_{i + 1}')
            save_data(info_all, save_path)

            info_all.print_info()
            time.sleep(2)

            time.sleep(0.1)

            com_info.read_com_info()
            info_all.print_info()
            save_path = os.path.join(info_save_path, f'info_cv_1_{frame_count}')
            print("save_path: ", save_path)
            save_data(info_all, save_path)

            # if info_all.check():
            #     print ("info_all.check failed")
            #     time.sleep(0.5)
            # else:
            #     break

            
            # 图像处理
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            start_time = time.time()
            image = frame
            result_image_list = img_classify(image, frame_count, info_all, ui_runtime)
            # result_image_list = img_classify(image, frame_count, info_all)
            print("result_img", len(result_image_list))
            end_time = time.time()
            val_time = end_time - start_time
            print("val模块花费的时间：", val_time)

            # 图像坡度筛选
            start_time = time.time()
            image_slope, est_cache, img_cache = calculate_slope_for_image(info_all)
            end_time = time.time()
            rect_time = end_time - start_time
            print("坡度筛选模块花费的时间：", rect_time)

            info_all.image_slope = image_slope
            info_all.est_cache = est_cache
            info_all.img_cache = img_cache

            save_path = os.path.join(info_save_path, f'info_cv_2_{frame_count}')
            save_data(info_all, save_path)

            img_list_all = []
            for i in range(len(result_image_list)):
                # 图像坡度 与 分类照片合并
                result_PIL = result_image_list[i]
                result_img = cv2.cvtColor(np.asarray(result_PIL), cv2.COLOR_RGB2BGR)

                start_time = time.time()
                mask = create_mask(image_slope, threshold=15)
                image_with_mask = apply_mask_to_image(result_img, mask)
                end_time = time.time()
                rect_time = end_time - start_time
                print("地块坡度融合模块花费的时间：", rect_time)

                # 划取最大矩形框
                start_time = time.time()
                img_name = "rect_img_" + str(frame_count) + "-" + str(i)
                img_list = draw_rect_max(image_with_mask, info_all, img_name)
                end_time = time.time()
                rect_time = end_time - start_time
                print("画框模块花费的时间：", rect_time)
                img_list_all.extend(img_list)

            print("开始地块推荐")
            if len(img_list_all) == 0:
                print("没有合适地块")
            else:
                # 返回最大地块  TODO
                best_point = all_best(img_list_all)

                # byte_str = get_center_str(best_point[2][0], best_point[2][1])
                # ceter_x = oct2hex(best_point[2][0])
                # ceter_y = oct2hex(best_point[2][1])
                # # byte_str = "55 AA 02 1A 8C 01 58 0F 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 C3 F0" +' ' + str(frame_count)
                # byte_str = "55 AA 02 1A 9C " + ceter_x +" " + ceter_y + " 00 00 00 00 50 50 00 00 00 00 00 00 00 00 00 00 00 00 C3 F0"
                # # current_info.send_control_signal.emit(byte_str)
                # work_1.producer.send_write_info(byte_str)

            save_path = os.path.join(info_save_path, f'info_cv_3_{frame_count}')
            save_data(info_all, save_path)

            end_picture = time.time()
            total_time = end_picture - start_picture
            print("一帧图像完整处理花费的时间：", total_time)

    释放视频文件
    serial_thread.terminate()
    video_capture.release()
    print(f"视频读帧图像保存至 {output_path_video}")

    sys.exit(app.exec_())


if __name__ == "__main__":
    # 保存原始的stdout
    original_stdout = sys.stdout
    # 打开文件，准备写入
    with open('logging_print.txt', 'w') as f:
        sys.stdout = f  # 重定向stdout到文件
        build_infer()
        # 重置stdout到原始状态
        sys.stdout = original_stdout
    # build_infer()