// #pragma once

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <vector>
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include "rknn_api.h"

// class PPSegmentor {
// public:
//     PPSegmentor();
//     ~PPSegmentor();

//     int Init(const std::string& model_path);

//     int Predict(const cv::Mat& src_img, cv::Mat& result_mask);

// private:
//     rknn_context m_ctx;
//     rknn_input_output_num m_io_num;
//     rknn_tensor_attr* m_input_attrs;
//     rknn_tensor_attr* m_output_attrs;

//     int m_model_w;
//     int m_model_h;
//     int m_model_c;
//     bool m_is_init;

//     static const std::vector<cv::Vec3b> m_color_table;

//     unsigned char* load_model(const char* filename, int* model_size);
//     void dump_tensor_attr(rknn_tensor_attr* attr);
// };


#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

class PaddleSegSarRKNN {
public:
    // 构造函数不再需要传入 class_names
    explicit PaddleSegSarRKNN(const std::string& model_path);

    ~PaddleSegSarRKNN();

    // 初始化模型（必须调用）
    int Init();

    // 预测接口
    std::vector<cv::Mat> Predict(const cv::Mat& src_img);

private:

    const std::vector<std::string> m_class_names = {
        "water", "forest", "Bareland", 
        "Road", "Building", "Mountain"
    };

    std::string m_model_path;

    rknn_context m_ctx = 0;
    rknn_input_output_num m_io_num = {0};
    rknn_tensor_attr* m_input_attrs = nullptr;
    rknn_tensor_attr* m_output_attrs = nullptr;

    int m_model_w = 0;
    int m_model_h = 0;

    // 加载模型文件
    unsigned char* load_model(const char* filename, int* size);

    // 禁止拷贝和赋值
    PaddleSegSarRKNN(const PaddleSegSarRKNN&) = delete;
    PaddleSegSarRKNN& operator=(const PaddleSegSarRKNN&) = delete;
};