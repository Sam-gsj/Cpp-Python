#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

class DeepGlobeRKNN {
public:
    DeepGlobeRKNN(const std::string& model_path);
    ~DeepGlobeRKNN();
    int Init();
    std::vector<cv::Mat> Predict(const cv::Mat& src_img);
    const std::vector<std::string>& GetClassNames() const { return m_class_names; }

private:
    std::string m_model_path;
    std::vector<std::string> m_class_names;
    rknn_context m_ctx;
    rknn_input_output_num m_io_num;
    rknn_tensor_attr *m_input_attrs;
    rknn_tensor_attr *m_output_attrs;
    int m_model_w;
    int m_model_h;
    bool m_is_init;

    unsigned char* load_model(const char* filename, int* size);
};
