// #include "ppseg.h"
// #include <chrono>


// const std::vector<cv::Vec3b> PPSegmentor::m_color_table = {
//     cv::Vec3b(255, 255, 0),   // 0: urban_land
//     cv::Vec3b(0, 255, 255),   // 1: agriculture_land
//     cv::Vec3b(255, 0, 255),   // 2: rangeland
//     cv::Vec3b(0, 255, 0),     // 3: forest_land
//     cv::Vec3b(0, 0, 255),     // 4: water 
//     cv::Vec3b(255, 255, 255), // 5: barren_land
//     cv::Vec3b(0, 0, 0)        // 6: unknown
// };

// PPSegmentor::PPSegmentor() 
//     : m_ctx(0), m_input_attrs(nullptr), m_output_attrs(nullptr), 
//       m_model_w(0), m_model_h(0), m_model_c(0), m_is_init(false) {}

// PPSegmentor::~PPSegmentor() {
//     if (m_input_attrs) free(m_input_attrs);
//     if (m_output_attrs) free(m_output_attrs);
//     if (m_ctx > 0) rknn_destroy(m_ctx);
// }

// unsigned char* PPSegmentor::load_model(const char* filename, int* model_size) {
//     FILE* fp = fopen(filename, "rb");
//     if (fp == NULL) return NULL;
//     fseek(fp, 0, SEEK_END);
//     int size = ftell(fp);
//     fseek(fp, 0, SEEK_SET);
//     unsigned char* data = (unsigned char*)malloc(size);
//     if (fread(data, 1, size, fp) != size) {
//         free(data);
//         fclose(fp);
//         return NULL;
//     }
//     fclose(fp);
//     *model_size = size;
//     return data;
// }

// void PPSegmentor::dump_tensor_attr(rknn_tensor_attr* attr) {
//     printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s\n",
//            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
//            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type));
// }

// int PPSegmentor::Init(const std::string& model_path) {
//     int ret;
//     int model_len = 0;
//     unsigned char* model_data = load_model(model_path.c_str(), &model_len);
//     if (!model_data) {
//         std::cerr << "Read model file failed: " << model_path << std::endl;
//         return -1;
//     }

//     ret = rknn_init(&m_ctx, model_data, model_len, 0, NULL);
//     free(model_data);
//     if (ret < 0) {
//         std::cerr << "rknn_init fail! ret=" << ret << std::endl;
//         return -1;
//     }


//     ret = rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &m_io_num, sizeof(m_io_num));
    
//     m_input_attrs = (rknn_tensor_attr*)malloc(m_io_num.n_input * sizeof(rknn_tensor_attr));
//     m_output_attrs = (rknn_tensor_attr*)malloc(m_io_num.n_output * sizeof(rknn_tensor_attr));

//     for (int i = 0; i < m_io_num.n_input; i++) {
//         m_input_attrs[i].index = i;
//         rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &(m_input_attrs[i]), sizeof(rknn_tensor_attr));
//         dump_tensor_attr(&(m_input_attrs[i]));
//     }
    
//     for (int i = 0; i < m_io_num.n_output; i++) {
//         m_output_attrs[i].index = i;
//         rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &(m_output_attrs[i]), sizeof(rknn_tensor_attr));
//         dump_tensor_attr(&(m_output_attrs[i]));
//     }

//     if (m_input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
//         m_model_w = m_input_attrs[0].dims[2];
//         m_model_h = m_input_attrs[0].dims[3];
//         m_model_c = m_input_attrs[0].dims[1];
//     } else {
//         m_model_w = m_input_attrs[0].dims[1];
//         m_model_h = m_input_attrs[0].dims[2];
//         m_model_c = m_input_attrs[0].dims[3];
//     }

//     m_is_init = true;
//     return 0;
// }

// int PPSegmentor::Predict(const cv::Mat& src_img, cv::Mat& result_mask) {
//     if (!m_is_init) return -1;

//     cv::Mat img_resized;
//     cv::resize(src_img, img_resized, cv::Size(m_model_w, m_model_h));
//     cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

//     rknn_input inputs[1];
//     memset(inputs, 0, sizeof(inputs));
//     inputs[0].index = 0;
//     inputs[0].type  = RKNN_TENSOR_UINT8; 
//     inputs[0].fmt   = RKNN_TENSOR_NHWC;
//     inputs[0].size  = m_model_w * m_model_h * m_model_c;
//     inputs[0].buf   = img_resized.data;
//     rknn_inputs_set(m_ctx, m_io_num.n_input, inputs);


//     rknn_run(m_ctx, NULL);

//     rknn_output outputs[1];
//     memset(outputs, 0, sizeof(outputs));
//     outputs[0].want_float = 1; 
//     int ret = rknn_outputs_get(m_ctx, 1, outputs, NULL);
//     if (ret < 0) return -1;

//     int num_classes = m_output_attrs[0].dims[1];
//     float* output_data = (float*)outputs[0].buf;
//     cv::Mat color_mask = cv::Mat::zeros(m_model_h, m_model_w, CV_8UC3);

//     for (int i = 0; i < m_model_h; i++) {
//         for (int j = 0; j < m_model_w; j++) {
//             float max_val = -1e10;
//             int max_cls = 0;
//             for (int k = 0; k < num_classes; k++) {
//                 float val = output_data[k * m_model_h * m_model_w + i * m_model_w + j];
//                 if (val > max_val) {
//                     max_val = val;
//                     max_cls = k;
//                 }
//             }
//             if (max_cls < (int)m_color_table.size()) {
//                 color_mask.at<cv::Vec3b>(i, j) = m_color_table[max_cls];
//             }
//         }
//     }

//     cv::resize(color_mask, result_mask, src_img.size(), 0, 0, cv::INTER_NEAREST);
//     rknn_outputs_release(m_ctx, 1, outputs);
//     return 0;
// }


#include "ppseg_sar.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

PaddleSegSarRKNN::PaddleSegSarRKNN(const std::string& model_path)
    : m_model_path(model_path) 
{
    m_input_attrs = nullptr;
    m_output_attrs = nullptr;
    Init();
}

PaddleSegSarRKNN::~PaddleSegSarRKNN() 
{
    if (m_input_attrs) {
        free(m_input_attrs);
        m_input_attrs = nullptr;
    }
    if (m_output_attrs) {
        free(m_output_attrs);
        m_output_attrs = nullptr;
    }
    if (m_ctx > 0) {
        rknn_destroy(m_ctx);
        m_ctx = 0;
    }
}

int PaddleSegSarRKNN::Init() 
{
    int model_len = 0;
    unsigned char* model_data = load_model(m_model_path.c_str(), &model_len);
    if (!model_data) {
        std::cerr << "Failed to load model: " << m_model_path << std::endl;
        return -1;
    }

    int ret = rknn_init(&m_ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) {
        std::cerr << "rknn_init failed! ret = " << ret << std::endl;
        return -1;
    }

    // 查询输入输出数量
    rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &m_io_num, sizeof(m_io_num));

    // 分配属性内存
    m_input_attrs = (rknn_tensor_attr*)malloc(m_io_num.n_input * sizeof(rknn_tensor_attr));
    m_output_attrs = (rknn_tensor_attr*)malloc(m_io_num.n_output * sizeof(rknn_tensor_attr));

    if (!m_input_attrs || !m_output_attrs) {
        std::cerr << "malloc failed for tensor attrs" << std::endl;
        return -1;
    }

    // 查询输入属性
    for (uint32_t i = 0; i < m_io_num.n_input; i++) {
        m_input_attrs[i].index = i;
        rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &m_input_attrs[i], sizeof(rknn_tensor_attr));
    }

    // 查询输出属性
    for (uint32_t i = 0; i < m_io_num.n_output; i++) {
        m_output_attrs[i].index = i;
        rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &m_output_attrs[i], sizeof(rknn_tensor_attr));
    }

    // 获取模型输入宽高（支持 NCHW 和 NHWC）
    if (m_input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        m_model_w = m_input_attrs[0].dims[2];  // width
        m_model_h = m_input_attrs[0].dims[3];  // height
    } else {  // NHWC
        m_model_w = m_input_attrs[0].dims[1];
        m_model_h = m_input_attrs[0].dims[2];

    }

    std::cout << "Model input size: " << m_model_w << " x " << m_model_h << std::endl;
    return 0;
}

std::vector<cv::Mat> PaddleSegSarRKNN::Predict(const cv::Mat& src_img) 
{
    if (src_img.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return {};
    }

    int orig_h = src_img.rows;
    int orig_w = src_img.cols;

    // 预处理：resize + BGR→RGB
    cv::Mat img_resized;
    cv::resize(src_img, img_resized, cv::Size(m_model_w, m_model_h));
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt  = RKNN_TENSOR_NHWC;
    inputs[0].size = m_model_w * m_model_h * 3;
    inputs[0].buf  = img_resized.data;

    rknn_inputs_set(m_ctx, m_io_num.n_input, inputs);

    // 推理
    rknn_run(m_ctx, NULL);

    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    rknn_outputs_get(m_ctx, 1, outputs, NULL);

    float* output_data = (float*)outputs[0].buf;
    int num_classes = m_output_attrs[0].dims[1];   // 类别数

    // 计算 argmax 得到类别 ID 图
    cv::Mat pred_id_map = cv::Mat::zeros(m_model_h, m_model_w, CV_8UC1);
    for (int i = 0; i < m_model_h; i++) {
        for (int j = 0; j < m_model_w; j++) {
            float max_val = -1e10f;
            int max_cls = 0;
            for (int k = 0; k < num_classes; k++) {
                float val = output_data[k * m_model_h * m_model_w + i * m_model_w + j];
                if (val > max_val) {
                    max_val = val;
                    max_cls = k;
                }
            }
            pred_id_map.at<uchar>(i, j) = static_cast<uchar>(max_cls);
        }
    }

    // resize 回原图大小（最近邻插值，保持类别标签不变）
    cv::Mat pred_resized;
    cv::resize(pred_id_map, pred_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

    // 生成每个类别的二值掩码
    std::vector<cv::Mat> mask_list;
    mask_list.reserve(m_class_names.size());

    for (size_t cls_id = 0; cls_id < m_class_names.size(); ++cls_id) {
        cv::Mat binary_mask(orig_h, orig_w, CV_8UC1, cv::Scalar(0));
        binary_mask.setTo(255, pred_resized == static_cast<int>(cls_id));
        mask_list.push_back(binary_mask);
    }

    rknn_outputs_release(m_ctx, 1, outputs);
    return mask_list;
}

// 私有辅助函数
unsigned char* PaddleSegSarRKNN::load_model(const char* filename, int* size) 
{
    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(*size);
    if (!data) {
        fclose(fp);
        return nullptr;
    }

    if (fread(data, 1, *size, fp) != static_cast<size_t>(*size)) {
        free(data);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    return data;
}