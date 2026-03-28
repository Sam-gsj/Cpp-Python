#include "ppseg.h"

PaddleSegRKNN::PaddleSegRKNN(const std::string& model_path) 
    : m_model_path(model_path), m_ctx(0), 
      m_input_attrs(nullptr), m_output_attrs(nullptr), m_model_w(0), m_model_h(0), m_is_init(false) {
    m_class_names = {
        "urban_land", "agriculture_land", "rangeland", 
        "forest_land", "water", "barren_land", "unknown"
    };
    Init();
}

PaddleSegRKNN::~PaddleSegRKNN() {
    if (m_input_attrs) free(m_input_attrs);
    if (m_output_attrs) free(m_output_attrs);
    if (m_ctx > 0) rknn_destroy(m_ctx);
}

unsigned char* PaddleSegRKNN::load_model(const char* filename, int* size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(*size);
    if (fread(data, 1, *size, fp) != (size_t)*size) {
        free(data);
        fclose(fp);
        return nullptr;
    }
    fclose(fp);
    return data;
}

int PaddleSegRKNN::Init() {
    int model_len = 0;
    unsigned char* model_data = load_model(m_model_path.c_str(), &model_len);
    if (!model_data) return -1;

    int ret = rknn_init(&m_ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) return -1;

    rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &m_io_num, sizeof(m_io_num));
    m_input_attrs = (rknn_tensor_attr*)malloc(m_io_num.n_input * sizeof(rknn_tensor_attr));
    m_output_attrs = (rknn_tensor_attr*)malloc(m_io_num.n_output * sizeof(rknn_tensor_attr));

    for (int i = 0; i < (int)m_io_num.n_input; i++) {
        m_input_attrs[i].index = i;
        rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &m_input_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < (int)m_io_num.n_output; i++) {
        m_output_attrs[i].index = i;
        rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &m_output_attrs[i], sizeof(rknn_tensor_attr));
    }

    if (m_input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        m_model_w = m_input_attrs[0].dims[2];
        m_model_h = m_input_attrs[0].dims[3];
    } else {
        m_model_w = m_input_attrs[0].dims[1];
        m_model_h = m_input_attrs[0].dims[2];
    }

    m_is_init = true;
    return 0;
}

std::vector<cv::Mat> PaddleSegRKNN::Predict(const cv::Mat& src_img) {
    if (!m_is_init || src_img.empty()) return {};

    int orig_h = src_img.rows;
    int orig_w = src_img.cols;

    cv::Mat img_resized;
    cv::resize(src_img, img_resized, cv::Size(m_model_w, m_model_h));
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = m_model_w * m_model_h * 3;
    inputs[0].buf   = img_resized.data;
    rknn_inputs_set(m_ctx, m_io_num.n_input, inputs);

    rknn_run(m_ctx, NULL);

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    rknn_outputs_get(m_ctx, 1, outputs, NULL);

    float* output_data = (float*)outputs[0].buf;
    int num_classes = m_output_attrs[0].dims[1];
    cv::Mat pred_id_map = cv::Mat::zeros(m_model_h, m_model_w, CV_8UC1);

    for (int i = 0; i < m_model_h; i++) {
        for (int j = 0; j < m_model_w; j++) {
            float max_val = -1e10;
            int max_cls = 0;
            for (int k = 0; k < num_classes; k++) {
                float val = output_data[k * m_model_h * m_model_w + i * m_model_w + j];
                if (val > max_val) {
                    max_val = val;
                    max_cls = k;
                }
            }
            pred_id_map.at<uchar>(i, j) = (uchar)max_cls;
        }
    }

    cv::Mat pred_resized;
    cv::resize(pred_id_map, pred_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

    std::vector<cv::Mat> mask_list;
    for (int cls_id = 0; cls_id < (int)m_class_names.size(); cls_id++) {
        cv::Mat binary_mask(orig_h, orig_w, CV_8UC1, cv::Scalar(0));
        binary_mask.setTo(255, pred_resized == cls_id);
        mask_list.push_back(binary_mask);
    }

    rknn_outputs_release(m_ctx, 1, outputs);
    return mask_list;
}