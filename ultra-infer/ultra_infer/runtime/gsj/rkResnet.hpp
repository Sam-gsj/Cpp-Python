#pragma once

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "const.hpp"
#include <mutex>
#include <string>

#include "utils.hpp"
#include "parallel.h" 
#include "postprocess.h"

class rkResnet
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data = nullptr;
    rknn_context ctx = 0;
    
    rknn_input_output_num io_num = {0};
    rknn_tensor_attr *input_attrs = nullptr;
    rknn_tensor_attr *output_attrs = nullptr;
    rknn_input inputs[1];

    int channel = 0, width = 0, height = 0;
    int img_width = 0, img_height = 0;

public:
    // 构造函数
    rkResnet(const std::string &model_path);
    rkResnet(const rkResnet&) = delete;
    rkResnet& operator=(const rkResnet&) = delete;

    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    resnet_results Predict(resnet_input& input);
    ~rkResnet();
};


class SuperrkResnet : public AutoParallelSimpleInferencePredictor<rkResnet, const std::string &, resnet_input, resnet_results>{

public:
    SuperrkResnet(const std::string& model_path,int thread_num = 3):AutoParallelSimpleInferencePredictor(model_path,thread_num){};
    std::vector<cv::Mat> Predict(cv::Mat input_image,int ROWS, int COLS){
        std::vector<resnet_input> inputs = split_image(input_image, ROWS, COLS);
        size_t task_count = inputs.size();

        for (auto& item : inputs) {
            if (item.img.empty() || item.img.cols == 0) {
                item.img = cv::Mat::zeros(32, 32, CV_8UC3);
            }
        }
        for (const auto& input : inputs) {
            AutoParallelSimpleInferencePredictor::PredictThread(input);
        }
        std::vector<resnet_results> results_vec;
        results_vec.reserve(task_count);

        for (size_t i = 0; i < task_count; ++i) {
            resnet_results res;
        
            if (AutoParallelSimpleInferencePredictor::GetResult(res)) {
                results_vec.push_back(res);
            } else {
                resnet_results dummy;
                dummy.id = inputs[i].id; 
                results_vec.push_back(dummy);
            }
        }

        auto outputs = synthesize_image(inputs, results_vec, ROWS, COLS);
        return outputs;
    }

};

