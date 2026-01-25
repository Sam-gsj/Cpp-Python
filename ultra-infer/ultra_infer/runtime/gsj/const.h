#pragma once 
#include <vector>
#include <opencv2/opencv.hpp>


#define CLASS_NUM  1





#define BLOCK_GAP 40
#define CLEAR_WHITE 3

struct element_t{
    float value;
    int index;
};


struct resnet_result {
    int cls;
    float score;
} ;

struct resnet_results
{
    resnet_result result[CLASS_NUM];
    int id;
};


struct resnet_input {
    cv::Mat img;
    int id;
    resnet_input(cv::Mat img, int id) : img(img), id(id) {}
};

