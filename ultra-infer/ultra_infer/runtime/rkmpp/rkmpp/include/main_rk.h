#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include "ultra_infer/runtime/ThreadSafeQueue.h"
#include <atomic>

void RkMpp(std::string url_str,ThreadSafeQueue* q ,std::atomic<bool>& stop,std::string ip = "192.168.1.35",std::string password = "firefly");