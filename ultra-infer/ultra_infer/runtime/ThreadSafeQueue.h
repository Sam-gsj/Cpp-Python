#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stack>
#include <deque>


#pragma once

// class ThreadSafeQueue {
// public:
//     void push(cv::Mat value) {
//         std::lock_guard<std::mutex> lock(mutex_);
//         queue_.push(std::move(value));
//         // if(queue_.size()>100){
//         //     queue_.pop();
//         // }
//         cond_.notify_one(); // 通知一个等待的线程
//     }

//     bool try_pop(cv::Mat& value) {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (queue_.empty()) {
//             return false;
//         }
//         value = std::move(queue_.front());
//         queue_.pop();
//         return true;
//     }

//     void wait_and_pop(cv::Mat& value) {
//         std::unique_lock<std::mutex> lock(mutex_);
//         cond_.wait(lock, [this] { return !queue_.empty(); }); // 等待直到队列不为空
//         value = std::move(queue_.front());
//         queue_.pop();
//     }

// private:
//     std::queue<cv::Mat> queue_;
//     std::mutex mutex_;
//     std::condition_variable cond_;
// };


// class ThreadSafeQueue { 
// public:
//     void push(cv::Mat value) {
//         std::lock_guard<std::mutex> lock(mutex_);
//         stack_.push(std::move(value)); 
//         cond_.notify_one();
//     }

//     // try_pop 依然是从顶部弹出
//     bool try_pop(cv::Mat& value) {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (stack_.empty()) { 
//             return false;
//         }
        
//         value = std::move(stack_.top()); 
//         stack_.pop();
//         return true;
//     }

//     void wait_and_pop(cv::Mat& value) {
//         std::unique_lock<std::mutex> lock(mutex_);
      
//         cond_.wait(lock, [this] { return !stack_.empty(); }); 
        
    
//         value = std::move(stack_.top());
//         stack_.pop();
//     }

// private:
//     std::stack<cv::Mat> stack_; 
//     std::mutex mutex_;
//     std::condition_variable cond_;
// };


class ThreadSafeQueue { 
public:
    // 限制最大容量为100，超过时移除最旧元素（队首）
    void push(cv::Mat value) {
        std::lock_guard<std::mutex> lock(mutex_);
        // 当容量达到100时，先移除最旧的元素（队首）
        if (deque_.size() >= 100) {
            deque_.pop_front(); // 弹出最旧元素
            // std::cout << "pop " <<std::endl;
        }
        deque_.push_back(std::move(value)); // 新元素加入队尾（成为最新元素）
        cond_.notify_one();
    }

    // 从顶部（队尾，最新元素）弹出
    bool try_pop(cv::Mat& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (deque_.empty()) { 
            return false;
        }
        
        value = std::move(deque_.back()); // 获取最新元素（队尾）
        deque_.pop_back(); // 移除最新元素
        return true;
    }

    // 从顶部（队尾，最新元素）弹出，若为空则等待
    void wait_and_pop(cv::Mat& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 等待直到队列非空
        cond_.wait(lock, [this] { return !deque_.empty(); }); 
        
        value = std::move(deque_.back()); // 获取最新元素（队尾）
        deque_.pop_back(); // 移除最新元素
    }

private:
    std::deque<cv::Mat> deque_; // 使用双端队列，支持高效的首尾操作
    std::mutex mutex_;
    std::condition_variable cond_;
};