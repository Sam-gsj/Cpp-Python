#include "ffmpeg_demuxer.hpp"
#include "mpp_decoder.h"
#include "opencv2/opencv.hpp"
#include "Log.h"
#include "Initializers/RollingFileInitializer.h"
#include "Appenders/ColorConsoleAppender.h"
#include "main_rk.h"


using namespace std;
using namespace cv;

#define __ARM_NEON 1
#define __aarch64__ 1
static void yuv420sp2rgb_nv12(const unsigned char* yuv420sp, unsigned char* rgb, int w, int h)
{
    const unsigned char* yptr = yuv420sp;
    const unsigned char* uvptr = yuv420sp + w * h;

#if __ARM_NEON
    uint8x8_t _v128 = vdup_n_u8(128);
    int8x8_t _v90 = vdup_n_s8(90);
    int8x8_t _v46 = vdup_n_s8(46);
    int8x8_t _v22 = vdup_n_s8(22);
    int8x8_t _v113 = vdup_n_s8(113);
#endif // __ARM_NEON

    for (int y = 0; y < h; y += 2)
    {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = rgb;
        unsigned char* rgb1 = rgb + w * 3;

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(uvptr), _v128));
            int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
            int8x8_t _uu = _uuuuvvvv.val[0];
            int8x8_t _vv = _uuuuvvvv.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _v90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _v46);
            _g0 = vmlsl_s8(_g0, _uu, _v22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _v113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _v90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _v46);
            _g1 = vmlsl_s8(_g1, _uu, _v22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _v113);

            uint8x8x3_t _rgb0;
            _rgb0.val[0] = vqshrun_n_s16(_r0, 6);
            _rgb0.val[1] = vqshrun_n_s16(_g0, 6);
            _rgb0.val[2] = vqshrun_n_s16(_b0, 6);

            uint8x8x3_t _rgb1;
            _rgb1.val[0] = vqshrun_n_s16(_r1, 6);
            _rgb1.val[1] = vqshrun_n_s16(_g1, 6);
            _rgb1.val[2] = vqshrun_n_s16(_b1, 6);

            vst3_u8(rgb0, _rgb0);
            vst3_u8(rgb1, _rgb1);

            yptr0 += 8;
            yptr1 += 8;
            uvptr += 8;
            rgb0 += 24;
            rgb1 += 24;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%3, #128]          \n"
                "vld1.u8    {d2}, [%3]!         \n"
                "vsub.s8    d2, d2, %12         \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0}, [%1]!         \n"
                "pld        [%2, #128]          \n"
                "vld1.u8    {d1}, [%2]!         \n"
                "vshll.u8   q2, d0, #6          \n"
                "vorr       d3, d2, d2          \n"
                "vshll.u8   q3, d1, #6          \n"
                "vorr       q9, q2, q2          \n"
                "vtrn.s8    d2, d3              \n"
                "vorr       q11, q3, q3         \n"
                "vmlsl.s8   q9, d3, %14         \n"
                "vorr       q8, q2, q2          \n"
                "vmlsl.s8   q11, d3, %14        \n"
                "vorr       q10, q3, q3         \n"
                "vmlal.s8   q8, d3, %13         \n"
                "vmlal.s8   q2, d2, %16         \n"
                "vmlal.s8   q10, d3, %13        \n"
                "vmlsl.s8   q9, d2, %15         \n"
                "vmlal.s8   q3, d2, %16         \n"
                "vmlsl.s8   q11, d2, %15        \n"
                "vqshrun.s16 d24, q8, #6        \n"
                "vqshrun.s16 d26, q2, #6        \n"
                "vqshrun.s16 d4, q10, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"
                "vqshrun.s16 d6, q3, #6         \n"
                "vqshrun.s16 d5, q11, #6        \n"
                "subs       %0, #1              \n"
                "vst3.u8    {d24-d26}, [%4]!    \n"
                "vst3.u8    {d4-d6}, [%5]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),    // %0
                "=r"(yptr0), // %1
                "=r"(yptr1), // %2
                "=r"(uvptr), // %3
                "=r"(rgb0),  // %4
                "=r"(rgb1)   // %5
                : "0"(nn),
                "1"(yptr0),
                "2"(yptr1),
                "3"(uvptr),
                "4"(rgb0),
                "5"(rgb1),
                "w"(_v128), // %12
                "w"(_v90),  // %13
                "w"(_v46),  // %14
                "w"(_v22),  // %15
                "w"(_v113)  // %16
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "d26");
        }
#endif // __aarch64__
#endif // __ARM_NEON

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
        for (; remain > 0; remain -= 2)
        {
            // R = 1.164 * yy + 1.596 * vv
            // G = 1.164 * yy - 0.813 * vv - 0.391 * uu
            // B = 1.164 * yy              + 2.018 * uu

            // R = Y + (1.370705 * (V-128))
            // G = Y - (0.698001 * (V-128)) - (0.337633 * (U-128))
            // B = Y + (1.732446 * (U-128))

            // R = ((Y << 6) + 87.72512 * (V-128)) >> 6
            // G = ((Y << 6) - 44.672064 * (V-128) - 21.608512 * (U-128)) >> 6
            // B = ((Y << 6) + 110.876544 * (U-128)) >> 6

            // R = ((Y << 6) + 90 * (V-128)) >> 6
            // G = ((Y << 6) - 46 * (V-128) - 22 * (U-128)) >> 6
            // B = ((Y << 6) + 113 * (U-128)) >> 6

            // R = (yy + 90 * vv) >> 6
            // G = (yy - 46 * vv - 22 * uu) >> 6
            // B = (yy + 113 * uu) >> 6

            int u = uvptr[0] - 128;
            int v = uvptr[1] - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            rgb0[0] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
            rgb0[1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
            rgb0[2] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

            int y01 = yptr0[1] << 6;
            rgb0[3] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
            rgb0[4] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
            rgb0[5] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

            int y10 = yptr1[0] << 6;
            rgb1[0] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
            rgb1[1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
            rgb1[2] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

            int y11 = yptr1[1] << 6;
            rgb1[3] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
            rgb1[4] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
            rgb1[5] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

            yptr0 += 2;
            yptr1 += 2;
            uvptr += 2;
            rgb0 += 6;
            rgb1 += 6;
        }
#undef SATURATE_CAST_UCHAR

        yptr += 2 * w;
        rgb += 2 * 3 * w;
    }
}


void RkMpp(std::string url_str,ThreadSafeQueue* q ,std::atomic<bool>& stop, std::string ip ,std::string password)
{
    while (true && !stop.load()) {
        try{
            std::string cmd = "echo '" + password + "' | sudo -S ifconfig eth0 " + ip + " netmask 255.255.255.0 up";
            int result = system(cmd.c_str());
            if (result == -1) {
                std::cerr << "设置IP执行失败！" << std::endl;
            } else {
                std::cout << "设置IP执行完毕:"<< ip <<"退出状态: "  << result << std::endl;
            }
            static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; // Create the 2nd appender.
            plog::init(plog::debug).addAppender(&consoleAppender); // Initialize the logger with the both appenders.

            auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(url_str, true);
            if(demuxer == nullptr){
                PLOGE << "demuxer create failed";
                continue;
            }
            int width = demuxer->get_width();
            int height = demuxer->get_height();
            int fps = demuxer->get_fps();
            auto decoder = FFHD_DECODER::create_mpp_decoder( width, height,
                        fps, demuxer->get_video_codec());
            if(decoder == nullptr){
                PLOGE << "decoder create failed";

            }
            LOGI << "w: " << width << " h: " << height << " fps: " << fps;
            uint8_t* packet_data = nullptr;
            int packet_size = 0;
            int64_t pts = 0;

            demuxer->get_extra_data(&packet_data, &packet_size);
            int frame_index = 0;
            vector<Mat> que_mat;
            decoder->Decode(packet_data, packet_size, que_mat);
            que_mat.clear();
            do{
                demuxer->demux(&packet_data, &packet_size, &pts);
                if (packet_size < 0)
                {
                    que_mat.clear();
                    break;
                }

                frame_index++;
                decoder->Decode(packet_data, packet_size, que_mat);

                std::string path = "./output/gsj" + std::to_string(frame_index) + ".jpg";
                for(auto& item: que_mat){
                    Mat image;
                    cvtColor(item, image, cv::COLOR_YUV2BGR_NV12);
                    q->push(image.clone()); 
                }

                que_mat.clear();
            }while(packet_size > 0 && !stop.load());
        } catch (const std::exception& e) {
            PLOGE << "rkmpp 执行异常: " << e.what() << "，准备重试...";
        } catch (...) {
            PLOGE << "发生未知异常，准备重试...";
        }
    }
}

// std::vector<cv::Mat> RkMpp(std::string url_str,ThreadSafeQueue* q )
// {
//     static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; // Create the 2nd appender.
//     plog::init(plog::debug).addAppender(&consoleAppender); // Initialize the logger with the both appenders.

//     auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(url_str, true);
//     if(demuxer == nullptr){
//         PLOGE << "demuxer create failed";
//     }
//     int width = demuxer->get_width();
//     int height = demuxer->get_height();
//     int fps = demuxer->get_fps();
//     auto decoder = FFHD_DECODER::create_mpp_decoder( width, height,
//                 fps, demuxer->get_video_codec());
//     if(decoder == nullptr){
//         PLOGE << "decoder create failed";
       
//     }
//     LOGI << "w: " << width << " h: " << height << " fps: " << fps;
//     uint8_t* packet_data = nullptr;
//     int packet_size = 0;
//     int64_t pts = 0;
    
//     demuxer->get_extra_data(&packet_data, &packet_size);
//     int frame_index = 0;
//     vector<Mat> que_mat;
//     decoder->Decode(packet_data, packet_size, que_mat);
//     que_mat.clear();
//     do{
//         demuxer->demux(&packet_data, &packet_size, &pts);
//         if (packet_size < 0)
//         {
//             que_mat.clear();
//             break;
//         }
        
//         frame_index++;
//         decoder->Decode(packet_data, packet_size, que_mat);
//         std::cout<<"size" << que_mat.size()<<std::endl;
//         std::string path = "./output/gsj" + std::to_string(frame_index) + ".jpg";
//         for(auto& item: que_mat){
//             Mat image;
//             cvtColor(item, image, cv::COLOR_YUV2BGR_NV12);
//             q->push(image.clone()); 
//         }
//         // if(que_mat.size()!=0){
//         //     Mat image;
//         //     cvtColor(que_mat[0], image, CV_YUV2BGR_NV12);
//         //     cv::imwrite(path, image);
//         // }
//         que_mat.clear();
//     }while(packet_size > 0 );
// }