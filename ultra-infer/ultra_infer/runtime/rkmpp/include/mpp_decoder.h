#ifndef __MPP_DECODER_H__
#define __MPP_DECODER_H__

#include <string.h>

#include <unistd.h>
#include <pthread.h>
#include <stddef.h>
#include <iostream>
#include <stdint.h>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "rockchip/mpp_buffer.h"
#include "rockchip/rk_mpi.h"
#include "rockchip/mpp_log.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/rk_mpi.h"
#include "rockchip/mpp_frame.h"
// #include "common.h"

// #include "rga.h"
// #include "RgaUtils.h"
// #include "im2d_type.h"
// #include "im2d_buffer.h"
// #include "im2d_single.h"
// #include "dma_alloc.h"
// #include "drm_alloc.hpp"
// #include "drm_fourcc.h"
// #include "drm_mode.h"
// #include "drm.h"
MppCodingType ffmpeg2NvCodecId(int ffmpeg_codec_id);


namespace FFHD_DECODER {
    
    #define msleep(x)                   usleep((x)*1000)
    #define MPI_DEC_STREAM_SIZE         (SZ_4K)
    #define MPI_DEC_LOOP_COUNT          4
    #define MAX_FILE_NAME_LENGTH        256

    typedef struct
    {
        MppCtx          ctx;
        MppApi          *mpi;
        RK_U32          eos;
        char            *buf;

        MppBufferGroup  frm_grp;
        MppBufferGroup  pkt_grp;
        MppPacket       packet;
        size_t          packet_size;
        MppFrame        frame;

        RK_S32          frame_count;
        RK_S32          frame_num;
        size_t          max_usage;
    } MpiDecLoopData;

    class mpp_decoder
    {
    public:
        // MppCtx mpp_ctx          = NULL;
        // MppApi *mpp_mpi         = NULL;
        mpp_decoder();
        ~mpp_decoder();
        bool open(int width, int height, int fps, int decod_type);
        int Decode(uint8_t* pkt_data, int pkt_size, std::vector<cv::Mat>& out_yuvs);
   
        void set_eos(){ pkt_eos = 1;};
    private:
        int pkt_eos = 0;

        RK_U32 width_mpp;
        RK_U32 height_mpp;
        MppCodingType mpp_type;
        size_t packet_size  = 2400*1300*3/2;
        MpiDecLoopData loop_data;
        int fps = -1;
        // int need_frame_count = 0;
        // int fps_scale = 1; // fps缩放比例
    };
    std::shared_ptr<mpp_decoder> create_mpp_decoder(int width, int height, int fps, int decod_type);
}

#endif //__MPP_DECODER_H__
