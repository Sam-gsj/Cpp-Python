#include <stdio.h>
#include <sys/time.h>
#include "mpp_decoder.h"
#include <unistd.h>
#include <pthread.h>
#include <sys/syscall.h>
#include "Log.h"
#include <iostream>

using namespace std; 

static unsigned long GetCurrentTimeMS() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000+tv.tv_usec/1000;
}
MppCodingType ffmpeg2NvCodecId(int ffmpeg_codec_id)
{
    switch (ffmpeg_codec_id) {    
        /*AV_CODEC_ID_H264*/ case 27        : return MPP_VIDEO_CodingAVC;         
        /*AV_CODEC_ID_HEVC*/ case 173       : return MPP_VIDEO_CodingHEVC;         
        /*AV_CODEC_ID_VP8*/ case 139        : return MPP_VIDEO_CodingVP8;          
        /*AV_CODEC_ID_VP9*/ case 167        : return MPP_VIDEO_CodingVP9;     
        /*AV_CODEC_ID_MJPEG*/ case 7        : return MPP_VIDEO_CodingMJPEG; 
        default                             : return MPP_VIDEO_CodingUnused;         
    }
}

namespace FFHD_DECODER {
    mpp_decoder::mpp_decoder()
    {
    }
    mpp_decoder::~mpp_decoder()
    {
        loop_data.mpi->reset(loop_data.ctx);
        if (loop_data.packet) {
            // LOGI << "1";
            mpp_packet_deinit(&loop_data.packet);
            loop_data.packet = NULL;
        }
        
        if (loop_data.frame) {
            // LOGI << "2";
            mpp_frame_deinit(&loop_data.frame);
            loop_data.frame = NULL;
        }
        if (loop_data.ctx) {
            // LOGI << "3";
            mpp_destroy(loop_data.ctx);
            loop_data.ctx = NULL;
        }
        if (loop_data.frm_grp) {
            // LOGI << "4";
            // mpp_buffer_group_clear(loop_data.frm_grp);
            // mpp_buffer_group_unused(loop_data.frm_grp);
            mpp_buffer_group_put(loop_data.frm_grp);
            loop_data.frm_grp = NULL;
        }
    }

    bool mpp_decoder::open(int width, int height, int fps, int decod_type)
    {
        bool ok = false;
        MPP_RET ret         = MPP_OK;
        width_mpp = width;
        height_mpp = height;
        this->fps = fps;
        MppCodingType type  = ffmpeg2NvCodecId(decod_type);
        if (type == MPP_VIDEO_CodingAVC)
        {
            //cout << "width: " << width_mpp << " height: " << height_mpp << " fps: " << fps << " mpp_type: H264";
        }else if (type == MPP_VIDEO_CodingHEVC)
        {
            //cout << "width: " << width_mpp << " height: " << height_mpp << " fps: " << fps << " mpp_type: H265";
        }else{
            return ok;
        }
        // cout << endl;
        mpp_type = type;
        memset(&loop_data, 0, sizeof(loop_data));
        
        ret = mpp_create(&loop_data.ctx, &loop_data.mpi);
        if (MPP_OK != ret) {
            cout << "mpp_create failed" << endl;
            return ok;
        }

        ret = mpp_init(loop_data.ctx, MPP_CTX_DEC, mpp_type);
        if (ret) {
            cout << "mpp_init failed" << endl;
            return ok;
        }

        ok = true;
        return ok;
    }
    
    int mpp_decoder::Decode(uint8_t* pkt_data, int pkt_size, std::vector<cv::Mat>& out_yuvs)
    {
        RK_U32 pkt_done = 0;
        RK_U32 err_info = 0;
        int pkt_eos  = 0;
        MPP_RET ret = MPP_OK;

        if (loop_data.packet == NULL) {
            ret = mpp_packet_init(&loop_data.packet, NULL, 0);
        }
        mpp_packet_set_data(loop_data.packet, pkt_data);
        mpp_packet_set_size(loop_data.packet, pkt_size);
        mpp_packet_set_pos(loop_data.packet, pkt_data);
        mpp_packet_set_length(loop_data.packet, pkt_size);

        if (pkt_eos)
            mpp_packet_set_eos(loop_data.packet);
        do {
            RK_S32 times = 5;
            // send the packet first if packet is not done
            if (!pkt_done) {
                ret = loop_data.mpi->decode_put_packet(loop_data.ctx, loop_data.packet);
                if (MPP_OK == ret)
                    pkt_done = 1;
            }
            // then get all available frame and release
            do {
                RK_S32 get_frm = 0;
                RK_U32 frm_eos = 0;

                try_again:
                ret = loop_data.mpi->decode_get_frame(loop_data.ctx, &loop_data.frame);
                if (MPP_ERR_TIMEOUT == ret) {
                    if (times > 0) {
                        times--;
                        usleep(2000);
                        goto try_again;
                    }
                    cout << "decode_get_frame failed too much time" << endl;
                }

                if (MPP_OK != ret) {
                    cout << "decode_get_frame failed ret " << ret << endl;
                    break;
                }

                if (loop_data.frame) {
                    RK_U32 hor_stride = mpp_frame_get_hor_stride(loop_data.frame);
                    RK_U32 ver_stride = mpp_frame_get_ver_stride(loop_data.frame);
                    RK_U32 hor_width = mpp_frame_get_width(loop_data.frame);
                    RK_U32 ver_height = mpp_frame_get_height(loop_data.frame);
                    RK_U32 buf_size = mpp_frame_get_buf_size(loop_data.frame);
                    // RK_S64 pts = mpp_frame_get_pts(frame);
                    // RK_S64 dts = mpp_frame_get_dts(frame);

                    // cout << "decoder require buffer w:h [" << hor_width << ":" << ver_height << "] stride [" << hor_stride << ":" << ver_stride << "] buf_size " << buf_size << endl;
                    if (mpp_frame_get_info_change(loop_data.frame)) {
                        // cout <<"decode_get_frame get info changed found "<< endl;
                        if (NULL == loop_data.frm_grp) {
                            /* If buffer group is not set create one and limit it */
                            ret = mpp_buffer_group_get_internal(&loop_data.frm_grp, MPP_BUFFER_TYPE_DRM);
                            if (ret) {
                                cout << "get mpp buffer group  failed ret " << ret << endl;
                                break;
                            }

                            /* Set buffer to mpp decoder */
                            ret = loop_data.mpi->control(loop_data.ctx, MPP_DEC_SET_EXT_BUF_GROUP, loop_data.frm_grp);
                            if (ret) {
                                cout << "set buffer group failed ret " << ret << endl;
                                break;
                            }
                        } else {
                            /* If old buffer group exist clear it */
                            ret = mpp_buffer_group_clear(loop_data.frm_grp);
                            if (ret) {
                                cout << "clear buffer group failed ret " << ret << endl;
                                break;
                            }
                        }

                        /* Use limit config to limit buffer count to 24 with buf_size */
                        ret = mpp_buffer_group_limit_config(loop_data.frm_grp, buf_size, 24);
                        if (ret) {
                            cout << "limit buffer group failed ret " << ret << endl;
                            break;
                        }

                        /*
                        * All buffer group config done. Set info change ready to let
                        * decoder continue decoding
                        */
                        ret = loop_data.mpi->control(loop_data.ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
                        if (ret) {
                            cout << "info change ready failed ret " << ret << endl;
                            break;
                        }
                    } else {
                        err_info = mpp_frame_get_errinfo(loop_data.frame) | mpp_frame_get_discard(loop_data.frame);
                        if (err_info) {
                            cout << "decoder_get_frame get err info:" << mpp_frame_get_errinfo(loop_data.frame) << " discard:" << mpp_frame_get_discard(loop_data.frame) << endl;
                        }
                        loop_data.frame_count++;
                        // if (loop_data.frame_count % fps_scale == 0)
                        {
                            auto buffer = mpp_frame_get_buffer(loop_data.frame);
                            char *input_data =(char *) mpp_buffer_get_ptr(buffer);
                            
                            char *base_y = input_data;
                            char *base_c = input_data + hor_stride * ver_stride;
                            cv::Mat yuvimg(ver_height * 3 / 2, hor_width, CV_8UC1);
                            int idx = 0;
                            for (int i = 0; i < ver_height; i++, base_y += hor_stride) {
                                memcpy(yuvimg.data + idx, base_y, hor_width);
                                idx += hor_width;
                            }
                            for (int i = 0; i < ver_height / 2; i++, base_c += hor_stride) {
                                memcpy(yuvimg.data + idx, base_c, hor_width);
                                idx += hor_width;
                            }
                            out_yuvs.push_back(yuvimg);
                        }
                    }
                    frm_eos = mpp_frame_get_eos(loop_data.frame);

                    ret = mpp_frame_deinit(&loop_data.frame);
                    loop_data.frame = NULL;
                    get_frm = 1;
                }

                // try get runtime frame memory usage
                if (loop_data.frm_grp) {
                    size_t usage = mpp_buffer_group_usage(loop_data.frm_grp);
                    if (usage > loop_data.max_usage)
                        loop_data.max_usage = usage;
                }

                // if last packet is send but last frame is not found continue
                if (pkt_eos && pkt_done && !frm_eos) {
                    usleep(1*1000);
                    continue;
                }

                if (frm_eos) {
                    cout << "found last frame" << endl;
                    break;
                }

                if (loop_data.frame_num > 0 && loop_data.frame_count >= loop_data.frame_num) {
                    loop_data.eos = 1;
                    break;
                }

                if (get_frm)
                    continue;
                break;
            } while (1);

            if (loop_data.frame_num > 0 && loop_data.frame_count >= loop_data.frame_num) {
                loop_data.eos = 1;
                cout << "reach max frame number " << loop_data.frame_count << endl;
                break;
            }

            if (pkt_done)
                break;

            /*
            为什么睡在这里?
            * mpi->decode_put_packet将失败，当内部队列中的数据包是
            *满，等待包被消耗。通常是硬件解码一个
            分辨率为1080p的帧需要2毫秒，所以这里我们睡了3毫秒
            * *就够了。
            */
            usleep(3*1000);
        } while (1);
        // mpp_packet_deinit(&loop_data.packet);

        return ret;
    }

    std::shared_ptr<mpp_decoder> create_mpp_decoder(int width, int height, int fps, int decod_type)
    {
        std::shared_ptr<mpp_decoder> instance(new mpp_decoder());
        if(!instance->open(width, height, fps, decod_type))
            instance.reset();
        return instance;
    }
}
