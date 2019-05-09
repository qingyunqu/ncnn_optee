#include "layer_registered.h"
#include "relu_teec_ta_defines.h"
#include <math.h>

TEE_Result relu_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("relu_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: ReLU_params* rlup;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                                     TEE_PARAM_TYPE_MEMREF_INPUT,
                                                     TEE_PARAM_TYPE_NONE,
                                                     TEE_PARAM_TYPE_NONE);
	if(param_types != exp_param_types){
		return TEE_ERROR_BAD_PARAMETERS;
	}
	ReLU_params* rlup = (ReLU_params*)params[1].memref.buffer;
	Mat_C* btb = &rlup->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;
	
	int w = btb->w;
	int h = btb->h;
	int channels = btb->c;
	int size = w * h;
	float slope = rlup->slope;
	if (slope == 0.f)
	{
		//#pragma omp parallel for num_threads(xxx)
		for (int q=0; q<channels; q++)
		{
            float* ptr = channel(bottom_top_blob,btb,q);
#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t vzero = vdup1_n_f32(0.f);
            for (; nn > 0; nn--){
                float32x4_t vp = vld1q_f32(ptr);
                vp = vmaxq_f32(vp, vzero);
                vst1q_f32(ptr, vp);

                ptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "veor       q1, q0, q0          \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vmax.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--){
                *ptr = max(*ptr, 0.f);
                ptr++;
            }
		}
	}
	else
	{
		//#pragma omp parallel for num_threads(xxx)
		for (int q=0; q<channels; q++)
		{
			float* ptr = channel(bottom_top_blob,btb,q);
#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t vzero = vdupq_n_f32(0.f);
            float32x4_t vslope = vdup_n_f32(slope);
            for (; nn > 0; nn--){
                float32x4_t vp = vld1q_f32(ptr);
                uint32x4_t vlemask = vcleq_f32(vp, vzero);
                float32x4_t vps = vmulq_f32(vp, vslope);
                vp = vbslq_f32(vlemask, vps, vp);
                vst1q_f32(ptr, vp);

                ptr += 4;
            }
#else
            if(nn > 0){
            asm volatile(
                "veor       q1, q0, q0          \n"
                "vdup.f32   q2, %4              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vcle.f32   q3, q0, q1          \n"
                "vmul.f32   q4, q0, q2          \n"
                "vbit.32    q0, q4, q3          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(slope)    // %4
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--){
                if(*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
		}
	}
	
	dprintf("relu_ta success\n");
	return TEE_SUCCESS;
}
