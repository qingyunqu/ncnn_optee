#include "layer_registered.h"
#include "scale_teec_ta_defines.h"

TEE_Result scale_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("scale_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: void* scale_data.data
	  * params[2]: void* bias_data.data
	  * params[3]: Scale_params* sp;
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INPUT);
	if (param_types != exp_param_types){
		return TEE_ERROR_BAD_PARAMETERS;
	}
	Scale_params* sp = (Scale_params*)params[3].memref.buffer;
	Mat_C* btb = &sp->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;
	float* scale_data = (float*)params[1].memref.buffer;
	float* bias_data = (float*)params[2].memref.buffer;
	
	int dims = btb->dims;
	int bias_term = sp->bias_term;
	if (dims == 1)
	{
		int w = btb->w;
		float* ptr = bottom_top_blob;
		if (bias_term)
		{
			//#pragma omp parallel for num_threads(xxx)
			for (int i=0; i<w; i++)
			{
				ptr[i] = ptr[i] * scale_data[i] + bias_data[i];
			}
		}
		else
		{
			//#pragma omp parallel for num_threads(xxx)
			for (int i=0; i<w; i++)
			{
				ptr[i] *= scale_data[i];
			}
		}
	}
	if (dims == 2)
	{
		int w = btb->w;
		int h = btb->h;
		if (bias_term)
		{
			//pragma omp parallel for num_threads(xxx)
			for (int i=0; i<h; i++)
			{
				float* ptr = row(bottom_top_blob,btb,i);
				float s = scale_data[i];
				float bias = bias_data[i];
				for (int j=0; j<w; j++)
				{
					ptr[j] = ptr[j] * s + bias;
				}
			}
		}
		else
		{
			//#pragma omp parallel for num_threads(xxx)
			for (int i=0; i<h; i++)
			{
				float* ptr = row(bottom_top_blob,btb,i);
				float s = scale_data[i];
				for (int j=0; j<w; j++)
				{
					ptr[j] *= s;
				}
			}
		}
	}
	if (dims == 3)
	{
		int w = btb->w;
		int h = btb->h;
		int channels = btb->c;
		int size = w * h;
		if (bias_term)
		{
            const float* scale_ptr = scale_data;
            const float* bias_ptr = bias_data;
            //#pragma omp parallel for num_threads(xxx)
            for (int q=0; q<channels; q++)
            {
                float* ptr = channel(bottom_top_blob,btb,q);

                float s = scale_ptr[q];
                float bias = bias_ptr[q];

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn >> 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t vs = vdupq_n_f32(s);
                float32x4_t vbias = vdupq_n_f32(bias);
                for (; nn>0; nn--){
                    float32x4_t vp = vld1q_f32(ptr);
                    vp = vmlaq_f32(vbias, vp, vs);
                    vst1q_f32(ptr, vp);

                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; remain>0; remain--){
                    *ptr = *ptr * s + bias;
                    ptr++;
                }
            }
		}
		else
		{
            const float* scale_ptr = scale_data;
			//#pragma omp parallel for num_threads(xxx)
			for (int q=0; q<channels; q++)
			{
				float* ptr = channel(bottom_top_blob,btb,q);
                float s = scale_ptr[q];
#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t vs = vdupq_n_f32(s);
                for(; nn>0; nn--){
                    float32x4_t vp = vld1q_f32(ptr);
                    vp = vmulq_f32(vp, vs);
                    vst1q_f32(ptr, vp);

                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; remain>0; remain--){
                    *ptr *= s;
                    ptr++;
                }
			}
		}
	}
	dprintf("scale_ta success\n");
	return TEE_SUCCESS;
}
