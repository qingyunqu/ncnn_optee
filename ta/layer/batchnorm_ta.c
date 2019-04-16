#include "layer_registered.h"
#include "batchnorm_teec_ta_defines.h"

//debug
#include <stdio.h>

TEE_Result batchnorm_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("batchnorm_ta\n");
	/**
	 * params[0]: void* bottom_top_blob.data
	 * params[1]: void* a_data.data
	 * params[2]: void* b_data.data
	 * params[3]: Batchnorm_param* bnp;
	 */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INPUT);
	if (param_types != exp_param_types){
		printf("error params!\n");
		return TEE_ERROR_BAD_PARAMETERS;
	}
	
	Batchnorm_params* bnp = (Batchnorm_params*)params[3].memref.buffer;
	Mat_C* btb = &bnp->bottom_top_blob;
	float* a_data = (float*)params[1].memref.buffer;
	float* b_data = (float*)params[2].memref.buffer;
	float* bottom_top_blob = (float*)params[0].memref.buffer;

	int dims = btb->dims;
	//printf("dims: %d\n",dims);
	if(dims == 1){
		int w = btb->w;
		float* ptr = bottom_top_blob;
		//#pragma omp parallel for num_threads(xxx)
		for(int i=0; i<w; i++){
			ptr[i] = b_data[i] * ptr[i] + a_data[i];
		}
	}
	if(dims == 2){
		int w = btb->w;
		int h = btb->h;
		//#pragma omp parallel
		for(int i=0; i<h; i++){
			float* ptr = row(bottom_top_blob,btb,i);
			float a = a_data[i];
			float b = b_data[i];
			for(int j=0; j<w; j++){
				ptr[j] = b * ptr[j] + a;
			}
		}
	}
	if(dims == 3){
		int w = btb->w;
		int h = btb->h;
		int size = w * h;
		//#pragma omp parallel
		for(int q=0; q<bnp->channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			float a = a_data[q];
			float b = b_data[q];

			int nn = size >> 2;
			int remain = size - (nn << 2);
#if __ARM_NEON
#if __aarch64__
			dprintf("neon64\n");
			if (nn > 0)
			{
			asm volatile(
				"dup        v1.4s, %w4             \n"
            	"dup        v2.4s, %w5             \n"
            	"0:                                \n"
            	"prfm       pldl1keep, [%1, #128]  \n"
            	"ld1        {v0.4s}, [%1]          \n"
            	"orr        v3.16b, v1.16b, v1.16b \n"
            	"fmla       v3.4s, v0.4s, v2.4s    \n"
            	"subs       %w0, %w0, #1           \n"
            	"st1        {v3.4s}, [%1], #16     \n"
            	"bne        0b                     \n"
            	: "=r"(nn),     // %0
            	  "=r"(ptr)     // %1
            	: "0"(nn),
            	  "1"(ptr),
            	  "r"(a),       // %4
            	  "r"(b)        // %5
            	: "cc", "memory", "v0", "v1", "v2", "v3"
			);
			}
#else
			if (nn > 0)
        	{
        	asm volatile(
            	"vdup.f32   q1, %4              \n"
            	"vdup.f32   q2, %5              \n"
            	"0:                             \n"
            	"pld        [%1, #128]          \n"
            	"vld1.f32   {d0-d1}, [%1 :128]  \n"
            	"vorr.32    q3, q1, q1          \n"
            	"vmla.f32   q3, q0, q2          \n"
            	"subs       %0, #1              \n"
            	"vst1.f32   {d6-d7}, [%1 :128]! \n"
            	"bne        0b                  \n"
            	: "=r"(nn),     // %0
            	  "=r"(ptr)     // %1
            	: "0"(nn),
            	  "1"(ptr),
            	  "r"(a),       // %4
            	  "r"(b)        // %5
            	: "cc", "memory", "q0", "q1", "q2", "q3"
        	);
			}
#endif // __aarch64__
			for (; remain>0; remain--)
			{
				*ptr = b * *ptr + a;

				ptr++;
			}
#else
			for(int i=0; i<size; i++){
				ptr[i] = b * ptr[i] + a;
			}
#endif // __ARM_NEON
		}
	}

	dprintf("batchnorm_ta success\n");	
	return TEE_SUCCESS;
}
