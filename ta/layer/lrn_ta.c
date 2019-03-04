#include "layer_registered.h"
#include "lrn_teec_ta_defines.h"
#include "math.h" // my implementation

#include <stdio.h>

TEE_Result lrn_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("lrn_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: LRN_params* lrnp;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_NONE,
													TEE_PARAM_TYPE_NONE);
	if(param_types != exp_param_types){
		printf("error params!\n");
		return TEE_ERROR_BAD_PARAMETERS;
	}
	LRN_params* lrnp = (LRN_params*)params[1].memref.buffer;
	Mat_C* btb = &lrnp->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;

	int w = btb->w;
	int h = btb->h;
	int channels = btb->c;
	size_t elemsize = btb->elemsize;
	int size = w * h;

	// squared values with local_size padding
	dprintf("malloc size: %d\n",w * h * channels * sizeof(float));
	float* square_blob = (float*) TEE_Malloc(w * h * channels * sizeof(float)/*elemsize*/, 0);// only support float32
	if(!square_blob){
		dprintf("error TEE_Malloc: square_blob!\n");
		return TEE_ERROR_OUT_OF_MEMORY;
	}
	
	//#pragma omp parallel
	for(int q=0; q<channels; q++){
		const float* ptr = channel(bottom_top_blob,btb,q);
		float* outptr = square_blob + w * h * q;
		for(int i=0; i<size; i++){
			outptr[i] = ptr[i] * ptr[i];
		}
	}
	
	if(lrnp->region_type == NormRegion_ACROSS_CHANNELS)
	{
		float* square_sum = (float*) TEE_Malloc(w * h * channels * sizeof(float)/*elemsize*/, 0);// only support float32
		if(!square_sum){
			dprintf("error TEE_Malloc: square_sum!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<w*h*channels; i++) // may not need
			square_sum[i] = 0.f;
		
		const float alpha_div_size = lrnp->alpha / lrnp->local_size;
		//#pragma omp parallel
		for(int q=0; q<channels; q++){
			// square sum
			float* ssptr = square_sum + w * h * q;
			for(int p = q - lrnp->local_size/2; p <= q + lrnp->local_size/2; p++){
				if(p<0 || p>=channels)
					continue;
				const float* sptr = square_blob + w * h * p;
				for(int i=0; i<size; i++){
					ssptr[i] += sptr[i];
				}
			}
			float* ptr = channel(bottom_top_blob,btb,q);
			for(int i=0; i<size; i++){
				ptr[i] = ptr[i] * pow(lrnp->bias + alpha_div_size * ssptr[i], -lrnp->beta);
			}
		}
		TEE_Free(square_sum);
	}
	
	TEE_Free(square_blob);
	dprintf("softmax_ta success\n");
	return TEE_SUCCESS;
	
	//dprintf("softmax_ta failed\n");
	//return TEE_ERROR_BAD_PARAMETERS;
}
