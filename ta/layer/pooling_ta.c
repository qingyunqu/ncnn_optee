#include "layer_registered.h"
#include "pooling_teec_ta_defines.h"

#include <stdio.h>
#include "math.h" // my implementation

TEE_Result pooling_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("pooling_ta\n");
	/**
	  * params[0]: void* bottom_blob.data
	  * params[1]: void* top_blob.data
	  * params[2]: Pooling_params* pp;
	  * params[3]: NONE
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_NONE);
	if (param_types != exp_param_types){
		printf("error params!\n");	
		return TEE_ERROR_BAD_PARAMETERS;
	}
	Pooling_params* pp = (Pooling_params*)params[2].memref.buffer;
	if (pp->global_pooling)
	{
		//Pooling_params* pp = (Pooling_params*)params[2].memref.buffer;
		Mat_C* bb = &pp->bottom_blob;
		//Mat_C* tb = &pp->top_blob;
		float* bottom_blob = (float*)params[0].memref.buffer;
		float* top_blob = (float*)params[1].memref.buffer;
	
		int w = bb->w;
		int h = bb->h;
		int channels = bb->c;
		//size_t elemsize = bb->elemsize;
		
		int size = w * h;
		if (pp->pooling_type == PoolMethod_MAX)
		{
			//#pragma omp parallel for num_threads(xxx)
			for (int q=0; q<channels; q++)
			{
				const float* ptr = channel(bottom_blob, bb, q);
				float max_ = ptr[0];
				for (int i=0; i<size; i++)
				{
					max_ = max(max_, ptr[i]);
				}
				top_blob[q] = max_;
			}
		}
		else if (pp->pooling_type == PoolMethod_AVE)
		{
			//#pragma omp parallel for num.threads(xxx)
			for (int q=0; q<channels; q++)
			{
				const float* ptr = channel(bottom_blob,bb,q);
				float sum = 0.f;
				for (int i=0; i<size; i++)
				{
					sum += ptr[i];
				}
				top_blob[q] = sum / size;
			}
		}
		dprintf("pooling_ta success\n");
		return TEE_SUCCESS;
	}

	dprintf("pooling_ta failed\n");
	return TEE_ERROR_BAD_PARAMETERS;
}
