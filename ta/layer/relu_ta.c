#include "layer_registered.h"
#include "relu_teec_ta_defines.h"

#include <stdio.h>

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
		printf("error params!\n");
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
			for (int i=0; i<size; i++)
			{
				if (ptr[i] < 0)
					ptr[i] = 0;
			}
		}
	}
	else
	{
		//#pragma omp parallel for num_threads(xxx)
		for (int q=0; q<channels; q++)
		{
			float* ptr = channel(bottom_top_blob,btb,q);
			for (int i=0; i<size; i++)
			{
				if (ptr[i] < 0)
					ptr[i] *= slope;
			}
		}
	}
	
	dprintf("relu_ta success\n");
	return TEE_SUCCESS;
}
