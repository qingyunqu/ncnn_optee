#include "layer_registered.h"
#include "flatten_teec_ta_defines.h"

#include <stdio.h>

TEE_Result flatten_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("flatten_ta\n");
	
	/**
	  * params[0]: void* bottom_blob.data
	  * params[1]: void* top_blob.data
	  * params[2]: Flatten_params* fp;
	  * params[3]: NONE
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_NONE);
	if(param_types != exp_param_types){
		printf("error params!\n");
		return TEE_ERROR_BAD_PARAMETERS;
	}
	Flatten_params* fp = (Flatten_params*)params[2].memref.buffer;
	Mat_C* bb = &fp->bottom_blob;
	Mat_C* tb = &fp->top_blob;
	float* bottom_blob = (float*)params[0].memref.buffer;
	float* top_blob = (float*)params[1].memref.buffer;
	
	int w = bb->w;
	int h = bb->h;
	int channels = bb->c;
	size_t elemsize = bb->elemsize;
	int size = w * h;

	//#pragma omp parallel for num_threads(xxx)
	for(int q=0; q<channels; q++){
		const float* ptr = channel(bottom_blob,bb,q);
		float* outptr = (float*)top_blob + size * q;
		for(int i=0; i<size; i++){
			outptr[i] = ptr[i];
		}
	}
	dprintf("flatten_ta success\n");
	return TEE_SUCCESS;
}
