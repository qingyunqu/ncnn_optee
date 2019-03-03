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
			for(int i=0; i<size; i++){
				ptr[i] = b * ptr[i] + a;
			}
		}
	}

	dprintf("batchnorm_ta success\n");	
	return TEE_SUCCESS;
}
