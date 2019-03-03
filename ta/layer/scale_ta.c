#include "layer_registered.h"
#include "scale_teec_ta_defines.h"

#include <stdio.h>

TEE_Result scale_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("scale_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: void* scale_blob.data
	  * params[2]: void* bias_data.data
	  * params[3]: Scale_params* sp;
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_MEMREF_INPUT);
	if (param_types != exp_param_types){
		printf("error params!\n");
		return TEE_ERROR_BAD_PARAMETERS;
	}
	Scale_params* sp = (Scale_params*)params[3].memref.buffer;
	Mat_C* btb = &sp->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;
	float* scale_blob = (float*)params[1].memref.buffer;
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
				ptr[i] = ptr[i] * scale_blob[i] + bias_data[i];
			}
		}
		else
		{
			//#pragma omp parallel for num_threads(xxx)
			for (int i=0; i<w; i++)
			{
				ptr[i] *= scale_blob[i];
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
				float s = scale_blob[i];
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
				float s = scale_blob[i];
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
			//#pragma omp parallel for num_threads(xxx)
			for (int q=0; q<channels; q++)
			{
				float* ptr = channel(bottom_top_blob,btb,q);
				float s = scale_blob[q];
				float bias = bias_data[q];
				for (int i=0; i<size; i++)
				{
					ptr[i] = ptr[i] * s + bias;
				}
			}
		}
		else
		{
			//#pragma omp parallel for num_threads(xxx)
			for (int q=0; q<channels; q++)
			{
				float* ptr = channel(bottom_top_blob,btb,q);
				float s = scale_blob[q];
				for(int i=0; i<size; i++)
				{
					ptr[i] *= s;
				}
			}
		}
	}
	dprintf("scale_ta success\n");
	return TEE_SUCCESS;
}
