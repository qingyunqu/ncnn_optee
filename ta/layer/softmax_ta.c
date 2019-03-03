#include "layer_registered.h"
#include "softmax_teec_ta_defines.h"
#include <float.h>
#include "math.h" // my implementation

#include <stdio.h>

TEE_Result softmax_ta(uint32_t param_types, TEE_Param params[4])
{
	// value = exp( value - global max value )
    // sum all value
    // value = value / sum
	dprintf("softmax_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: Softmax_params* sp;
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
	Softmax_params* sp = (Softmax_params*)params[1].memref.buffer;
	Mat_C* btb = &sp->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;

	int dims = btb->dims;
	size_t elemsize = btb->elemsize;
	int axis = sp->axis;
	if(dims == 1) // axis == 0
	{
		int w = btb->w;
		float* ptr = bottom_top_blob;
		float max_ = -FLT_MAX;
		for(int i=0; i<w; i++){
			max_ = max(max_, ptr[i]);
		}
		for(int i=0; i<w; i++){
			ptr[i] = exp(ptr[i] - max_);
		}
		float sum = 0.f;
		for(int i=0; i<w; i++){
			sum += ptr[i];
		}
		for(int i=0; i<w; i++){
			ptr[i] /= sum;
		}
		dprintf("softmax_ta success\n");
		return TEE_SUCCESS;
	}
	if(dims == 2 && axis == 0)
	{
		int w = btb->w;
		int h = btb->h;

		float* max_ = (float*) TEE_Malloc(w * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!max_){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<w; i++)
			max_[i] = -FLT_MAX;

		for(int i=0; i<h; i++){
			const float* ptr = row(bottom_top_blob,btb,i);
			for(int j=0; j<w; j++){
				max_[j] = max(max_[j], ptr[j]);
			}
		}
		for(int i=0; i<h; i++){
			float* ptr = row(bottom_top_blob,btb,i);
			for(int j=0; j<w; j++){
				ptr[j] += exp(ptr[j] - max_[j]);
			}
		}
		TEE_Free(max_);
		
		float* sum = (float*) TEE_Malloc(w * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!sum){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<w; i++)  // this can be commented
			sum[i] = 0.f;
		
		for(int i=0; i<h; i++){
			const float* ptr = row(bottom_top_blob,btb,i);
			for(int j=0; j<w; j++){
				sum[j] += ptr[j];
			}
		}
		for(int i=0; i<h; i++){
			float* ptr = row(bottom_top_blob,btb,i);
			for( int j=0; j<w; j++){
				ptr[j] /= sum[j];
			}
		}
		TEE_Free(sum);
		
		dprintf("softmax_ta success\n");
		return TEE_SUCCESS;
	}
	if(dims == 2 && axis == 1)
	{
		int w = btb->w;
		int h = btb->h;
		
		float* max_ = (float*) TEE_Malloc(h * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!max_){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		
		for(int i=0; i<h; i++){
			const float* ptr = row(bottom_top_blob,btb,i);
			float m = -FLT_MAX;
			for(int j=0; j<w; j++){
				m = max(m, ptr[j]);
			}
			max_[i] = m;
		}
		for(int i=0; i<h; i++){
			float* ptr = row(bottom_top_blob,btb,i);
			float m = max_[i];
			for(int j=0; j<w; j++){
				ptr[j] = exp(ptr[j] - m);
			}
		}
		TEE_Free(max_);

		float* sum = (float*) TEE_Malloc(h * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!sum){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		
		for(int i=0; i<h; i++){
			const float* ptr = row(bottom_top_blob,btb,i);
			float s = 0.f;
			for(int j=0; j<w; j++){
				s += ptr[j];
			}
			sum[i] += s;
		}
		for(int i=0; i<h; i++){
			float* ptr = row(bottom_top_blob,btb,i);
			float s = sum[i];
			for(int j=0; j<w; j++){
				ptr[j] /= s;
			}
		}
		TEE_Free(sum);

		dprintf("softmax_ta success\n");
		return TEE_SUCCESS;
	}
	if(dims == 3 && axis == 0)
	{
		int w = btb->w;
		int h = btb->h;
		int channels = btb->c;
		int size = w * h;
		
		float* max_ = (float*) TEE_Malloc(w * h * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!max_){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<size; i++)
			max_[i] = -FLT_MAX;
		
		for(int q=0; q<channels; q++){
			const float* ptr = channel(bottom_top_blob,btb,q);
			for(int i=0; i<size; i++){
				max_[i] = max(max_[i],ptr[i]);
			}
		}
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			for(int i=0; i<size; i++){
				ptr[i] = exp(ptr[i] - max_[i]);
			}
		}
		TEE_Free(max_);
		
		float* sum = (float*) TEE_Malloc(w * h * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!sum){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<size; i++)
			sum[i] = 0.f;
		
		for(int q=0; q<channels; q++){
			const float* ptr = channel(bottom_top_blob,btb,q);
			for(int i=0; i<size; i++){
				sum[i] += ptr[i];
			}
		}
		//#pragma omp parallel num_threads(xxx)
		for(int q=0; q<channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			for(int i=0; i<size; i++){
				ptr[i] /= sum[i];
			}
		}

		dprintf("softmax_ta success\n");
		return TEE_SUCCESS;
	}
	if(dims == 3 && axis == 1)
	{
		int w = btb->w;
		int h = btb->h;
		int channels = btb->c;
		
		float* max_ = (float*) TEE_Malloc(h * channels * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!max_){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<h*channels; i++)
			max_[i] = -FLT_MAX;
		
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			const float* ptr = channel(bottom_top_blob,btb,q);
			float* max_ptr = (float*)max_ + h * q; // only support float32
			for(int i=0; i<h; i++){
				float max_tmp = -FLT_MAX;
				for(int j=0; j<w; j++){
					max_tmp = max(max_tmp, ptr[j]);
				}
				max_ptr[i] = max_tmp;
				ptr += w;
			}
		}
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			float* max_ptr = (float*)max_ + h * q; // only support float32
			for(int i=0; i<h; i++){
				float max_tmp = max_ptr[i];
				for(int j=0; j<w; j++){
					ptr[j] = exp(ptr[j] - max_tmp);
				}
				ptr += w;
			}
		}
		TEE_Free(max_);
			
		float* sum = (float*) TEE_Malloc(h * channels * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!sum){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<h*channels; i++)
			sum[i] = 0.f;
		
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			const float* ptr = channel(bottom_top_blob,btb,q);
			float* sumptr = (float*)sum + h * q; // only support float32
			for(int i=0; i<h; i++){
				float sum_tmp = 0.f;
				for(int j=0; j<w; j++){
					sum_tmp += ptr[j];
				}
				sumptr[i] = sum_tmp;
				ptr += w;
			}
		}
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			float* sumptr = (float*)sum + h * q; // only support float32
			for(int i=0; i<h; i++){
				float sum_tmp = sumptr[i];
				for(int j=0; j<w; j++){
					ptr[j] /= sum_tmp;
				}
				ptr += w;
			}
		}
		TEE_Free(sum);

		dprintf("softmax_ta success\n");
		return TEE_SUCCESS;
	}
	if(dims == 3 && axis == 2)
	{
		int w = btb->w;
		int h = btb->h;
		int channels = btb->c;
		
		float* max_ = (float*) TEE_Malloc(w * channels * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!max_){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<w*channels; i++)
			max_[i] = -FLT_MAX;
		
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			const float* ptr = channel(bottom_top_blob,btb,q);
			float* max_ptr = (float*)max_ + w * q;// only support float32
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					max_ptr[j] = max(max_ptr[j],ptr[j]);
				}
				ptr += w;
			}
		}
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			float* max_ptr = (float*)max_ + w * q;// only support float32
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					ptr[j] = exp(ptr[j] - max_ptr[j]);
				}
				ptr += w;
			}
		}
		TEE_Free(max_);
		
		float* sum = (float*) TEE_Malloc(w * channels * sizeof(float)/*elemsize*/, 0);// only support float32, elemsize = 4
		if(!sum){
			printf("error TEE_Malloc!\n");
			return TEE_ERROR_OUT_OF_MEMORY;
		}
		for(int i=0; i<w*channels; i++)
			sum[i] = 0.f;
		
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			const float* ptr = channel(bottom_top_blob,btb,q);
			float* sumptr = (float*)sum + w * q;// only support float32
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					sumptr[j] += ptr[j];
				}
				ptr += w;
			}
		}
		//#pragma omp parallel for num_threads(xxx)
		for(int q=0; q<channels; q++){
			float* ptr = channel(bottom_top_blob,btb,q);
			float* sumptr = (float*)sum + w * q;// only support float32
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					ptr[j] /= sumptr[j];
				}
				ptr += w;
			}
		}
		TEE_Free(sum);
		
		dprintf("softmax_ta success\n");
		return TEE_SUCCESS;
	}

	dprintf("softmax_ta failed\n");
	return TEE_ERROR_BAD_PARAMETERS;	
}
