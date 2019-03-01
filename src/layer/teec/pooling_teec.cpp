#include <stdio.h>

#include "pooling_teec.h"
#include "teec.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling_teec)

int Pooling_teec::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
	// max value in NxN window
    // avg value in NxN window
	printf("Pooling_teec::forward\n");
	
	if(ctx_flag != 1){
		prepare_tee_session(&ctx);
		ctx_flag = 1;
	}
	TEEC_Result res;
	uint32_t origin;
	TEEC_Operation op;
	memset(&op,0,sizeof(op));
	
	//int w = bottom_blob.w;
	//int h = bottom_blob.h;
	int channels = bottom_blob.c;
	size_t elemsize = bottom_blob.elemsize;
	
	//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", bottom_blob.w, bottom_blob.h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);
	if (global_pooling)
	{
		//top_blob.create(channels, elemsize, opt.blob_allocator);
		//if (top_blob.empty())
		//	return -100;
		/**
		  * params[0]: void* bottom_blob.data
		  * params[1]: void* top_blob.data
		  * params[2]: Pooling_params* pp;
		  * params[3]: NONE
		  */
	}
		
	
	return Pooling::forward(bottom_blob,top_blob,opt);
	return 0;
}

} // namespace ncnn
