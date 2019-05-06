#include <stdio.h>

#include "pooling_teec.h"
#include "teec.h"
#include "pooling_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling_teec)

int Pooling_teec::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
	// max value in NxN window
    // avg value in NxN window
	dprintf("Pooling_teec::forward\n");
	
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
		top_blob.create(channels, elemsize, opt.blob_allocator);
		if (top_blob.empty())
			return -100;
		/**
		  * params[0]: void* bottom_blob.data
		  * params[1]: void* top_blob.data
		  * params[2]: Pooling_params* pp;
		  * params[3]: NONE
		  */
		op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
										TEEC_MEMREF_TEMP_INOUT,
										TEEC_MEMREF_TEMP_INPUT,
										TEEC_NONE);
		op.params[0].tmpref.buffer = bottom_blob.data;
		op.params[0].tmpref.size = bottom_blob.total() * bottom_blob.elemsize;
		op.params[1].tmpref.buffer = top_blob.data;
		op.params[1].tmpref.size = top_blob.total() * top_blob.elemsize;
		
		Pooling_params pp;
		init_mat_c_from_mat(&pp.bottom_blob,bottom_blob);
		init_mat_c_from_mat(&pp.top_blob,top_blob);
		pp.pooling_type = pooling_type;
		pp.kernel_w = kernel_w;
		pp.kernel_h = kernel_h;
		pp.stride_w = stride_w;
		pp.stride_h = stride_h;
		pp.pad_left = pad_left;
		pp.pad_right = pad_right;
		pp.pad_top = pad_top;
		pp.pad_bottom = pad_bottom;
		pp.global_pooling = global_pooling;
		pp.pad_mode = pad_mode;
		op.params[2].tmpref.buffer = (void*)&pp;
		op.params[2].tmpref.size = sizeof(pp);
		
		res = TEEC_InvokeCommand(&(ctx.sess),TA_POOLING,&op,&origin);
		if(res != TEEC_SUCCESS){
			dprintf("Pooling_teec::forward failed\n");
			return Pooling_arm::forward(bottom_blob,top_blob,opt);
		}
		dprintf("Pooling_teec::forward success\n");
		return 0;
	}
	
	return Pooling_arm::forward(bottom_blob,top_blob,opt);
	//return 0;
}

} // namespace ncnn
