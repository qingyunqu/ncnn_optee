#include <stdio.h>

#include "scale_teec.h"
#include "teec.h"
#include "scale_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Scale_teec)

int Scale_teec::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
	Mat& bottom_top_blob = bottom_top_blobs[0];
	const Mat& scale_blob = bottom_top_blobs[1];
	
	dprintf("Scale_teec::forward\n");
	if (ctx_flag != 1){
		prepare_tee_session(&ctx);
		ctx_flag = 1;
	}
	TEEC_Result res;
	uint32_t origin;
	TEEC_Operation op;
	memset(&op,0,sizeof(op));
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: void* scale_blob.data
	  * params[2]: void* bias_data.data
	  * params[3]: Scale_params* sp;
	  */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_MEMREF_TEMP_INPUT);
	op.params[0].tmpref.buffer = bottom_top_blob.data;
	op.params[0].tmpref.size = bottom_top_blob.total() * bottom_top_blob.elemsize;
	op.params[1].tmpref.buffer = scale_blob.data;
	op.params[1].tmpref.size = scale_blob.total() * scale_blob.elemsize;
	op.params[2].tmpref.buffer = bias_data.data;
	op.params[2].tmpref.size = bias_data.total() * bias_data.elemsize;
	Scale_params sp;
	init_mat_c_from_mat(&sp.bottom_top_blob,bottom_top_blob);
	init_mat_c_from_mat(&sp.scale_blob,scale_blob);
	init_mat_c_from_mat(&sp.bias_data,bias_data);
	sp.scale_data_size = scale_data_size;
	sp.bias_term = bias_term;
	op.params[3].tmpref.buffer = (void*)&sp;
	op.params[3].tmpref.size = sizeof(sp);
	
	res = TEEC_InvokeCommand(&(ctx.sess),TA_SCALE,&op,&origin);
	if(res != TEEC_SUCCESS){
		dprintf("Scale_teec::forward failed\n");
		return Scale::forward_inplace(bottom_top_blobs, opt);
	}
	
	dprintf("Scale_teec::forward success\n");
	return 0;
}

int Scale_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	std::vector<Mat> bottom_top_blobs(2);
	bottom_top_blobs[0] = bottom_top_blob;
	bottom_top_blobs[1] = scale_data;
	
	return forward_inplace(bottom_top_blobs,opt);
}

} // namespace ncnn
