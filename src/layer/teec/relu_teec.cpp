#include <stdio.h>

#include "relu_teec.h"
#include "teec.h"
#include "relu_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(ReLU_teec)

int ReLU_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	dprintf("ReLU_teec::forward\n");
	if(ctx_flag != 1){
		prepare_tee_session(&ctx);
		ctx_flag = 1;
	}
	TEEC_Result res;
	uint32_t origin;
	TEEC_Operation op;
	memset(&op,0,sizeof(op));
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: ReLU_params* rlup;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_NONE,
									TEEC_NONE);
	op.params[0].tmpref.buffer = bottom_top_blob.data;
	op.params[0].tmpref.size = bottom_top_blob.total() * bottom_top_blob.elemsize;
	ReLU_params rlup;
	init_mat_c_from_mat(&rlup.bottom_top_blob,bottom_top_blob);
	rlup.slope = slope;
	op.params[1].tmpref.buffer = (void*)&rlup;
	op.params[1].tmpref.size = sizeof(rlup);
	
	res = TEEC_InvokeCommand(&(ctx.sess),TA_RELU,&op,&origin);
	if(res != TEEC_SUCCESS){
		dprintf("ReLU_teec:forward failed\n");
		return ReLU::forward_inplace(bottom_top_blob, opt);
	}
	
	dprintf("ReLU_teec::forward success\n");
	return 0;
}

} // namespace ncnn
