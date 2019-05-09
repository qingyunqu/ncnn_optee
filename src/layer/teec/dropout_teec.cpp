#include "dropout_teec.h"
#include "teec.h"
#include "dropout_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Dropout_teec)

int Dropout_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	if(scale == 1.f){
		return 0;
	}

	dprintf("Dropout_teec::forward\n");
	
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
	  * params[1]: Dropout_params* dp;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_NONE,
									TEEC_NONE);
	op.params[0].tmpref.buffer = bottom_top_blob.data;
	op.params[0].tmpref.size = bottom_top_blob.total() * bottom_top_blob.elemsize;
	Dropout_params dp;
	init_mat_c_from_mat(&dp.bottom_top_blob,bottom_top_blob);
	dp.scale = scale;
	op.params[1].tmpref.buffer = (void*)&dp;
	op.params[1].tmpref.size = sizeof(dp);
	
	res = TEEC_InvokeCommand(&(ctx.sess),TA_DROPOUT,&op,&origin);
	if(res != TEEC_SUCCESS){
		dprintf("Dropout_teec::forward failed\n");
		return Dropout_arm::forward_inplace(bottom_top_blob,opt);
	}
	
	dprintf("Dropout_teec::forward success\n");
	return 0;
}

} // namespace ncnn
