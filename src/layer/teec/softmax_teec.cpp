#include <stdio.h>

#include "softmax_teec.h"
#include "teec.h"
#include "softmax_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Softmax_teec)

int Softmax_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	// value = exp( value - global max value )
    // sum all value
    // value = value / sum
	dprintf("Softmax_teec::forward\n");

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
	  * params[1]: Softmax_params* sp;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_NONE,
									TEEC_NONE);
	op.params[0].tmpref.buffer = bottom_top_blob.data;
	op.params[0].tmpref.size = bottom_top_blob.total() * bottom_top_blob.elemsize;
	Softmax_params sp;
	init_mat_c_from_mat(&sp.bottom_top_blob,bottom_top_blob);
	sp.axis = axis;
	op.params[1].tmpref.buffer = (void*)&sp;
	op.params[1].tmpref.size = sizeof(sp);
	
	res = TEEC_InvokeCommand(&(ctx.sess),TA_SOFTMAX,&op,&origin);
	if (res != TEEC_SUCCESS){
		dprintf("Softmax_teec::forward failed\n");
		return Softmax::forward_inplace(bottom_top_blob,opt);
	}
	dprintf("Softmax_teec::forward success\n");
	return 0;
}

} // namespace  ncnn
