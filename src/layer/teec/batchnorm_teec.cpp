// implement the batchnorm layer by teec and run forward in the optee os

#include "batchnorm_teec.h"
#include <stdio.h>

#include "teec.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm_teec)

int BatchNorm_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	printf("BatchNorm_teec::forward_inplace\n");
	if(ctx_flag!=1){
		prepare_tee_session(&ctx);
		ctx_flag = 1;
	}
	
	TEEC_Result res;
	uint32_t origin;
	TEEC_Operation op;
	memset(&op,0,sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE,
									TEEC_NONE,
									TEEC_NONE,
									TEEC_NONE);
	res = TEEC_InvokeCommand(&(ctx.sess),TA_BATCHNORM,&op,&origin);
	//terminate_tee_session(&ctx);
	
	return BatchNorm::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
