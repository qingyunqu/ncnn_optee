#include <stdio.h>

#include "lrn_teec.h"
#include "teec.h"
#include "lrn_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(LRN_teec)

int LRN_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	/*dprintf("LRN_teec::forward\n");
	
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
	  * params[1]: LRN_params* lrnp;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	/*op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_NONE,
									TEEC_NONE);
	op.params[0].tmpref.buffer = bottom_top_blob.data;
	op.params[0].tmpref.size = bottom_top_blob.total() * bottom_top_blob.elemsize;
	LRN_params lrnp;
	init_mat_c_from_mat(&lrnp.bottom_top_blob,bottom_top_blob);
	lrnp.region_type = region_type;
	lrnp.local_size = local_size;
	lrnp.alpha = alpha;
	lrnp.beta = beta;
	lrnp.bias = bias;
	op.params[1].tmpref.buffer = (void*)&lrnp;
	op.params[1].tmpref.size = sizeof(lrnp);

	if (region_type == NormRegion_ACROSS_CHANNELS)
	{
		res = TEEC_InvokeCommand(&(ctx.sess),TA_LRN,&op,&origin);
		if(res != TEEC_SUCCESS){
			dprintf("LRN_teec::forward failed\n");
			return LRN::forward_inplace(bottom_top_blob, opt);
		}
		dprintf("LRN_teec::forward success\n");
		return 0;
	}*/
	return LRN_arm::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
