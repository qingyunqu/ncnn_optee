// implement the batchnorm layer by teec and run forward in the optee os

#include "batchnorm_teec.h"
#include <stdio.h>

#include "teec.h"
#include "batchnorm_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm_teec)

int BatchNorm_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	// a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a
	dprintf("BatchNorm_teec::forward\n");

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
	 * params[1]: void* a_data.data
	 * params[2]: void* b_data.data
	 * params[3]: Batchnorm_params* bnp;
	 */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_MEMREF_TEMP_INPUT,
									TEEC_MEMREF_TEMP_INPUT);
	op.params[0].tmpref.buffer = bottom_top_blob.data;
	size_t totalsize = alignSize(bottom_top_blob.total() * bottom_top_blob.elemsize, 4);
	op.params[0].tmpref.size = totalsize;// + (int)sizeof(*(bottom_top_blob.refcount));
	op.params[1].tmpref.buffer = a_data.data;
	op.params[1].tmpref.size = a_data.total() * a_data.elemsize;
	op.params[2].tmpref.buffer = b_data.data;
	op.params[2].tmpref.size = b_data.total() * b_data.elemsize;

	dprintf("data size: %d %d %d\n",op.params[0].tmpref.size,op.params[1].tmpref.size,op.params[2].tmpref.size);

	Batchnorm_params bnp;
	bnp.channels = channels;
	init_mat_c_from_mat(&bnp.bottom_top_blob,bottom_top_blob);
	op.params[3].tmpref.buffer = (void*)&bnp;
	op.params[3].tmpref.size = sizeof(bnp);

	res = TEEC_InvokeCommand(&(ctx.sess),TA_BATCHNORM,&op,&origin);
	if(res != TEEC_SUCCESS){
		dprintf("BatchNorm_teec::forward failed\n");
		return BatchNorm_arm::forward_inplace(bottom_top_blob, opt);
	}

	//terminate_tee_session(&ctx);
	dprintf("BatchNorm_teec::forward success\n");
	return 0;
}

} // namespace ncnn
