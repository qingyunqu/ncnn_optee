#include "flatten_teec.h"
#include "teec.h"
#include "flatten_teec_ta_defines.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Flatten_teec)

int Flatten_teec::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
	dprintf("Flatten_teec::forward\n");
	
	if(ctx_flag != 1){
		prepare_tee_session(&ctx);
		ctx_flag = 1;
	}
	TEEC_Result res;
	uint32_t origin;
	TEEC_Operation op;
	memset(&op,0,sizeof(op));
	
	int w = bottom_blob.w;
	int h = bottom_blob.h;
	int channels = bottom_blob.c;
	size_t elemsize = bottom_blob.elemsize;
	int size = w * h;

	top_blob.create(size * channels, elemsize, opt.blob_allocator);
	if(top_blob.empty())
		return -100;
	
	/**
	  * params[0]: void* bottom_blob.data
	  * params[1]: void* top_blob.data
	  * params[2]: Flatten_params* fp;
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
	Flatten_params fp;
	init_mat_c_from_mat(&fp.bottom_blob,bottom_blob);
	init_mat_c_from_mat(&fp.top_blob,top_blob);
	
	op.params[2].tmpref.buffer = (void*)&fp;
	op.params[2].tmpref.size = sizeof(fp);
	
	res = TEEC_InvokeCommand(&(ctx.sess),TA_FLATTEN,&op,&origin);
	if(res != TEEC_SUCCESS){
		dprintf("Flatten_teec::forward failed\n");
		return Flatten_arm::forward(bottom_blob,top_blob,opt);//top_blob.create()
	}
	dprintf("Flatten_teec::forward success\n");
	return 0;
}

} // namespace ncnn
