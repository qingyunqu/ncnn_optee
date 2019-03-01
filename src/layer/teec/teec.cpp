//implemention of teec.h

#include "teec.h"

namespace ncnn {
void init_mat_c_from_mat(Mat_C* mat_c, const Mat& mat){
	mat_c->elemsize = mat.elemsize;
	mat_c->dims = mat.dims;
	mat_c->w = mat.w;
	mat_c->h = mat.h;
	mat_c->c = mat.c;
	mat_c->cstep = mat.cstep;
}
} // namespace ncnn

#ifdef __cplusplus
extern "C"{
#endif 

void prepare_tee_session(ncnn_ctx* ctx)
{
	TEEC_UUID uuid = TA_NCNN_UUID;
	uint32_t origin;
	TEEC_Result res;
	
	res = TEEC_InitializeContext(NULL, &ctx->ctx);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);
	
	res = TEEC_OpenSession(&ctx->ctx, &ctx->sess, &uuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_OpenSession failed with code 0x%x origin 0x%x", res, origin);
}
void terminate_tee_session(ncnn_ctx *ctx)
{
	TEEC_CloseSession(&ctx->sess);
	TEEC_FinalizeContext(&ctx->ctx);
}


ncnn_ctx ctx;
int ctx_flag = 0;

#ifdef __cplusplus
}
#endif
