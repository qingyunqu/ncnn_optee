//teec
#ifndef TEEC_H
#define TEEC_H

#ifdef __cplusplus
extern "C"{
#endif

#include <err.h>
#include <stdio.h>
#include <string.h>

#include <tee_client_api.h>
#include <ncnn_ta.h>

typedef struct {
	TEEC_Context ctx;
	TEEC_Session sess;
}ncnn_ctx;

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
//terminate_tee_session(&ctx); // this need to be done

#ifdef __cplusplus
}// extern "C"
#endif

#endif // TEEC_H
