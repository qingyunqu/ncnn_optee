//something useful for implement teec layer

#ifndef TEEC_H
#define TEEC_H

#include "mat.h"
#include "teec_ta_defines.h"
namespace ncnn {
void init_mat_c_from_mat(Mat_C* mat_c, Mat& mat);
} // namespace ncnn


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

void prepare_tee_session(ncnn_ctx* ctx);
void terminate_tee_session(ncnn_ctx *ctx);

extern ncnn_ctx ctx;
extern int ctx_flag;
//terminate_tee_session(&ctx); // this need to be done

#ifdef __cplusplus
}// extern "C"
#endif

#endif // TEEC_H
