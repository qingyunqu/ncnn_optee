// implement the batchnorm layer by teec and run forward in the optee os

#include "batchnorm_teec.h"
#include <stdio.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm_teec)

int BatchNorm_teec::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	printf("BatchNorm_teec::forward_inplace\n");
	return BatchNorm::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
