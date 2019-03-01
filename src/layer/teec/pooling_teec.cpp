#include <stdio.h>

#include "pooling_teec.h"
#include "teec.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling_teec)

int Pooling_teec::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
	printf("Pooling_teec::forward\n");
	return Pooling::forward(bottom_blob,top_blob,opt);
	return 0;
}

} // namespace ncnn
