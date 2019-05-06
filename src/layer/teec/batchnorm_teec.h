// implement the batchnorm layer by teec and run forward in the optee os

#ifndef BATCHNORM_TEEC_H
#define BATCHNORM_TEEC_H

#include "batchnorm_arm.h"

namespace ncnn {

class BatchNorm_teec : public BatchNorm_arm
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_TEEC_H
