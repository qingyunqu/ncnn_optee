// implement the relu layer on teec

#ifndef RELU_TEEC_H
#define RELU_TEEC_H

#include "relu.h"

namespace ncnn {

class ReLU_teec : public ReLU
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // RELU_TEEC_H
