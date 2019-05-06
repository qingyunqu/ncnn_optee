// implement the softmax layer on teec

#ifndef SOFTMAX_TEEC_H
#define SOFTMAX_TEEC_H

#include "softmax_arm.h"

namespace ncnn {

class Softmax_teec : public Softmax_arm
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // SOFTMAX_TEEC_H
