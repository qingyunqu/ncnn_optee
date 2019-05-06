// implement the pooling layer on teec

#ifndef POOLING_TEEC_H
#define POOLING_TEEC_H

#include "pooling_arm.h"

namespace ncnn {

class Pooling_teec : public Pooling_arm
{
public:
	virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // POOLING_TEEC_H
