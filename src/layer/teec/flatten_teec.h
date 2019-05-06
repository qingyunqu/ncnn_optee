// implement the flatten layer on teec

#ifndef FLATTEN_TEEC_H
#define FLATTEN_TEEC_H

#include "flatten_arm.h"

namespace ncnn {

class Flatten_teec : public Flatten_arm
{
public:
	virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // FLATTEN_TEEC_H 
