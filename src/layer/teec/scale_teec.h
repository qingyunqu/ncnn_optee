// implement the scale layer on teec

#ifndef SCALE_TEEC_H
#define SCALE_TEEC_H

#include "scale_arm.h"

namespace ncnn {

class Scale_teec :  public Scale_arm
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // SCALE_TEEC_H
