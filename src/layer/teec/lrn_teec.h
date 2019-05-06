// implement the LRN layer on teec

#ifndef LRN_TEEC_H
#define LRN_TEEC_H

#include "lrn_arm.h"

namespace ncnn {

class LRN_teec : public LRN_arm
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LRN_TEEC_H
