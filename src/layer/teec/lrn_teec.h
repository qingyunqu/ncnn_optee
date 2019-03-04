// implement the LRN layer on teec

#ifndef LRN_TEEC_H
#define LRN_TEEC_H

#include "lrn.h"

namespace ncnn {

class LRN_teec : public LRN
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
}

} // namespace ncnn

#endif // LRN_TEEC_H
