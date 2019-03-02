// implement the scale layer on teec

#ifndef SCALE_TEEC_H
#define SCALE_TEEC_H

#include "scale.h"

namespace ncnn {

class Scale_teec :  public Scale
{
public:
	virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // SCALE_TEEC_H
