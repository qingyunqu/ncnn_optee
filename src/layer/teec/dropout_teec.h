// implement the Dropout layer on teec

#ifndef DROPOUT_TEEC_H
#define DROPOUT_TEEC_H

#include "dropout_arm.h"

namespace ncnn {

class Dropout_teec : public Dropout_arm
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // DROPOUT_TEEC_H
