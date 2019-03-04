// implement the Dropout layer on teec

#ifndef DROPOUT_TEEC_H
#define DROPOUT_TEEC_H

#include "dropout.h"

namespace ncnn {

class Dropout_teec : public Dropout
{
public:
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // DROPOUT_TEEC_H
