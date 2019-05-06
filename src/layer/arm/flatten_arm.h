#ifndef LAYER_FLATTEN_ARM_H
#define LAYER_FLATTEN_ARM_H

#include "flatten.h"

namespace ncnn {

class Flatten_arm : public Flatten
{
public:
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLATTEN_ARM_H