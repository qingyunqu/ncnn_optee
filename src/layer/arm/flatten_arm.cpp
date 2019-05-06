#include "flatten_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Flatten_arm)

int Flatten_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    return Flatten::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn