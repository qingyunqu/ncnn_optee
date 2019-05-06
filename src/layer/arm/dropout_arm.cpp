#include "dropout_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Dropout_arm)

int Dropout_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    return Dropout::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn