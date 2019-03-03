## record bugs
* need to ternimate_tee_session, see `src/layer/teec/teec.h` and `src/layer/teec/batchnorm_teec.cpp`
* only support `float32` mat operation in optee os
* run `mobilenet` `batchnorm_ta`, there exists Error `D/TC:? 0 tee_ta_invoke_command:625 Error: ffff000c of 3`
* float precise in `pooling_ta`, can't use `math.h`
* only realize part of `pooling_ta`, but it is useful
