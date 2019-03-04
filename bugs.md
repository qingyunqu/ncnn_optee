## record bugs
* need to ternimate_tee_session, see `src/layer/teec/teec.h` and `src/layer/teec/batchnorm_teec.cpp`
* only support `float32` mat operation in optee os: `teec_ta_defines.h` `teec_ta_defines.c` `softmax_ta.c` `math.c`
* run `mobilenet` `batchnorm_ta`, there exists Error `D/TC:? 0 tee_ta_invoke_command:625 Error: ffff000c of 3`
* how to check the running result is the as same as the original framework
* float precise in `math.c`, can't use `<math.h>`
* only realize part of `pooling_ta.c` and `lrn_ta.c`
