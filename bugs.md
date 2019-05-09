## record bugs
* need to ternimate_tee_session, see `src/layer/teec/teec.h` and `src/layer/teec/batchnorm_teec.cpp`
* only support `float32` mat operation in optee os: `teec_ta_defines.h` `teec_ta_defines.c` `softmax_ta.c` `math.c`
* how to check the running result is the as same as the original framework
## wait tio realize
* LRN
* Pooling
* Softmax
## fixed
* run `mobilenet` `batchnorm_ta`, there exists Error `D/TC:? 0 tee_ta_invoke_command:625 Error: ffff000c of 3` : out of memory, data size is to big(more than 1MB)

