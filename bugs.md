## revord bugs
* need to ternimate_tee_session, see `src/layer/teec/teec.h` and `src/layer/teec/batchnorm_teec.cpp`
* only support `float32` in optee os
* run `batchnorm_ta`, there exists Error `D/TC:? 0 tee_ta_invoke_command:625 Error: ffff000c of 3`