global-incdirs-y += include
srcs-y += ncnn_ta.c
srcs-y += layer/teec_ta_defines.c
srcs-y += layer/math.c
srcs-y += layer/batchnorm_ta.c
srcs-y += layer/pooling_ta.c
srcs-y += layer/relu_ta.c
srcs-y += layer/scale_ta.c
srcs-y += layer/softmax_ta.c

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
