#include "layer.h"

//debug
#include <stdio.h>

TEE_Result batchnorm_ta(uint32_t param_types, TEE_Param params[4])
{
	printf("batchnorm_ta\n");
	return TEE_SUCCESS;
}
