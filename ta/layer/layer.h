// layers' forward implementation
#ifndef LAYER_H
#define LAYER_H

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

TEE_Result batchnorm_ta(uint32_t param_types, TEE_Param params[4]);
TEE_Result pooling_ta(uint32_t param_types, TEE_Param params[4]);

#endif // LAYER_H
