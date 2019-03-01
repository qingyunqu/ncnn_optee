// universal param used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef TEEC_TA_DEFINES_H
#define TEEC_TA_DEFINES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	size_t elemsize;
	int dims;
	
	int w;
	int h;
	int c;
	
	size_t cstep;
} Mat_C;

// see init_mat_c_from_mat() in

inline float* row(void* data, Mat_C* mat, int y){  // only support float32
	return (float*)data + mat->w * y;
}
inline float* channel(void* data, Mat_C* mat, int c){ // only support float32
	return (float*)data + mat->cstep * c;
}

#ifdef __cplusplus
}
#endif

#endif // TEEC_TA_DEFINES_H
