// for debug

#ifndef DEBUG_H
#define DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#define MY_DEBUG
#ifdef MY_DEBUG
	#define dprintf(format,...) printf(format,##__VA_ARGS__)
#else
	#define dprintf(format,...)
#endif

#ifdef __cplusplus
}
#endif

#endif // DEBUG_H
