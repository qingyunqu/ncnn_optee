// for debug

#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#define MY_DEBUG
#ifdef MY_DEBUG
	#define dprintf(format,...) printf(format,##__VA_ARGS__)
#else
	#define dprintf(format,...)
#endif

#endif // DEBUG_H
