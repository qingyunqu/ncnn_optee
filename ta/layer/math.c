// quick implementation of lib "math.h", may lose part of accuracy
// only support float32

#include "math.h"

float exp(float x){ // quick implementation of exp()
	x = 1.0 + x / 256.0;
  	x *= x; x *= x; x *= x; x *= x;
  	x *= x; x *= x; x *= x; x *= x;
  	return x;
}

float max(float a, float b){ // may should implement more precisely
	return a > b ? a : b;
}

static float pow_i(float x, int y){ // y is a integer
	float tmp = 1.f;
	for(int i=0; i<y; i++)
		tmp *= x;
	return tmp;
}
static float pow_f(float x,float y){ // 0=<x<2
	float powf = 0.f;
	float tmpf = 1.f;
	float x_tmp = x - 1;
	for(int i=1; tmpf > 1e-12 || tmpf<-1e-12; i++){
		for(int j=1,tmpf=1; j<=i; j++)
			tmpf *= (y-j+1) * x_tmp / j;
		powf += tmpf;
	}
	return powf+1;
}

float pow(float x, float y){
	if(x==0 && y!=0)
		return 0.f;
	else if(x==0 && y==0)
		return 1.f;
	else if(x<0 && y-(int)y!=0)
		return 0.f;
	if(x>2){
		x = 1/x;
		y = -y;
	}
	if(y<0)
		return 1/pow(x,-y);
	if(y-(int)y==0)
		return pow_i(x,y);
	else
		return pow_f(x,y-(int)y) * pow_i(x,(int)y);
	return pow_f(x,y);
}
