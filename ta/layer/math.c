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

float pow(float x, float y){

}
