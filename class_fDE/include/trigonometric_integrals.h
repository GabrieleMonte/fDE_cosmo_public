/**
 * definitions for module trigonometric_integrals.c
 */

#ifndef __TRIGONOMETRIC_INTEGRALS__
#define __TRIGONOMETRIC_INTEGRALS__

#include "common.h"
#define _EIN_ONE_ 0.79659959929705313   /* Ein(1) = gamma + E1(1) */
/**
 * Boilerplate for C++
 */
#ifdef __cplusplus
extern "C" {
#endif

  int cosine_integral(
				 double x,
				 double *Ci,
         ErrorMsg error_message
				 );

  int sine_integral(
				 double x,
				 double *Si,
         ErrorMsg error_message
				 );
  int exponential_integral_a_to_1(double a,
								  double *J,
						ErrorMsg error_message
								  );
#ifdef __cplusplus
}
#endif

#endif
