#ifndef __OPENCL_VERSION__

#define __constant
#define __global
#define __local
#define __kernel
#include <math.h>
#include <stdio.h>
int get_global_id(int);

#endif

typedef struct ffm_node
{
  int f;			// field index
  int j;			// feature index
  float v;			// value
} ffm_node;

typedef struct ffm_parameter
{
  float eta;
  float lambda;
  int nr_iters;
  int k;
  int normalization;
  int auto_stop;
} ffm_parameter;

__constant int const kALIGNByte = 4;
__constant int const kALIGN = kALIGNByte/sizeof(float);
__constant int const kCHUNK_SIZE = 10000000;
__constant int const kMaxLineSize = 100000;

inline int get_k_aligned(int k)
{
  return (int) ceil((float)k / kALIGN) * kALIGN;
}

double
wTx(__global struct ffm_node * begin, __global struct ffm_node * end,
    float scale, __global float *weight, int feature, int fields, int latents,
    float lambda, float kappa, float eta, int do_update)
{
  int align0 = 2 * get_k_aligned(latents);
  int align1 = fields * align0;

  float t = 0;

  int gid = get_global_id(0);
  __global struct ffm_node *N1 = begin + gid;

  int j1 = N1->j;
  int f1 = N1->f;
  float v1 = N1->v;
  if (j1 >= feature || f1 >= fields)
    {
      return t;
    }
  for (__global struct ffm_node *N2 = N1 + 1; N2 != end; N2++)
    {
      int j2 = N2->j;
      int f2 = N2->f;
      float v2 = N2->v;
      if (j2 >= feature || f2 >= fields)
	continue;

      __global float *w1 = weight + (long)j1 * align1 + f2 * align0;
      __global float *w2 = weight + (long)j2 * align1 + f1 * align0;

      float v = v1 * v2 * scale;

      if (do_update)
	{
	  __global float *wg1 = w1 + kALIGN;
	  __global float *wg2 = w2 + kALIGN;
	  for (int d = 0; d < align0; d += kALIGN * 2)
	    {
	      float g1 = lambda * w1[d] + kappa * w2[d] * v;
	      float g2 = lambda * w2[d] + kappa * w1[d] * v;

	      wg1[d] += g1 * g1;
	      wg2[d] += g2 * g2;

	      w1[d] -= eta / sqrt(wg1[d]) * g1;
	      w2[d] -= eta / sqrt(wg2[d]) * g2;
	    }
	}
      else
	{
	  for(int d = 0; d < align0; d += kALIGN * 2)
	    t += w1[d] * w2[d] * v;
	}
    }
  return t;
}

__kernel void
update_block(__global struct ffm_node *X, __global float *Y,
	     __global int *nnzs,          __global float *scales,

	     __global float *weight, int feature, int fields, int latents,

	     struct ffm_parameter param, int l, int do_update,

	     __global float *loss)
{
  __global struct ffm_node * begin;
  __global struct ffm_node * end;

  for(int ii = 0; ii < l; ii++)
    {
      int i = ii;

      float y = Y[i];

      begin = &X[nnzs[i]]; // P[i]: number of features of

      end = &X[nnzs[i+1]];

      float scale = param.normalization? scales[i] : 1;
      /* wTx(
	 __global struct ffm_node * begin, __global struct ffm_node * end,
	 float scale, __global float *weight, int feature, int fields, int latents,
	 float lambda, float kappa, float eta, int do_update
	 )
      */
      double t = wTx(begin, end,
		     scale, weight, feature, fields, latents,
		     0, 0, 0, do_update);

      double expnyt = exp(-y*t);

      loss[0] += log1p(expnyt);

      if(do_update)
      	{

      	  float kappa = -y * expnyt /(1 + expnyt);

      	  wTx(begin, end,
	      scale, weight, feature, fields, latents,
	      kappa, param.eta, param.lambda, 1);
      	}
    }
}
