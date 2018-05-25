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
  int use_cl;

} ffm_parameter;

__constant int const kALIGNByte = 4;
__constant int const kALIGN = kALIGNByte / sizeof(float);

inline int get_k_aligned(int k)
{
  return (int) ceil((float)k / kALIGN) * kALIGN;
}

double
wTx(__global struct ffm_node* X, size_t begin, size_t end, size_t x_size,
    __global float *weight, int feature, int fields, int latents,
    float lambda, float kappa, float scale, float eta, int do_update)
{
  int align0 = 2 * get_k_aligned(latents);
  int align1 = fields * align0;

  float t = 0;

  for(size_t N1 = begin; N1 != end; N1++)
    {
      if (N1 >= x_size)
	printf("N1:%lu\n", N1);

      int j1 = X[N1].j;
      int f1 = X[N1].f;
      float v1 = X[N1].v;

      if(j1 >= feature || f1 >= fields)
	{
	  continue;
	}

      for (size_t N2 = N1+1; N2 != end; N2++)
	{
	  if (N2 >= x_size)
	    printf("N2:%lu\n", N2);

	  int j2 = X[N2].j;
	  int f2 = X[N2].f;
	  float v2 = X[N2].v;

	  if(j2 >= feature || f2 >= fields)
	    {
	      continue;
	    }

	  long w1i = (long)j1 * align1 + f2 * align0;
	  long w2i = (long)j2 * align1 + f1 * align0;

	  float v = v1 * v2 * scale;

	  if(do_update)
	    {
	      int wg1i = w1i + kALIGN;
	      int wg2i = w2i + kALIGN;

	      for(int d = 0; d < align0; d += kALIGN * 2)
	  	{
	  	  float g1 = lambda * weight[w1i+d] + kappa * weight[w2i+d] * v;
	  	  float g2 = lambda * weight[w2i+d] + kappa * weight[w1i+d] * v;

		  weight[wg1i+d] = g1 * g1;
		  weight[wg2i+d] = g2 * g2;

		  weight[w1i+d] -= eta / sqrt(weight[wg1i+d]) * g1;
		  weight[w2i+d] -= eta / sqrt(weight[wg2i+d]) * g2;
	  	}
	    }
	  else
	    {
	      for (int d = 0; d < align0; d += kALIGN * 2)
		t += weight[w1i+d] * weight[w2i+d] * v;
	    }
	}
    }
  return t;
}

__kernel void
update_block(__global struct ffm_node *x, int x_size,
	     __global float *Y, int y_size,
	     __global long *nnzs, int nnz_size,
	     __global float *scales, int scales_size,

	     __global float *weight, int feature, int fields, int latents,

	     struct ffm_parameter param, int l, int do_update,

	     __global float *loss)
{
  int gid = get_global_id(0);
  int i = gid;

  float y = Y[i];

  if (nnzs[i] >= x_size)
    printf("nnzs[i]: %lu\n", nnzs[i]);

  size_t begin_index = nnzs[gid];
  size_t end_index = nnzs[i+1];

  float scale = param.normalization? scales[i] : 1;

  double t = wTx(x, begin_index, end_index, x_size,
  		 weight, feature, fields, latents,
  		 0, 0, scale, 0, 0);

  double expnyt = exp(-y*t);

  loss[0] += log1p(expnyt);
  /* printf("loss: %f\n", loss[0]); */
  if(do_update)
    {

      float kappa = -y * expnyt /(1 + expnyt);

      wTx(x, begin_index, end_index, x_size,
      	  weight, feature, fields, latents,
      	  param.lambda, kappa, scale, param.eta, 1);
    }
}
