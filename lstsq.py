import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
#import pandas as pd
'''
Executing a Kernel
write the corresponding CUDA C code, and feed it into the constructor of a pycuda.compiler.SourceModule:
'''
mod1 = SourceModule("""
    #include <stdio.h>
  __global__ void add(int n,float *x, float*y, float *xsum, float *ysum, float *xsq, float *xy, float *coef)
  {
    int idx = threadIdx.x + threadIdx.y*n;
    atomicAdd(xsum, x[idx]);
    atomicAdd(ysum, y[idx]);
    atomicAdd(xsq, x[idx]*x[idx]);
    atomicAdd(xy, x[idx]*y[idx]);
    

  }
  """)

mod = SourceModule("""
    #include <stdio.h>
  __global__ void add(int xsize, float *x, float*y,float *a,float *b, float *chi2)
  {
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    //printf("threadIdx.x= %d\\n",idx);
    //printf("threadIdx.y= %d\\n",idy);
    float xsum=0;
    float ysum=0;
    float xsq=0;
    float xy=0;
    for(int k=0;k<xsize;k++)
    {
        xsum+=x[k];
	ysum+=y[xsize*idy+k];
	xsq+=x[k]*x[k];
        xy+=x[k]*y[xsize*idy+k];
    }
    //__syncthreads();
    a[idy]=((xsum*ysum - xsize*xy)*1.0/(xsum*xsum -xsize*xsq)*1.0);
    b[idy]=(ysum-a[idy]*xsum)*1.0/xsize*1.0;

    for(int i=0;i<xsize;i++)
    {
     float res=y[xsize*idy+i]-(a[idy]*x[i]+b[idy]);
     chi2[idy]+=res*res;
    }
    chi2[idy]=chi2[idy]/(xsize-2);
  }
  """)

def lstsqr(x,y):
    '''
    x,y consists of double precision numbers, but most nVidia devices only support single precision:
    '''
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if not isinstance(y[0], np.ndarray):
         y = np.array([y])
    '''
    # allocate memory on the device:
    '''
    x_gpu = cuda.mem_alloc(x.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    a_gpu = cuda.mem_alloc(y[:,1].nbytes)
    b_gpu = cuda.mem_alloc(y[:,1].nbytes)
    chi2_gpu = cuda.mem_alloc(y[:,1].nbytes)
    '''
    #transfer the data to the GPU
    '''
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(y_gpu, y)
    thready=(np.shape(y[:,1]))
    threadx=np.shape(x)
    narr=np.int32(threadx[0])
    func = mod.get_function("add")
    func(narr, x_gpu, y_gpu, a_gpu, b_gpu, chi2_gpu,block=(threadx[0],thready[0],1))
    a=np.empty_like(y[:,1])
    b=np.empty_like(y[:,1])
    chi2=np.empty_like(y[:,1])
    cuda.memcpy_dtoh(a, a_gpu)
    cuda.memcpy_dtoh(b, b_gpu)
    cuda.memcpy_dtoh(chi2, chi2_gpu)
    #print(chi2)
    dict={}
    for A, B, C, Y,  in zip(chi2,a,b,y):
        dict[A]=A, (B,C), Y
    minkey=(min(dict, key=dict.get))
    #print(minkey)
    #print(dict.get(minkey)[1]
    return dict.get(minkey)[1], dict.get(minkey)[2], minkey
def lstsqr_s(x,y):
   coef=np.zeros(2)
   coef=np.float32(coef)
   x = x.astype(np.float32)
   y = y.astype(np.float32)
   coef = coef.astype(np.float32)
   thready=np.shape(y)
   threadx=np.shape(x)
   narr=np.int32(threadx[0])

   '''
   allocate memory on the device:
   '''
   x_gpu = cuda.mem_alloc(x.nbytes)
   y_gpu = cuda.mem_alloc(y.nbytes)
   xsum_gpu = cuda.mem_alloc(x[0].nbytes)
   ysum_gpu = cuda.mem_alloc(y[0].nbytes)
   xsq_gpu = cuda.mem_alloc(x[0].nbytes)
   xy_gpu = cuda.mem_alloc(y[0].nbytes)
   coef_gpu = cuda.mem_alloc(coef.nbytes)
   '''
   transfer the data to the GPU
   '''
   cuda.memcpy_htod(x_gpu, x)
   cuda.memcpy_htod(y_gpu, y)
   cuda.memcpy_htod(coef_gpu, coef)
   func = mod1.get_function("add")
   func(narr,x_gpu, y_gpu, xsum_gpu, ysum_gpu, xsq_gpu, xy_gpu, coef_gpu,block=(threadx[0],thready[0],1))
   sumx = np.empty_like(x[0])
   sumy = np.empty_like(y[0])
   sumx2 = np.empty_like(x[0])
   sumxy=np.empty_like(x[0])
   cuda.memcpy_dtoh(sumx, xsum_gpu)
   cuda.memcpy_dtoh(sumy, ysum_gpu)
   cuda.memcpy_dtoh(sumx2, xsq_gpu)
   cuda.memcpy_dtoh(sumxy, xy_gpu)
   a=((sumx*sumy - narr*sumxy)*1.0/(sumx*sumx -narr*sumx2)*1.0)
   b=(sumy - a*sumx)*1.0/narr*1.0
   coef[0]=a
   coef[1]=b
   chi2=0
   for i in range(len(x)):
       res=(y[i]-(a*x[i]+b))
       res=res*res
       chi2=chi2+res
       chi2 = chi2/(len(x)-2)
   return coef, chi2
