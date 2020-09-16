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

mod = SourceModule("""
    #include <stdio.h>
  __global__ void add(int xsize, float *x, float*y,float *a,float *b, float *chi2)
  {
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int idz = threadIdx.z;
    //printf("blockDim.x= %d\\n",blockDim.x);
    //printf("blockDim.y= %d\\n",blockDim.y);
    //printf("blockDim.z= %d\\n",blockDim.z);
    //printf("threadIdx.y= %d\\n",idy);
    float xsum=0;
    float ysum=0;
    float xsq=0;
    float xy=0;
    float dy=0;
    for(int k=0;k<xsize;k++)
    {
        //printf("x=%d\\n",blockDim.x*idz);
        xsum+=x[blockDim.x*idz +k];
	ysum+=y[blockDim.x*blockDim.y*idz+blockDim.x*idy+k];
	xsq+=x[blockDim.x*idz+k]*x[blockDim.x*idz+k];
        xy+=x[blockDim.x*idz+k]*y[blockDim.x*blockDim.y*idz+blockDim.x*idy+k];
    }
    //__syncthreads();
    a[idz*blockDim.y+idy]=((xsum*ysum - xsize*xy)*1.0/(xsum*xsum -xsize*xsq)*1.0);
    b[idz*blockDim.y+idy]=(ysum-a[idz*blockDim.y+idy]*xsum)*1.0/xsize*1.0;

    for(int i=0;i<xsize;i++)
    {
     float res=y[blockDim.x*blockDim.y*idz+blockDim.x*idy+i]-(a[idz*blockDim.y+idy]*x[blockDim.x*idz+i]+b[idz*blockDim.y+idy]);
     chi2[idz*blockDim.y+idy]+=res*res;
    }

  }
  """)
def lstsq_3d(x,y,chem):
    print("x",x)
    print("np.shape(x)",np.shape(x))
    print("np.shape(y)",np.shape(y))
    print("np.shape(y[0,:,0])",np.shape(y[0,:,0])) # y axis
    print("np.shape(y[:,0,0])",np.shape(y[:,0,0])) # z axis
    print("np.shape(y[0,0,:])",np.shape(y[0,0,:])) # x axis
    print("x[1,:]",x[1,:])
    '''
    #But wait–a consists of double precision numbers, but most nVidia devices only support single precision:
    '''
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    '''
    # allocate memory on the device:
    '''
    x_gpu = cuda.mem_alloc(x.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    a_gpu = cuda.mem_alloc(y[:,:,0].nbytes)
    b_gpu = cuda.mem_alloc(y[:,:,0].nbytes)
    chi2_gpu = cuda.mem_alloc(y[:,:,0].nbytes)
    '''
    #transfer the data to the GPU
    '''
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(y_gpu, y)
    threadz=(np.shape(y[:,0,0]))
    thready=(np.shape(y[0,:,0]))
    threadx=np.shape(x[0,:])
    narr=np.int32(threadx[0])
    func = mod.get_function("add")
    func(narr, x_gpu, y_gpu, a_gpu, b_gpu, chi2_gpu,block=(threadx[0],thready[0],threadz[0]))
    a=np.empty_like(y[:,:,0])
    b=np.empty_like(y[:,:,0])
    chi2=np.empty_like(y[:,:,0])
    cuda.memcpy_dtoh(a, a_gpu)
    cuda.memcpy_dtoh(b, b_gpu)
    cuda.memcpy_dtoh(chi2, chi2_gpu)
    dict={}
    coef={}
    yf={}
    res={}
    i=0
    for ch in range(np.shape(chem)[0]):
        dict={}
        for A, B, C, Y,  in zip(chi2,a,b,y):
            dict[A[i]]=A[i], (B[i],C[i]), Y[i]
        minkey=(min(dict, key=dict.get))
        coef[ch]=dict.get(minkey)[1]
        yf[ch]={"x":x[i,:],"y":dict.get(minkey)[2]}
        res[ch]=dict.get(minkey)[0]
        i=i+1
    print("coef", coef)
    return coef, yf
