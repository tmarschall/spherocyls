
// -*- c++ -*-
/*
* strain_dynamics.cu
*
*
*/

#include <cuda.h>
#include "spherocyl_box.h"
#include "cudaErr.h"
#include <math.h>
#include <sm_20_atomic_functions.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

using namespace std;

const double D_PI = 3.14159265358979;


///////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
//  Energy Minimization  //
///////////////////////////


template<Potential ePot>
__global__ void calc_fe(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, double *pdMOI, 
			double *pdFx, double *pdFy, double *pdFt, float *pfEnergy, double *pdEnergyBlocks)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  sData[thid] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on
  
  while (nPID < nSpherocyls)
    {
      double dFx = 0.0;
      double dFy = 0.0;
      double dFt = 0.0;
      
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nSpherocyls];
	  double dAdjX = pdX[nAdjPID];
	  double dAdjY = pdY[nAdjPID];
	  
	  double dDeltaX = dX - dAdjX;
	  double dDeltaY = dY - dAdjY;
	  double dPhiB = pdPhi[nAdjPID];
	  //if (dPhi == dPhiB) {
	    //printf("Spherocyls %d, %d parallel\n", nPID, nAdjPID);
	  //}
	  double dSigma = dR + pdR[nAdjPID];
	  double dB = pdA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dLx * ((dDeltaX < -0.5*dLx) - (dDeltaX > 0.5*dLx));
	  dDeltaY += dLy * ((dDeltaY < -0.5*dLy) - (dDeltaY > 0.5*dLy));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;
	  
	  double nxA = dA * cos(dPhi);
	  double nyA = dA * sin(dPhi);
	  double nxB = dB * cos(dPhiB);
	  double nyB = dB * sin(dPhiB);

	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s,t;
	  if (delta <= 5e-14) {
	    double s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    double s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (s1 == s2) {
	      s = s1;
	      t = fmin(fmax( -(s*b+e)/c, -1.0), 1.0);
	      double dDx = dDeltaX + s*nxA - t*nxB;
	      double dDy = dDeltaY + s*nyA - t*nyB;
	      double dDSqr = dDx * dDx + dDy * dDy;
	      if (dDSqr < dSigma*dSigma)
		{
		  double dDij = sqrt(dDSqr);
		  double dDVij;
		  double dAlpha;
		  if (ePot == HARMONIC)
		    {
		      dDVij = (1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.0;
		    }
		  else if (ePot == HERTZIAN)
		    {
		      dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.5;
		    }
		  double dPfx = dDx * dDVij / dDij;
		  double dPfy = dDy * dDVij / dDij;
		  dFx += dPfx;
		  dFy += dPfy;
		  dFt += s*nxA * dPfy - s*nyA * dPfx;
		  sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		}
	    }
	    else {
	      double t1,t2;
	      if (b < 0) {
		t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
		t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	      }
	      else {
		t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
		t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	      }
	      double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	      double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	      double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	      if (dDSqr1 < dSigma*dSigma)
		{
		  double dDij = sqrt(dDSqr1);
		  double dDVij;
		  double dAlpha;
		  if (ePot == HARMONIC)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.0;
		    }
		  else if (ePot == HERTZIAN)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.5;
		    }
		  double dPfx = dDx1 * dDVij / dDij;
		  double dPfy = dDy1 * dDVij / dDij;
		  dFx += dPfx;
		  dFy += dPfy;
		  dFt += s1*nxA * dPfy - s1*nyA * dPfx;
		  sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		}
	      double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	      double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	      double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	      if (dDSqr2 < dSigma*dSigma)
		{
		  double dDij = sqrt(dDSqr2);
		  double dDVij;
		  double dAlpha;
		  if (ePot == HARMONIC)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.0;
		    }
		  else if (ePot == HERTZIAN)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.5;
		    }
		  double dPfx = dDx2 * dDVij / dDij;
		  double dPfy = dDy2 * dDVij / dDij;
		  dFx += dPfx;
		  dFy += dPfy;
		  dFt += s2*nxA * dPfy - s2*nyA * dPfx;
		  sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		}
	    }
	  }
	  else {
	    t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s = -(b*t+d)/a;
	    double sarg = fabs(s);
	    s = fmin( fmax(s,-1.), 1. );
	    if (sarg > 1) 
	      t = fmin( fmax( -(b*s+e)/c, -1.), 1.);
	 
	    // Check if they overlap and calculate forces
	    double dDx = dDeltaX + s*nxA - t*nxB;
	    double dDy = dDeltaY + s*nyA - t*nyB;
	    double dDSqr = dDx * dDx + dDy * dDy;
	    if (dDSqr < dSigma*dSigma)
	      {
		double dDij = sqrt(dDSqr);
		double dDVij;
		double dAlpha;
		if (ePot == HARMONIC)
		  {
		    dDVij = (1.0 - dDij / dSigma) / dSigma;
		    dAlpha = 2.0;
		  }
		else if (ePot == HERTZIAN)
		  {
		    dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		    dAlpha = 2.5;
		  }
		double dPfx = dDx * dDVij / dDij;
		double dPfy = dDy * dDVij / dDij;
		dFx += dPfx;
		dFy += dPfy;
		//  Find the point of contact (with respect to the center of the spherocyl)
		//double dCx = s*nxA - 0.5*dDx;
		//double dCy = s*nyA - 0.5*dDy; 
		//double dCx = s*nxA;
		//double dCy = s*nyA;
		dFt += s*nxA * dPfy - s*nyA * dPfx;
		sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      }
	  }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      pdFt[nPID] = dFt;
      
      nPID += nThreads;
    }
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = blockDim.x / 2;  // stride is 1/2 block size, all threads perform two adds
    int base = thid;
    while (stride > 32)
      {
	if (thid < stride)
	  {
	    sData[base] += sData[base + stride];
	  }
	stride /= 2;  
	__syncthreads();
      }
    if (thid < 32) {
      sData[base] += sData[base + 32];
      if (thid < 16) {
	sData[base] += sData[base + 16];
	if (thid < 8) {
	  sData[base] += sData[base + 8];
	  if (thid < 4) {
	    sData[base] += sData[base + 4];
	    if (thid < 2) {
	      sData[base] += sData[base + 2];
	      if (thid == 0) {
		sData[0] += sData[1];
		pdEnergyBlocks[blockIdx.x] = sData[0];
		float tot = atomicAdd(pfEnergy, (float)sData[0]);	    
	      }
	    }
	  }
	}  
      }
    }
}

template<Potential ePot>
__global__ void calc_energy(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			    double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, float *pfEnergy, 
			    double *pdEnergyBlocks)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  sData[thid] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on
  
  while (nPID < nSpherocyls)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nSpherocyls];
	  double dAdjX = pdX[nAdjPID];
	  double dAdjY = pdY[nAdjPID];
	  
	  double dDeltaX = dX - dAdjX;
	  double dDeltaY = dY - dAdjY;
	  double dPhiB = pdPhi[nAdjPID];
	  //if (dPhi == dPhiB) {
	  //printf("Spherocyls %d, %d parallel\n", nPID, nAdjPID);
	  //}
	  double dSigma = dR + pdR[nAdjPID];
	  double dB = pdA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dLx * ((dDeltaX < -0.5*dLx) - (dDeltaX > 0.5*dLx));
	  dDeltaY += dLy * ((dDeltaY < -0.5*dLy) - (dDeltaY > 0.5*dLy));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;
	  
	  double nxA = dA * cos(dPhi);
	  double nyA = dA * sin(dPhi);
	  double nxB = dB * cos(dPhiB);
	  double nyB = dB * sin(dPhiB);

	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s,t;
	  if (delta <= 5e-14) {
	    double s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    double s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (s1 == s2) {
	      s = s1;
	      t = fmin(fmax( -(s*b+e)/c, -1.0), 1.0);
	      double dDx = dDeltaX + s*nxA - t*nxB;
	      double dDy = dDeltaY + s*nyA - t*nyB;
	      double dDSqr = dDx * dDx + dDy * dDy;
	      if (dDSqr < dSigma*dSigma)
		{
		  double dDij = sqrt(dDSqr);
		  double dDVij;
		  double dAlpha;
		  if (ePot == HARMONIC)
		    {
		      dDVij = (1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.0;
		    }
		  else if (ePot == HERTZIAN)
		    {
		      dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.5;
		    }
		  
		  sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		}
	    }
	    else {
	      double t1,t2;
	      if (b < 0) {
		t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
		t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	      }
	      else {
		t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
		t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	      }
	      double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	      double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	      double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	      if (dDSqr1 < dSigma*dSigma)
		{
		  double dDij = sqrt(dDSqr1);
		  double dDVij;
		  double dAlpha;
		  if (ePot == HARMONIC)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.0;
		    }
		  else if (ePot == HERTZIAN)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.5;
		    }
		  
		  sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		}
	      double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	      double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	      double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	      if (dDSqr2 < dSigma*dSigma)
		{
		  double dDij = sqrt(dDSqr2);
		  double dDVij;
		  double dAlpha;
		  if (ePot == HARMONIC)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.0;
		    }
		  else if (ePot == HERTZIAN)
		    {
		      dDVij = 0.5*(1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		      dAlpha = 2.5;
		    }
		  
		  sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		}
	    }
	  }
	  else {
	    t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s = -(b*t+d)/a;
	    double sarg = fabs(s);
	    s = fmin( fmax(s,-1.), 1. );
	    if (sarg > 1) 
	      t = fmin( fmax( -(b*s+e)/c, -1.), 1.);
	 
	    // Check if they overlap and calculate forces
	    double dDx = dDeltaX + s*nxA - t*nxB;
	    double dDy = dDeltaY + s*nyA - t*nyB;
	    double dDSqr = dDx * dDx + dDy * dDy;
	    if (dDSqr < dSigma*dSigma)
	      {
		double dDij = sqrt(dDSqr);
		double dDVij;
		double dAlpha;
		if (ePot == HARMONIC)
		  {
		    dDVij = (1.0 - dDij / dSigma) / dSigma;
		    dAlpha = 2.0;
		  }
		else if (ePot == HERTZIAN)
		  {
		    dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		    dAlpha = 2.5;
		  }
	
		sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      }
	  }
	}
      
      nPID += nThreads;
    }
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = blockDim.x / 2;  // stride is 1/2 block size
    int base = thid;
    while (stride > 32)
      {
	if (thid < stride)
	  {
	    sData[base] += sData[base + stride];
	  }
	stride /= 2;  
	__syncthreads();
      }
    if (thid < 32) {
      sData[base] += sData[base + 32];
      if (thid < 16) {
	sData[base] += sData[base + 16];
	if (thid < 8) {
	  sData[base] += sData[base + 8];
	  if (thid < 4) {
	    sData[base] += sData[base + 4];
	    if (thid < 2) {
	      sData[base] += sData[base + 2];
	      if (thid == 0) {
		sData[0] += sData[1];
		pdEnergyBlocks[blockIdx.x] = sData[0];
		float tot = atomicAdd(pfEnergy, (float)sData[0]);	    
	      }
	    }
	  }
	}  
      }
    }
}

template<Potential ePot>
__global__ void calc_ftsq(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			  double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, float *pfFtSq, 
			  double *pdBlockSums)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  sData[thid] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on
  
  while (nPID < nSpherocyls)
    {
      //double dFx = 0;
      //double dFy = 0;
      double dFt = 0;

      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dPhi = pdPhi[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nSpherocyls];
	  double dAdjX = pdX[nAdjPID];
	  double dAdjY = pdY[nAdjPID];
	  
	  double dDeltaX = dX - dAdjX;
	  double dDeltaY = dY - dAdjY;
	  double dPhiB = pdPhi[nAdjPID];
	  //if (dPhi == dPhiB) {
	  //printf("Spherocyls %d, %d parallel\n", nPID, nAdjPID);
	  //}
	  double dSigma = dR + pdR[nAdjPID];
	  double dB = pdA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dLx * ((dDeltaX < -0.5*dLx) - (dDeltaX > 0.5*dLx));
	  dDeltaY += dLy * ((dDeltaY < -0.5*dLy) - (dDeltaY > 0.5*dLy));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;
	  
	  double nxA = dA * cos(dPhi);
	  double nyA = dA * sin(dPhi);
	  double nxB = dB * cos(dPhiB);
	  double nyB = dB * sin(dPhiB);

	  double s, t;
	  if (dA == 0) {
	    s = 0;
	    if (dB == 0) {
	      t = 0;
	    }
	    else {
	      double c = dB*dB;
	      double e = nxB * dDeltaX + nyB * dDeltaY;
	      t = fmin(fmax(e/c, -1), 1);
	    }
	  }
	  else {
	    if (dB == 0) {
	      t = 0;
	      double a = dA*dA;
	      double d = nxA * dDeltaX + nyA * dDeltaY;
	      s = fmin(fmax(-d/a, -1), 1);
	    }
	    else {
	      double a = dA * dA;
	      double b = -(nxA * nxB + nyA * nyB);
	      double c = dB * dB;
	      double d = nxA * dDeltaX + nyA * dDeltaY;
	      double e = -nxB * dDeltaX - nyB * dDeltaY;
	      double delta = a * c - b * b;
	      if (delta <= 0) {
		double s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
		double s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
		s = (s1 + s2) / 2;
		t  = fmin(fmax( -(b*s+e)/c, -1.0), 1.0);
	      }
	      else {
		t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
		s = -(b*t+d)/a;
		double sarg = fabs(s);
		s = fmin( fmax(s,-1.), 1. );
		if (sarg > 1) 
		  t = fmin( fmax( -(b*s+e)/c, -1.), 1.);
	      }
	    }
	  }

	  // Check if they overlap and calculate forces
	  double dDx = dDeltaX + s*nxA - t*nxB;
	  double dDy = dDeltaY + s*nyA - t*nyB;
	  double dDSqr = dDx * dDx + dDy * dDy;
	  if (dDSqr < dSigma*dSigma)
	    {
	      double dDij = sqrt(dDSqr);
	      double dDVij;
	      double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDx * dDVij / dDij;
	      double dPfy = dDy * dDVij / dDij;
	      //dFx += dPfx;
	      //dFy += dPfy;
	      dFt += s*nxA * dPfy - s*nyA * dPfx;
	      
	    }
	}

      sData[thid] += dFt*dFt / nSpherocyls;
      nPID += nThreads;
    }
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = blockDim.x / 2;  // stride is 1/2 block size
    int base = thid;
    while (stride > 32)
      {
	if (thid < stride)
	  {
	    sData[base] += sData[base + stride];
	  }
	stride /= 2;  
	__syncthreads();
      }
    if (thid < 32) {
      sData[base] += sData[base + 32];
      if (thid < 16) {
	sData[base] += sData[base + 16];
	if (thid < 8) {
	  sData[base] += sData[base + 8];
	  if (thid < 4) {
	    sData[base] += sData[base + 4];
	    if (thid < 2) {
	      sData[base] += sData[base + 2];
	      if (thid == 0) {
		sData[0] += sData[1];
		pdBlockSums[blockIdx.x] = sData[0];
		float tot = atomicAdd(pfFtSq, (float)sData[0]);	    
	      }
	    }
	  }
	}  
      }
    }
}

__global__ void square_forces(int nDim, double *pdFx, double *pdFy, double *pdFt, double *pdMOI, float *pfSquare)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  int arrid = thid + (blid/3) * blockDim.x;
  extern __shared__ double sData[];
  sData[thid] = 0;
  
  double val;
  while (arrid < nDim) {
    if (blid % 3 == 0)
      val = pdFx[arrid];
    else if (blid % 3 == 1)
      val = pdFy[arrid];
    else
      val = pdFt[arrid];
      //val = pdFt[arrid] / pdMOI[arrid]; 
    sData[thid] += val*val;
    arrid += nThreads;
  }
  __syncthreads();

  int stride = blockDim.x / 2;
  while (stride > 32) {
    if (thid < stride) {
      sData[thid] += sData[thid + stride];
    }
    stride /= 2;
    __syncthreads();
  }
  if (thid < 32) {
    sData[thid] += sData[thid + 32];
    if (thid < 16) {
      sData[thid] += sData[thid + 16];
      if (thid < 8) {
	sData[thid] += sData[thid + 8];
	if (thid < 4) {
	  sData[thid] += sData[thid + 4];
	  if (thid < 2) {
	    sData[thid] += sData[thid + 2];
	    if (thid == 0) {
	      sData[0] += sData[1];
	      float tot = atomicAdd(pfSquare, (float)sData[0]);	    
	    }
	  }
	}
      }  
    }
  }
    
}

__global__ void dbl_square_forces(int nDim, double *pdFx, double *pdFy, double *pdFt, double *pdMOI, double *pdBlockSums)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  int arrid = thid + (blid/3) * blockDim.x;
  extern __shared__ double sData[];
  sData[thid] = 0;
  
  double val;
  while (arrid < nDim) {
    if (blid % 3 == 0)
      val = pdFx[arrid];
    else if (blid % 3 == 1)
      val = pdFy[arrid];
    else
      val = pdFt[arrid];
      //val = pdFt[arrid] / pdMOI[arrid]; 
    sData[thid] += val*val;
    arrid += nThreads;
  }
  __syncthreads();

  int stride = blockDim.x / 2;
  while (stride > 32) {
    if (thid < stride) {
      sData[thid] += sData[thid + stride];
    }
    stride /= 2;
    __syncthreads();
  }
  if (thid < 32) {
    sData[thid] += sData[thid + 32];
    if (thid < 16) {
      sData[thid] += sData[thid + 16];
      if (thid < 8) {
	sData[thid] += sData[thid + 8];
	if (thid < 4) {
	  sData[thid] += sData[thid + 4];
	  if (thid < 2) {
	    sData[thid] += sData[thid + 2];
	    if (thid == 0) {
	      sData[0] += sData[1];
	      pdBlockSums[blid] = sData[0];	    
	    }
	  }
	}
      }  
    }
  }
    
}


__global__ void mult_forces(int nDim, double *pdFx0, double *pdFy0, double *pdFt0, double *pdFx, 
			    double *pdFy, double *pdFt, double *pdMOI, float *pfFF0)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  int arrid = thid + (blid/3) * blockDim.x;
  extern __shared__ double sData[];
  sData[thid] = 0;
  
  while (arrid < nDim) {
    if (blid % 3 == 0)
      sData[thid] += pdFx0[arrid] * pdFx[arrid];
    else if (blid % 3 == 1)
      sData[thid] += pdFy0[arrid] * pdFy[arrid];
    else
      sData[thid] += pdFt0[arrid] * pdFt[arrid];

    arrid += nThreads;
  }
  __syncthreads();

  int stride = blockDim.x / 2;
  while (stride > 32) {
    if (thid < stride) {
      sData[thid] += sData[thid + stride];
    }
    stride /= 2;
    __syncthreads();
  }
  if (thid < 32) {
    sData[thid] += sData[thid + 32];
    if (thid < 16) {
      sData[thid] += sData[thid + 16];
      if (thid < 8) {
	sData[thid] += sData[thid + 8];
	if (thid < 4) {
	  sData[thid] += sData[thid + 4];
	  if (thid < 2) {
	    sData[thid] += sData[thid + 2];
	    if (thid == 0) {
	      sData[0] += sData[1];
	      float tot = atomicAdd(pfFF0, (float)sData[0]);	    
	    }
	  }
	}
      }  
    }
  }
    
}

__global__ void new_pr_conj_dir(int nSpherocyls, double *pdFx, double *pdFy, double *pdFt, 
				double *pdMOI, double *pdDx, double *pdDy, double *pdDt, 
				float *pfF0Sq, float *pfFSq, float *pfFF0, double dAngleDamping)
{
  int thid = threadIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  //int nThreads = blockDim.x * gridDim.x / 2;
  int nPID = thid + (blockIdx.x / 3) * blockDim.x;
  //int nPID = thid + (blockIdx.x / 2) * blockDim.x;
  float fBeta = (*pfFSq - *pfFF0) / (*pfF0Sq);

  if (blockIdx.x % 3 == 0) {
    //if (blockIdx.x % 2 == 0) {
    while (nPID < nSpherocyls)  {
      double dNewDx = pdFx[nPID] + fBeta * pdDx[nPID];
      pdDx[nPID] = dNewDx;
      nPID += nThreads;
    }
  }
  else if (blockIdx.x % 3 == 1) {
    //else {
    while (nPID < nSpherocyls)  {
      double dNewDy = pdFy[nPID] + fBeta * pdDy[nPID];
      pdDy[nPID] = dNewDy;
      nPID += nThreads;
    }
  }
  else {
    while (nPID < nSpherocyls)  {
      double dNewDt = pdFt[nPID] + fBeta * pdDt[nPID];
      //double dNewDt = pdFt[nPID] + fBeta * pdDt[nPID];
      pdDt[nPID] = dAngleDamping*dNewDt;
      nPID += nThreads;
    }
  }

}

__global__ void new_fr_conj_dir(int nSpherocyls, double *pdFx, double *pdFy, double *pdFt, double *pdMOI,
				double *pdDx, double *pdDy, double *pdDt, float *pfF0Sq, float *pfFSq)
{
  int thid = threadIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  //int nThreads = blockDim.x * gridDim.x / 2;
  int nPID = thid + (blockIdx.x / 3) * blockDim.x;
  //int nPID = thid + (blockIdx.x / 2) * blockDim.x;
  float fBeta = (*pfFSq) / (*pfF0Sq);

  if (blockIdx.x % 3 == 0) {
    //if (blockIdx.x % 2 == 0) {
    while (nPID < nSpherocyls)  {
      double dNewDx = pdFx[nPID] + fBeta * pdDx[nPID];
      pdDx[nPID] = dNewDx;
      nPID += nThreads;
    }
  }
  else if (blockIdx.x % 3 == 1) {
    //else {
    while (nPID < nSpherocyls)  {
      double dNewDy = pdFy[nPID] + fBeta * pdDy[nPID];
      pdDy[nPID] = dNewDy;
      nPID += nThreads;
    }
  }
  
  else {
    while (nPID < nSpherocyls)  {
      double dNewDt = pdFt[nPID] + fBeta * pdDt[nPID];
      //double dNewDt = pdFt[nPID] + fBeta * pdDt[nPID];
      pdDt[nPID] = dNewDt;
      nPID += nThreads;
    }
  }
  
}

__global__ void temp_move_step(int nSpherocyls, double *pdX, double *pdY, double *pdPhi, 
			       double *pdTempX, double *pdTempY, double *pdTempPhi, 
			       double *pdDx, double *pdDy, double *pdDt, double dStep)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nPID = thid + (blid / 3) * blockDim.x;
  //int nPID = thid + (blid / 2) * blockDim.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  //int nThreads = blockDim.x * gridDim.x / 2;
  //printf("t: %d, b: %d, p: %d\n", thid, blid, nPID);
  
  
  if (blid % 3 == 0) {
    //if (blid % 2 == 0) {
    while (nPID < nSpherocyls) {
      pdTempX[nPID] = pdX[nPID] + dStep * pdDx[nPID];
      nPID += nThreads;
    }
  }
  else if (blid % 3 == 1) {
    //else {
    while (nPID < nSpherocyls) {
      pdTempY[nPID] = pdY[nPID] + dStep * pdDy[nPID];
      nPID += nThreads;
    }
  }
  
  else {
    while (nPID < nSpherocyls) {
      pdTempPhi[nPID] = pdPhi[nPID] + dStep * pdDt[nPID];
      nPID += nThreads;
    }
  }
  
}

__global__ void move_step(int nSpherocyls, double *pdX, double *pdY, double *pdPhi,
			  double *pdDx, double *pdDy, double *pdDt, double dStep)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nPID = thid + (blid / 3) * blockDim.x;
  //int nPID = thid + (blid / 2) * blockDim.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  //int nThreads = blockDim.x * gridDim.x / 2;
  
  if (blid % 3 == 0) {
    //if (blid % 2 == 0) {
    while (nPID < nSpherocyls) {
      pdX[nPID] = pdX[nPID] + dStep * pdDx[nPID];
      nPID += nThreads;
    }
  }
  else if (blid % 3 == 1) {
    //else {
    while (nPID < nSpherocyls) {
      pdY[nPID] = pdY[nPID] + dStep * pdDy[nPID];
      nPID += nThreads;
    }
  }
  
  else {
    while (nPID < nSpherocyls) {
      pdPhi[nPID] = pdPhi[nPID] + dStep * pdDt[nPID];
      nPID += nThreads;
    }
  }
  
}


int Spherocyl_Box::line_search(bool bFirstStep, bool bSecondStep, double dMinStep, double dMaxStep)
{
  bool bFindMin = true;
  bool bMinStep = false;
  if (bFirstStep) {
    cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
    
    //temp_move_step <<<2*m_nGridSize, m_nBlockSize>>> 
    temp_move_step <<<3*m_nGridSize, m_nBlockSize>>>  
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_energy <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pfSE+4, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_pfSE+1, d_pfSE+1, sizeof(float), cudaMemcpyDeviceToHost);
    h_pdLineEnergy[1] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[1] += h_pdBlockSums[j];
    }
    //printf("Step Energy: %.10g\n", dEnergy); 
    if (h_pdLineEnergy[1] > h_pdLineEnergy[0]) {
      //printf("First step %g, line energies: %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1]); 
      if (m_dStep > dMinStep) {
	m_dStep /= 2;
	h_pdLineEnergy[2] = h_pdLineEnergy[1];
	h_pdLineEnergy[1] = 0;
	//cudaMemcpyAsync(d_pfSE+1, h_pfSE+1, 2*sizeof(float), cudaMemcpyHostToDevice);
	int ret = line_search(1,0, dMinStep, dMaxStep);
	if (ret == 1) {
	  return 1;
	}
	else if (ret == 2) {
	  return 2;
	}
	bFindMin = false;
      }
      else {
	bMinStep = true;
      }
    }
  }
  if (bSecondStep) {
    cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));

    //temp_move_step <<<2*m_nGridSize, m_nBlockSize>>>
    temp_move_step <<<3*m_nGridSize, m_nBlockSize>>> 
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, 2*m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_energy <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pfSE+4, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_pfSE+2, d_pfSE+2, sizeof(float), cudaMemcpyDeviceToHost);
    h_pdLineEnergy[2] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[2] += h_pdBlockSums[j];
    }
    //printf("Step Energy: %.10g\n", dEnergy);
    if (h_pdLineEnergy[2] <= h_pdLineEnergy[1]) {
      //printf("Second step %g , line energies: %.10g %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
      if (m_dStep < dMaxStep) {
	m_dStep *= 2;
	h_pdLineEnergy[1] = h_pdLineEnergy[2];
	h_pdLineEnergy[2] = 0;
	//cudaMemcpyAsync(d_pfSE+1, h_pfSE+1, 2*sizeof(float), cudaMemcpyHostToDevice);
	int ret = line_search(0,1, dMinStep, dMaxStep);
	if (ret == 1) {
	  return 1;
	}
	else if (ret == 2) {
	  return 2;
	}
	bFindMin = false;
      }
    }
  }

  if (bFindMin) {
    if (h_pdLineEnergy[1] > h_pdLineEnergy[0]) {
      printf("Line search result, step %g, line energies %.10g %.10g %.10g\n", 
	     m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
      printf("Cannot find lower energy\n");
      return 1;
    }
    else {
      double dLineMin = (3*h_pdLineEnergy[0] - 4*h_pdLineEnergy[1] + h_pdLineEnergy[2])/(2*h_pdLineEnergy[0] - 4*h_pdLineEnergy[1] + 2*h_pdLineEnergy[2]);
      if (dLineMin > 2) {
	dLineMin = 2;
      }
      else if (h_pdLineEnergy[1] == h_pdLineEnergy[0]) {
	dLineMin = 1;
      }       
      printf("Line search result, step %g, line energies %.12g %.12g %.12g, line min %g\n", 
           m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2], dLineMin);
      move_step <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, dLineMin*m_dStep);
      cudaDeviceSynchronize();
      checkCudaError("Moving Spherocyls");
    }
  }

  if (bMinStep)
    return 2;
  return 0;
}


int Spherocyl_Box::force_line_search(bool bFirstStep, bool bSecondStep, double dMinStep, double dMaxStep)
{
  bool bFindMin = true;
  bool bMinStep = false;
  if (bFirstStep) {
    cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
    
    //temp_move_step <<<2*m_nGridSize, m_nBlockSize>>> 
    temp_move_step <<<3*m_nGridSize, m_nBlockSize>>>  
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, 
       d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE+4, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");

    dbl_square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Summing forces");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_pfSE+1, d_pfSE+1, sizeof(float), cudaMemcpyDeviceToHost);
    h_pdLineEnergy[1] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[1] += h_pdBlockSums[j];
    }
    //printf("Step Energy: %.10g\n", dEnergy); 
    if (h_pdLineEnergy[1] > h_pdLineEnergy[0]) {
      //printf("First step %g, line energies: %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1]); 
      if (m_dStep > dMinStep) {
	m_dStep /= 2;
	h_pdLineEnergy[2] = h_pdLineEnergy[1];
	h_pdLineEnergy[1] = 0;
	//cudaMemcpyAsync(d_pfSE+1, h_pfSE+1, 2*sizeof(float), cudaMemcpyHostToDevice);
	int ret = line_search(1,0, dMinStep, dMaxStep);
	if (ret == 1) {
	  return 1;
	}
	else if (ret == 2) {
	  return 2;
	}
	bFindMin = false;
      }
      else {
	bMinStep = true;
      }
    }
  }
  if (bSecondStep) {
    cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));

    //temp_move_step <<<2*m_nGridSize, m_nBlockSize>>>
    temp_move_step <<<3*m_nGridSize, m_nBlockSize>>> 
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, 2*m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, 
       d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE+4, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");

    dbl_square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Summing forces");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_pfSE+2, d_pfSE+2, sizeof(float), cudaMemcpyDeviceToHost);
    h_pdLineEnergy[2] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[2] += h_pdBlockSums[j];
    }
    //printf("Step Energy: %.10g\n", dEnergy);
    if (h_pdLineEnergy[2] <= h_pdLineEnergy[1]) {
      //printf("Second step %g , line energies: %.10g %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
      if (m_dStep < dMaxStep) {
	m_dStep *= 2;
	h_pdLineEnergy[1] = h_pdLineEnergy[2];
	h_pdLineEnergy[2] = 0;
	//cudaMemcpyAsync(d_pfSE+1, h_pfSE+1, 2*sizeof(float), cudaMemcpyHostToDevice);
	int ret = line_search(0,1, dMinStep, dMaxStep);
	if (ret == 1) {
	  return 1;
	}
	else if (ret == 2) {
	  return 2;
	}
	bFindMin = false;
      }
    }
  }

  if (bFindMin) {
    if (h_pdLineEnergy[1] > h_pdLineEnergy[0]) {
      printf("Line search result, step %g, line energies %.10g %.10g %.10g\n", 
	     m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
      printf("Cannot find lower energy\n");
      return 1;
    }
    else {
      double dLineMin = (3*h_pdLineEnergy[0] - 4*h_pdLineEnergy[1] + h_pdLineEnergy[2])/(2*h_pdLineEnergy[0] - 4*h_pdLineEnergy[1] + 2*h_pdLineEnergy[2]);
      if (dLineMin > 2) {
	dLineMin = 2;
      }
      else if (h_pdLineEnergy[1] == h_pdLineEnergy[0]) {
	dLineMin = 1;
      }       
      //printf("Line search result, step %g, line energies %.12g %.12g %.12g, line min %g\n", 
      //     m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2], dLineMin);
      move_step <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, dLineMin*m_dStep);
      cudaDeviceSynchronize();
      checkCudaError("Moving Spherocyls");
    }
  }

  if (bMinStep)
    return 2;
  return 0;
}


int Spherocyl_Box::cjpr_relax_step(double dMinStep, double dMaxStep, double dAngleDamping)
{
  float *d_pfF0Square;
  float *d_pfFSquare;
  float *d_pfFF0;
  cudaMalloc((void **) &d_pfF0Square, sizeof(float));
  cudaMalloc((void **) &d_pfFSquare, sizeof(float));
  cudaMalloc((void **) &d_pfFF0, sizeof(float));
  cudaMemset((void *) d_pfF0Square, 0, sizeof(float));
  cudaMemset((void *) d_pfFSquare, 0, sizeof(float));
  cudaMemset((void *) d_pfFF0, 0, sizeof(float));
  cudaMemset((void *) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
  cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));
  checkCudaError("Setting memory");

  cudaMemcpyAsync(d_pdTempFx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  //if (!bAngle) {
  //  cudaMemset((void *) d_pdTempFt, 0 , m_nSpherocyls*sizeof(double));
  //} 
  
  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums); 

  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfF0Square);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces 1");

  //if (!bAngle) {
  //  cudaMemset((void *) d_pdFt, 0, m_nSpherocyls*sizeof(double));
  //}
  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfFSquare);
  mult_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfFF0);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  
  cudaMemcpyAsync(h_pfSE, d_pfSE, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  float *h_pfF0Square;
  float *h_pfFSquare;
  //float *h_pfFF0;
  h_pfF0Square = (float*) malloc(sizeof(float));
  h_pfFSquare = (float*) malloc(sizeof(float));
  //h_pfFF0 = (float*) malloc(sizeof(float));
  cudaMemcpy(h_pfF0Square, d_pfF0Square, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_pfFF0, d_pfFF0, sizeof(float), cudaMemcpyDeviceToHost);
  //printf("F0-square: %g, F-Square: %g, F*F0: %g\n", *h_pfF0Square, *h_pfFSquare, *h_pfFF0);
  if (*h_pfFSquare == 0) {
    printf("Zero energy found, stopping minimization");
    return 1;
  }
  cudaDeviceSynchronize();
  h_pdLineEnergy[4] = h_pdLineEnergy[0];
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  if (h_pdLineEnergy[4] == h_pdLineEnergy[0] && *h_pfFSquare == *h_pfF0Square) {
    printf("Minimum Energy Found, stopping minimization");
    return 2;
  }

  //new_conjugate_direction <<<2*m_nGridSize, m_nBlockSize>>>
  new_pr_conj_dir <<<3*m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdDx, d_pdDy, d_pdDt, 
     d_pfF0Square, d_pfFSquare, d_pfFF0, dAngleDamping);
  cudaDeviceSynchronize();
  checkCudaError("Finding new conjugate direction");

  //if (!bAngle) {
  //  cudaMemset((void *) d_pdDt, 0, m_nSpherocyls*sizeof(double));
  //}

  m_dStep = 0.000001 / sqrt(*h_pfFSquare/(3*m_nSpherocyls));
  int ret = line_search(1,1, dMinStep, dMaxStep);
  if (ret == 1) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 0.000001 / sqrt(*h_pfFSquare/(3*m_nSpherocyls));
    cudaDeviceSynchronize();
    ret = line_search(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      printf("Stopping due to step size\n");
      return 1;
    }
  }
  else if (ret == 2) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 0.000001 / sqrt(*h_pfFSquare/(3*m_nSpherocyls));
    cudaDeviceSynchronize();
    ret = line_search(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      printf("Stopping due to step size\n");
      return 1;
    }
  }

  m_dMove += 3*m_dStep*sqrt(*h_pfFSquare/(3*m_nSpherocyls));
  if (m_dMove > m_dEpsilon) {
    find_neighbors();
    m_dMove = 0;
  }

  free(h_pfF0Square); free(h_pfFSquare);
  cudaFree(d_pfF0Square); cudaFree(d_pfFSquare); cudaFree(d_pfFF0);

  return 0;
}

int Spherocyl_Box::force_cjpr_relax_step(double dMinStep, double dMaxStep, bool bAngle = true)
{
  float *d_pfF0Square;
  float *d_pfFSquare;
  float *d_pfFF0;
  cudaMalloc((void **) &d_pfF0Square, sizeof(float));
  cudaMalloc((void **) &d_pfFSquare, sizeof(float));
  cudaMalloc((void **) &d_pfFF0, sizeof(float));
  cudaMemset((void *) d_pfF0Square, 0, sizeof(float));
  cudaMemset((void *) d_pfFSquare, 0, sizeof(float));
  cudaMemset((void *) d_pfFF0, 0, sizeof(float));
  cudaMemset((void *) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
  cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));
  checkCudaError("Setting memory");

  cudaMemcpyAsync(d_pdTempFx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  if (!bAngle) {
    cudaMemset((void *) d_pdTempFt, 0 , m_nSpherocyls*sizeof(double));
  } 
  
  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums); 


  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfF0Square);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces 1");

  if (!bAngle) {
    cudaMemset((void *) d_pdFt, 0, m_nSpherocyls*sizeof(double));
  }

  dbl_square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Summing forces");
  
  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfFSquare);
  
  mult_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfFF0);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  
  cudaMemcpyAsync(h_pfSE, d_pfSE, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  float *h_pfF0Square;
  float *h_pfFSquare;
  //float *h_pfFF0;
  h_pfF0Square = (float*) malloc(sizeof(float));
  h_pfFSquare = (float*) malloc(sizeof(float));
  //h_pfFF0 = (float*) malloc(sizeof(float));
  cudaMemcpy(h_pfF0Square, d_pfF0Square, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_pfFF0, d_pfFF0, sizeof(float), cudaMemcpyDeviceToHost);
  //printf("F0-square: %g, F-Square: %g, F*F0: %g\n", *h_pfF0Square, *h_pfFSquare, *h_pfFF0);
  if (*h_pfFSquare == 0) {
    printf("Zero energy found, stopping minimization");
    return 1;
  }
  cudaDeviceSynchronize();
  h_pdLineEnergy[4] = h_pdLineEnergy[0];
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  if (h_pdLineEnergy[4] == h_pdLineEnergy[0] && *h_pfFSquare == *h_pfF0Square) {
    printf("Minimum Energy Found, stopping minimization");
    return 2;
  }

  //new_conjugate_direction <<<2*m_nGridSize, m_nBlockSize>>>
  new_pr_conj_dir <<<3*m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdDx, d_pdDy, d_pdDt, d_pfF0Square, d_pfFSquare, d_pfFF0, 1.0);
  cudaDeviceSynchronize();
  checkCudaError("Finding new conjugate direction");

  if (!bAngle) {
    cudaMemset((void *) d_pdDt, 0, m_nSpherocyls*sizeof(double));
  }

  m_dStep = 0.00001 / sqrt(*h_pfFSquare);
  int ret = line_search(1,1, dMinStep, dMaxStep);
  if (ret == 1) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 0.00001 / sqrt(*h_pfFSquare);
    cudaDeviceSynchronize();
    ret = line_search(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      printf("Stopping due to step size\n");
      return 1;
    }
  }
  else if (ret == 2) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 0.00001 / sqrt(*h_pfFSquare);
    cudaDeviceSynchronize();
    ret = line_search(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      printf("Stopping due to step size\n");
      return 1;
    }
  }

  m_dMove += m_dStep*(*h_pfFSquare);
  if (m_dMove > m_dEpsilon) {
    find_neighbors();
    m_dMove = 0;
  }

  free(h_pfF0Square); free(h_pfFSquare);
  cudaFree(d_pfF0Square); cudaFree(d_pfFSquare); cudaFree(d_pfFF0);

  return 0;
}


int Spherocyl_Box::cjfr_relax_step(double dMinStep, double dMaxStep)
{
  float *d_pfF0Square;
  float *d_pfFSquare;
  cudaMalloc((void **) &d_pfF0Square, sizeof(float));
  cudaMalloc((void **) &d_pfFSquare, sizeof(float));
  cudaMemset((void *) d_pfF0Square, 0, sizeof(float));
  cudaMemset((void *) d_pfFSquare, 0, sizeof(float));
  cudaMemset((void *) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
  cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));
  checkCudaError("Setting memory");
  
  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
          (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
	   d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfF0Square);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces");

  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
          (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
	   d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  cudaMemcpyAsync(h_pfSE, d_pfSE, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
	     
  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfFSquare);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces");

  float *h_pfF0Square;
  float *h_pfFSquare;
  h_pfF0Square = (float*) malloc(sizeof(float));
  h_pfFSquare = (float*) malloc(sizeof(float));
  cudaMemcpy(h_pfF0Square, d_pfF0Square, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  printf("F-square: %g  %g\n", *h_pfF0Square, *h_pfFSquare);
  if (*h_pfFSquare == 0) {
    printf("Stopping due to zero Energy\n");
    return 1;
  }
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  //printf("Energy: %.10g %.10g\n", h_pfSE[0], dEnergy);

  //new_conjugate_direction <<<2*m_nGridSize, m_nBlockSize>>>
  new_fr_conj_dir <<<3*m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdDx, d_pdDy, d_pdDt, d_pfF0Square, d_pfFSquare);
  //cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  //cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  //cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  checkCudaError("Finding new conjugate direction");

  m_dStep = 0.00001 / sqrt(*h_pfFSquare);
  int ret = line_search(1,1, dMinStep, dMaxStep);
  if (ret == 1) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 0.00001 / sqrt(*h_pfFSquare);
    cudaDeviceSynchronize();
    ret = line_search(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      printf("Stopping due to step size\n");
      return 1;
    }
  }
  //else if (ret == 2) {
    

  free(h_pfF0Square); free(h_pfFSquare);
  cudaFree(d_pfF0Square); cudaFree(d_pfFSquare);

  return 0;
}

int Spherocyl_Box::gd_relax_step(double dMinStep, double dMaxStep)
{
  float *d_pfFSquare;
  cudaMalloc((void **) &d_pfFSquare, sizeof(float));
  cudaMemset((void *) d_pfFSquare, 0, sizeof(float));
  cudaMemset((void *) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
  cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));
  checkCudaError("Setting memory");
  
  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
          (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
	   d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdDx, d_pdDy, d_pdDt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pfSE, d_pfSE, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);

  square_forces <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdDx, d_pdDy, d_pdDt, d_pdMOI, d_pfFSquare);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces");

  float *h_pfFSquare;
  h_pfFSquare = (float*) malloc(sizeof(float));
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  printf("F-square: %g\n", *h_pfFSquare);
  if (*h_pfFSquare == 0) {
    printf("Stopping due to zero Energy\n");
    return 1;
  }
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }

  m_dStep = 0.00001 / sqrt(*h_pfFSquare);
  int ret = line_search(1,1, dMinStep, dMaxStep);
  if (ret == 1) {
    printf("Stopping due to step size\n");
    return 1;
  }
  //else if (ret == 2) {    

  free(h_pfFSquare);
  cudaFree(d_pfFSquare);

  return 0;
}

void Spherocyl_Box::gd_relax(double dMinStep, double dMaxStep)
{

  cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));

  for (int i = 0; i < 1000000; i++) {
    //m_dStep = 0.1;
    int ret = gd_relax_step(dMinStep, dMaxStep);
    if (ret == 1) {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      save_positions(i);
      break;
    }  
    find_neighbors();
  }

}

void Spherocyl_Box::cjfr_relax(double dMinStep, double dMaxStep)
{

  cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));

  for (int i = 0; i < 1000000; i++) {
    //m_dStep = 0.1;
    int ret = cjfr_relax_step(dMinStep, dMaxStep);
    if (ret == 1) {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      save_positions(i);
      break;
    }  
    find_neighbors();
  }

}


void Spherocyl_Box::cjpr_relax(double dMinStep, double dMaxStep, int nMaxSteps = 10000, double dAngleDamping = 1.0)
{

  cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));

  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  h_pdLineEnergy[0] = 0;
  m_dMove = 0;
  find_neighbors();

  long unsigned int nTime = 0;
  while (nTime < nMaxSteps) {
    //m_dStep = 0.1;
    int ret = cjpr_relax_step(dMinStep, dMaxStep, dAngleDamping);
    //printf("Step return: %d\n", ret);
    if (ret == 1 || ret == 2) {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      //save_positions(nTime);
      break;
    }  
    //find_neighbors();
    nTime += 1;
  }

  find_neighbors();

  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }

}

void Spherocyl_Box::force_cjpr_relax(double dMinStep, double dMaxStep, int nMaxSteps = 10000)
{
  bool bAngle = true;

  cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));

  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  h_pdLineEnergy[0] = 0;
  m_dMove = 0;
  find_neighbors();

  long unsigned int nTime = 0;
  while (nTime < nMaxSteps) {
    //m_dStep = 0.1;
    int ret = force_cjpr_relax_step(dMinStep, dMaxStep, bAngle);
    //printf("Step return: %d\n", ret);
    if (ret == 1 || ret == 2) {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      //save_positions(nTime);
      break;
    }  
    //find_neighbors();
    nTime += 1;
  }

  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }

}


void Spherocyl_Box::quasistatic_compress(double dMaxPack, double dResizeStep, double dMinStep = 1e-10) {
  assert(dResizeStep != 0 && dResizeStep != -0);
  dResizeStep = fabs(dResizeStep);
  calculate_packing();
  assert(dMaxPack >= m_dPacking);

  int nCompressTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }
  
  cjpr_relax(dMinStep, 200);
  save_positions(nCompressTime);
  calculate_stress_energy();
  fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nCompressTime, m_dPacking, h_pfSE[0], h_pfSE[1], h_pfSE[2], h_pfSE[3], h_pfSE[4]);

  while(h_pdLineEnergy[0] > 0) {
    nCompressTime += 1;
    m_dStep = 0.01;
    resize_step(nCompressTime, dResizeStep, 0, 0);
    cjpr_relax(dMinStep, 200);

    calculate_stress_energy();
    cudaMemcpy(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    save_positions(nCompressTime);
    calculate_packing();
    fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nCompressTime, m_dPacking, h_pfSE[0], h_pfSE[1], h_pfSE[2], h_pfSE[3], h_pfSE[4]);
    fflush(m_pOutfSE);
  }

  while (m_dPacking < dMaxPack) {
    nCompressTime += 1;
    m_dStep = 0.01;
    resize_step(nCompressTime, -dResizeStep, 0, 0);
    //cjpr_relax(dMinStep, 200, true);
    //cjpr_relax(10*dMinStep, 200, false);
    cjpr_relax(dMinStep, 200);

    calculate_stress_energy();
    cudaMemcpy(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    save_positions(nCompressTime);
    calculate_packing();
    fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nCompressTime, m_dPacking, h_pfSE[0], h_pfSE[1], h_pfSE[2], h_pfSE[3], h_pfSE[4]);
    fflush(m_pOutfSE);
  }
  fclose(m_pOutfSE);
}

int Spherocyl_Box::qs_one_cycle(int nTime, double dMaxE, double dResizeStep, double dMinStep = 1e-10) {
  assert(dResizeStep != 0 && dResizeStep != -0);
  dResizeStep = fabs(dResizeStep);
  calculate_packing();
  
  cjpr_relax(dMinStep, 200);
  save_positions(nTime);
  calculate_packing();
  fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nTime, m_dPacking, h_pfSE[0], h_pfSE[1], h_pfSE[2], h_pfSE[3], h_pfSE[4]);

  while(h_pdLineEnergy[0] > 0) {
    nTime += 1;
    m_dStep = 0.01;
    resize_step(nTime, dResizeStep, 0, 0);
    cjpr_relax(dMinStep, 200, 10000, 1.0);
    cjpr_relax(dMinStep, 20, 5000, 0);
    cjpr_relax(dMinStep, 50, 2000, 0.04);
    cjpr_relax(dMinStep, 100, 2000, 0.1);
    cjpr_relax(dMinStep, 200, 2000, 0.4);
    cjpr_relax(dMinStep, 200, 5000, 1.0);

    calculate_stress_energy();
    cudaMemcpy(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    save_positions(nTime);
    calculate_packing();
    fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nTime, m_dPacking, h_pfSE[0], h_pfSE[1], h_pfSE[2], h_pfSE[3], h_pfSE[4]);
    fflush(m_pOutfSE);
  }

  while (h_pdLineEnergy[0] < dMaxE) {
    nTime += 1;
    m_dStep = 0.01;
    resize_step(nTime, -dResizeStep, 0, 0);
    cjpr_relax(dMinStep, 200, 10000, 1.0);
    cjpr_relax(dMinStep, 20, 5000, 0);
    cjpr_relax(dMinStep, 50, 2000, 0.04);
    cjpr_relax(dMinStep, 100, 2000, 0.1);
    cjpr_relax(dMinStep, 200, 2000, 0.4);
    cjpr_relax(dMinStep, 200, 5000, 1.0);

    calculate_stress_energy();
    cudaMemcpy(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    save_positions(nTime);
    calculate_packing();
    fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nTime, m_dPacking, h_pfSE[0], h_pfSE[1], h_pfSE[2], h_pfSE[3], h_pfSE[4]);
    fflush(m_pOutfSE);
  }

  return nTime;
}

void Spherocyl_Box::quasistatic_cycle(int nCycles, double dMaxE, double dResizeStep, double dMinStep = 1e-10)
{
  
  int nTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }

  for (int c = 0; c < nCycles; c++) {
    nTime = qs_one_cycle(nTime, dMaxE, dResizeStep, dMinStep);
    nTime += 1;
  }

  fclose(m_pOutfSE);
}

void Spherocyl_Box::quasistatic_find_jam(double dMaxE, double dMinE, double dResizeStep, double dMinStep = 1e-10)
{
  
  int nTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }

  double dMinResizeStep = m_dLx*1e-15;
  while (dMaxE > dMinE && dResizeStep > dMinResizeStep) {
    nTime = qs_one_cycle(nTime, dMaxE, dResizeStep, dMinStep);
    dMaxE *= 0.5;
    dResizeStep *= 0.5;
    nTime += 1;
  }

  fclose(m_pOutfSE);
}

int Spherocyl_Box::resize_to_energy_1p(int nTime, double dEnergy, double dStep)
{
  dStep = fabs(dStep);
  double dCJMinStep = 1e-16;
  double dCJMaxStep = 10;
  int nCJMaxSteps = 400000;

  calc_fe <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  h_pdLineEnergy[0] = 0;
  cudaDeviceSynchronize();
  for (int i = 0; i < m_nGridSize; i++) {
    h_pdLineEnergy[0] += h_pdBlockSums[i];
  }

  if (h_pdLineEnergy[0] < dEnergy) {
    while (h_pdLineEnergy[0] < dEnergy) {
      nTime += 1;
      m_dStep = 0.01;
      resize_step(nTime, -dStep, 0, 0);
      cjpr_relax(dCJMinStep, dCJMaxStep, nCJMaxSteps, 1.0);

      calculate_stress_energy();
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      calculate_packing();
      save_positions(nTime);
      cudaDeviceSynchronize();
      fprintf(m_pOutfSE, "%d %.9f %g %g %g %g %g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
      fflush(m_pOutfSE);
      dStep *= 1.1;
    }
  }
  else {
    while (h_pdLineEnergy[0] > dEnergy) {
      nTime += 1;
      m_dStep = 0.01;
      resize_step(nTime, dStep, 0, 0);
      cjpr_relax(dCJMinStep, dCJMaxStep, nCJMaxSteps, 1.0);

      calculate_stress_energy();
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      calculate_packing();
      save_positions(nTime);
      cudaDeviceSynchronize();
      fprintf(m_pOutfSE, "%d %.9f %g %g %g %g %g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
      fflush(m_pOutfSE);
      dStep *= 1.1;
    }
  }
  
  return nTime;
}

void Spherocyl_Box::find_jam_1p(double dJamE, double dResizeStep)
{
  int nTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }
  save_positions(0);

  double dMinResizeStep = m_dLx*1e-13;
  int nOldTime = 0;
  while (dResizeStep > dMinResizeStep) {
    nTime = resize_to_energy_1p(nTime, dJamE, dResizeStep);
    dResizeStep = dResizeStep*pow(1.1, (nTime-nOldTime))/2;
    nOldTime = nTime;
  }
  //save_positions(9999999999);
  save_positions_bin(nTime);

  fclose(m_pOutfSE);
}
