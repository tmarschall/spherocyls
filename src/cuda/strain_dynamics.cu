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

/*
__global__ void calc_rot_consts(int nSpherocyls, double *pdR, double *pdAs, 
			       double *pdAb, double *pdCOM, double *pdMOI, 
			       double *pdSinCoeff, double *pdCosCoeff)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x + gridDim.x;
  while (nPID < nSpherocyls) {
    double dR = pdR[nPID];
    double dAs = pdAs[nPID];
    double dA = dAs + 2. * dR;
    double dB = pdAb[nPID];
    double dC = pdCOM[nPID];

    double dIntdS = 2*dA + 4*dB;
    double dIntSy2SinCoeff = dA*dA*(2*dA/3 + 4*dB);
    double dIntSy2CosCoeff = dB*dB*(16*dB/3 - 8*dC) + 2*(dA + 2*dB)*dC*dC;
    double dIntS2 = dIntSy2SinCoeff + dIntSy2CosCoeff;
    pdMOI[nPID] = dIntS2 / dIntdS;
    pdSinCoeff[nPID] = dIntSy2SinCoeff / dIntS2;
    pdCosCoeff[nPID] = dIntSy2CosCoeff / dIntS2;

    nPID += nThreads;
  }
  
}
*/

///////////////////////////////////////////////////////////////
//
//   Deprecated Method
//
///////////////////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void euler_est_sc(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy,
			     double dGamma, double dStrain, double dStep, double *pdX, double *pdY,
			     double *pdPhi, double *pdR, double *pdA, double *pdMOI, double *pdFx, double *pdFy,
			     double *pdFt, float *pfSE, double *pdTempX, double *pdTempY, double *pdTempPhi)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset = blockDim.x + 8; // +8 should help to avoid a few bank conflict
  if (bCalcStress) {
    for (int i = 0; i < 5; i++)
      sData[i*offset + thid] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on
  }
  
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
	  
	  double dDeltaX = dX - pdX[nAdjPID];
	  double dDeltaY = dY - pdY[nAdjPID];
	  double dPhiB = pdPhi[nAdjPID];
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
	  double t;
	  double s;
	  if (dA == 0) {
	    s = 0;
	    if (dB == 0) {
		t = 0;
	    }
	    else {
	      double c = dB*dB;
	      double e = nxB * dDeltaX + nyB *dDeltaY;
	      t = fmin(fmax(e/c, -1), 1);
	    }
	  }
	  else {
	    if (dB == 0) {
	      t = 0;
	      double a = dA*dA;
	      double d = nxA *dDeltaX + nyA * dDeltaY;
	      s = fmin(fmax(-d/a, -1), 1);
	    }
	    else {
	      double a = dA * dA;
	      double b = -(nxA * nxB + nyA * nyB);
	      double c = dB * dB;
	      double d = nxA * dDeltaX + nyA * dDeltaY;
	      double e = -nxB * dDeltaX - nyB * dDeltaY;
	      double delta = a * c - b * b;
	      if (delta == 0) {
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
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s*nxA * dPfy - s*nyA * dPfx;
	      if (bCalcStress)
		{
		  double dCx = 0.5*dDx - s*nxA;
		  double dCy = 0.5*dDy - s*nyA;
		  sData[thid] += dCx * dPfx / (dLx * dLy);
		  sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		  sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		  sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		  if (nAdjPID > nPID)
		    {
		      sData[thid + 4*offset] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		    }
		}
	    }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      pdFt[nPID] = dFt;
      
      pdTempX[nPID] = dX + dStep * (dFx - dGamma * dFy);
      pdTempY[nPID] = dY + dStep * dFy;
      double dSinPhi = sin(dPhi);
      pdTempPhi[nPID] = dPhi + dStep * 
	(dFt / pdMOI[nPID] - dStrain * dSinPhi * dSinPhi);
      
      nPID += nThreads;
    }
  if (bCalcStress) {
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = blockDim.x / 2;  // stride is 1/2 block size, all threads perform two adds
    int base = thid % stride + offset * (thid / stride);
    sData[base] += sData[base + stride];
    base += 2*offset;
    sData[base] += sData[base + stride];
    if (thid < stride) {
      base += 2*offset;
      sData[base] += sData[base + stride];
    }
    stride /= 2; // stride is 1/4 block size, all threads perform 1 add
    __syncthreads();
    base = thid % stride + offset * (thid / stride);
    sData[base] += sData[base + stride];
    if (thid < stride) {
      base += 4*offset;
      sData[base] += sData[base+stride];
    }
    stride /= 2;
    __syncthreads();
    while (stride > 4)
      {
	if (thid < 5 * stride)
	  {
	    base = thid % stride + offset * (thid / stride);
	    sData[base] += sData[base + stride];
	  }
	stride /= 2;  
	__syncthreads();
      }
    if (thid < 20)
      {
	base = thid % 4 + offset * (thid / 4);
	sData[base] += sData[base + 4];
	if (thid < 10)
	  {
	    base = thid % 2 + offset * (thid / 2);
	    sData[base] += sData[base + 2];
	    if (thid < 5)
	      {
		sData[thid * offset] += sData[thid * offset + 1];
		float tot = atomicAdd(pfSE+thid, (float)sData[thid*offset]);	    
	      }
	  }
      }
  }  
} 


/////////////////////////////////////////////////////////////////
//
//  First step (Netonian estimate) for simple shear
//
//////////////////////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void euler_est(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			  double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, double *pdR, 
			  double *pdA, double dKd, double *pdArea, double *pdMOI, double *pdIsoC, double *pdFx, 
			  double *pdFy, double *pdFt, float *pfSE, double *pdTempX, double *pdTempY, double *pdTempPhi)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  extern __shared__ double sData[];
  int offset = blockDim.x + 8; // +8 should help to avoid a few bank conflict
  if (bCalcStress) {
    for (int i = 0; i < 5; i++)
      sData[i*offset + thid] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on
  }
  
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
	  double t;
	  double s;
	  if (dA == 0) {
	    s = 0;
	    if (dB == 0) {
		t = 0;
	    }
	    else {
	      double c = dB*dB;
	      double e = nxB * dDeltaX + nyB *dDeltaY;
	      t = fmin(fmax(e/c, -1), 1);
	    }
	  }
	  else {
	    if (dB == 0) {
	      t = 0;
	      double a = dA*dA;
	      double d = nxA *dDeltaX + nyA * dDeltaY;
	      s = fmin(fmax(-d/a, -1), 1);
	    }
	    else {
	      double a = dA * dA;
	      double b = -(nxA * nxB + nyA * nyB);
	      double c = dB * dB;
	      double d = nxA * dDeltaX + nyA * dDeltaY;
	      double e = -nxB * dDeltaX - nyB * dDeltaY;
	      double delta = a * c - b * b;
	      if (delta == 0) {
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
	      dFx += dPfx;
	      dFy += dPfy;
	      //  Find the point of contact (with respect to the center of the spherocyl)
	      //double dCx = s*nxA - 0.5*dDx;
	      //double dCy = s*nyA - 0.5*dDy; 
	      //double dCx = s*nxA;
	      //double dCy = s*nyA;
	      dFt += s*nxA * dPfy - s*nyA * dPfx;
	      if (bCalcStress)
		{
		  double dCx = 0.5*dDx - s*nxA;
		  double dCy = 0.5*dDy - s*nyA;
		  sData[thid] += dCx * dPfx / (dLx * dLy);
		  sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		  sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		  sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		  if (nAdjPID > nPID)
		    {
		      sData[thid + 4*offset] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
		    }
		}
	    }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      pdFt[nPID] = dFt;
      
      double dArea = pdArea[nPID];
      dFx /= (dKd*dArea);
      dFy /= (dKd*dArea);
      dFt /= (dKd*dArea*pdMOI[nPID]);
      pdTempX[nPID] = dX + dStep * (dFx - dGamma * dFy);
      pdTempY[nPID] = dY + dStep * dFy;
      double dRIso = 0.5*(1-pdIsoC[nPID]*cos(2*dPhi));
      pdTempPhi[nPID] = dPhi + dStep * (dFt - dStrain * dRIso);
      
      nPID += nThreads;
    }
  if (bCalcStress) {
    __syncthreads();
    
    // Now we do a parallel reduction sum to find the total number of contacts
    int stride = blockDim.x / 2;  // stride is 1/2 block size, all threads perform two adds
    int base = thid % stride + offset * (thid / stride);
    sData[base] += sData[base + stride];
    base += 2*offset;
    sData[base] += sData[base + stride];
    if (thid < stride) {
      base += 2*offset;
      sData[base] += sData[base + stride];
    }
    stride /= 2; // stride is 1/4 block size, all threads perform 1 add
    __syncthreads();
    base = thid % stride + offset * (thid / stride);
    sData[base] += sData[base + stride];
    if (thid < stride) {
      base += 4*offset;
      sData[base] += sData[base + stride];
    }
    stride /= 2;
    __syncthreads();
    while (stride > 4)
      {
	if (thid < 5 * stride)
	  {
	    base = thid % stride + offset * (thid / stride);
	    sData[base] += sData[base + stride];
	  }
	stride /= 2;  
	__syncthreads();
      }
    if (thid < 20)
      {
	base = thid % 4 + offset * (thid / 4);
	sData[base] += sData[base + 4];
	if (thid < 10)
	  {
	    base = thid % 2 + offset * (thid / 2);
	    sData[base] += sData[base + 2];
	    if (thid < 5)
	      {
		sData[thid * offset] += sData[thid * offset + 1];
		float tot = atomicAdd(pfSE+thid, (float)sData[thid*offset]);	    
	      }
	  }
      }
  }  
} 


///////////////////////////////////////////////////////////////////
//
//
/////////////////////////////////////////////////////////////////
template<Potential ePot>
__global__ void heun_corr(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			  double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, double *pdR, 
			  double *pdA, double dKd, double *pdArea, double *pdMOI, double *pdIsoC,
			  double *pdFx, double *pdFy, double *pdFt, double *pdTempX, double *pdTempY,
			  double *pdTempPhi, double *pdXMoved, double *pdYMoved, double dEpsilon, int *bNewNbrs)
{ 
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  // Declare shared memory pointer, the size is passed at the kernel launch
  
  while (nPID < nSpherocyls)
    {
      double dFx = 0.0;
      double dFy = 0.0;
      double dFt = 0.0;
      
      double dX = pdTempX[nPID];
      double dY = pdTempY[nPID];
      double dPhi = pdTempPhi[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      double dNewGamma = dGamma + dStep * dStrain;

      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nSpherocyls];
	  double dAdjX = pdTempX[nAdjPID];
	  double dAdjY = pdTempY[nAdjPID];

	  double dDeltaX = dX - dAdjX;
	  double dDeltaY = dY - dAdjY;
	  double dPhiB = pdTempPhi[nAdjPID];
	  double dSigma = dR + pdR[nAdjPID];
	  double dB = pdA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dLx * ((dDeltaX < -0.5*dLx) - (dDeltaX > 0.5*dLx));
	  dDeltaY += dLy * ((dDeltaY < -0.5*dLy) - (dDeltaY > 0.5*dLy));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dNewGamma * dDeltaY;
	  
	  double nxA = dA * cos(dPhi);
	  double nyA = dA * sin(dPhi);
	  double nxB = dB * cos(dPhiB);
	  double nyB = dB * sin(dPhiB);
	  double t;
	  double s;
	  if (dA == 0) {
	    s = 0;
	    if (dB == 0) {
		t = 0;
	    }
	    else {
	      double c = dB*dB;
	      double e = nxB * dDeltaX + nyB *dDeltaY;
	      t = fmin(fmax(e/c, -1), 1);
	    }
	  }
	  else {
	    if (dB == 0) {
	      t = 0;
	      double a = dA*dA;
	      double d = nxA *dDeltaX + nyA * dDeltaY;
	      s = fmin(fmax(-d/a, -1), 1);
	    }
	    else {
	      double a = dA * dA;
	      double b = -(nxA * nxB + nyA * nyB);
	      double c = dB * dB;
	      double d = nxA * dDeltaX + nyA * dDeltaY;
	      double e = -nxB * dDeltaX - nyB * dDeltaY;
	      double delta = a * c - b * b;
	      if (delta == 0) {
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
	      //double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDij / dSigma) / dSigma;
		  //dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  //dAlpha = 2.5;
		}
	      double dPfx = dDx * dDVij / dDij;
	      double dPfy = dDy * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      //double dCx = s*nxA - 0.5*dDx;
	      //double dCy = s*nyA - 0.5*dDy;
	      double dCx = s*nxA;
	      double dCy = s*nyA;
	      dFt += dCx * dPfy - dCy * dPfx;
	    }
	}
  
      double dMOI = pdMOI[nPID];
      double dIsoC = pdIsoC[nPID];
      double dArea = pdArea[nPID];
      dFx /= (dKd*dArea);
      dFy /= (dKd*dArea);
      dFt /= (dKd*dArea*dMOI);
      
      dFx -= dNewGamma * dFy;
      dFt = dFt - dStrain * 0.5 * (1 - dIsoC*cos(2*dPhi));

      double dFy0 = pdFy[nPID] / (dKd*dArea);
      double dFx0 = pdFx[nPID] / (dKd*dArea) - dGamma * dFy0;
      double dPhi0 = pdPhi[nPID] / (dKd*dArea*dMOI);
      
      double dFt0 = pdFt[nPID] - dStrain * 0.5 * (1 - dIsoC*cos(2*dPhi0));

      double dDx = 0.5 * dStep * (dFx0 + dFx);
      double dDy = 0.5 * dStep * (dFy0 + dFy);
      pdX[nPID] += dDx;
      pdY[nPID] += dDy;
      pdPhi[nPID] += 0.5 * dStep * (dFt0 + dFt);
       
      pdXMoved[nPID] += dDx;
      pdYMoved[nPID] += dDy;
      if (fabs(pdXMoved[nPID]) + fabs(pdYMoved[nPID]) > 0.5*dEpsilon)
	*bNewNbrs = 1;

      nPID += nThreads;
    }
}


////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////
void Spherocyl_Box::strain_step(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, d_pdFx, 
	     d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, d_pdFx, 
	     d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions, calculating stresses");

      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      if (bSvPos)
	{
	  cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	}
      cudaThreadSynchronize();
    }
  else
    {
      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions");
    }

  switch (m_ePotential)
    {
    case HARMONIC:
      heun_corr <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, 
	 d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      heun_corr <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, 
	 d_pdYMoved, m_dEpsilon, d_bNewNbrs);
    }

  if (bSvStress)
    {
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g %.7g\n", 
	      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
      if (bSvPos)
	save_positions(tTime);
    }

  cudaThreadSynchronize();
  checkCudaError("Updating estimates, moving particles");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);

  m_dGamma += m_dStep * m_dStrainRate;
  m_dTotalGamma += m_dStep * m_dStrainRate;
  cudaThreadSynchronize();

  if (m_dGamma > 0.5*m_dLx/m_dLy)
    set_back_gamma();
  else if (*h_bNewNbrs)
    find_neighbors();
  else if (m_dGamma - floor(m_dGamma) < 1.1 * m_dStep * m_dStrainRate)
    find_neighbors();
}

void Spherocyl_Box::strain_step_ngl(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions, calculating stresses");

      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      if (bSvPos)
	{
	  cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	}
      cudaThreadSynchronize();
    }
  else
    {
      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	}
      cudaThreadSynchronize();
      checkCudaError("Estimating new particle positions");
    }

  switch (m_ePotential)
    {
    case HARMONIC:
      heun_corr <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, 
	 d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      heun_corr <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, 
	 d_pdYMoved, m_dEpsilon, d_bNewNbrs);
    }

  if (bSvStress)
    {
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g %.7g\n", 
	      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
      if (bSvPos)
	save_positions(tTime);
    }

  cudaThreadSynchronize();
  checkCudaError("Updating estimates, moving particles");
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);

  m_dGamma += m_dStep * m_dStrainRate;
  m_dTotalGamma += m_dStep * m_dStrainRate;
  cudaThreadSynchronize();

  if (m_dGamma > 0.5*m_dLx/m_dLy)
    find_neighbors();
  else if (*h_bNewNbrs)
    find_neighbors();
  else if (m_dGamma - int(m_dGamma) < 1.1 * m_dStep * m_dStrainRate)
    find_neighbors();
}


/////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////
void Spherocyl_Box::save_positions(long unsigned int nTime)
{
  char szBuf[150];
  sprintf(szBuf, "%s/sd%010lu.dat", m_szDataDir, nTime);
  const char *szFilePos = szBuf;
  FILE *pOutfPos;
  pOutfPos = fopen(szFilePos, "w");
  if (pOutfPos == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }
  
  int i = h_pnMemID[0];
  fprintf(pOutfPos, "%d %.17g %.17g %f %.17g %.14g %g %g\n",
	  m_nSpherocyls, m_dLx, m_dLy, m_dPacking, m_dGamma, m_dTotalGamma, m_dStrainRate, m_dStep);
  for (int p = 0; p < m_nSpherocyls; p++)
    {
      i = h_pnMemID[p];
      fprintf(pOutfPos, "%.17g %.17g %.17g %.2g %.17g\n",
	      h_pdX[i], h_pdY[i], h_pdPhi[i], h_pdR[i], h_pdA[i]);
    }

  fclose(pOutfPos); 
}

void Spherocyl_Box::save_positions_bin(long unsigned int nTime)
{
  char szBuf[150];
  sprintf(szBuf, "%s/sd%010lu.bin", m_szDataDir, nTime);
  const char *szFilePos = szBuf;
  FILE *pOutfPos;
  pOutfPos = fopen(szFilePos, "wb");
  if (pOutfPos == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }
  
  fwrite(&m_nSpherocyls, sizeof(m_nSpherocyls), 1, pOutfPos);
  fwrite(&m_dLx, sizeof(m_dLx), 1, pOutfPos);
  fwrite(&m_dLy, sizeof(m_dLy), 1, pOutfPos);
  fwrite(&m_dPacking, sizeof(m_dPacking), 1, pOutfPos);
  fwrite(&m_dGamma, sizeof(m_dGamma), 1, pOutfPos);
  fwrite(&m_dTotalGamma, sizeof(m_dTotalGamma), 1, pOutfPos);
  fwrite(&m_dStrainRate, sizeof(m_dStrainRate), 1, pOutfPos);
  fwrite(&m_dStep, sizeof(m_dStep), 1, pOutfPos);
  for (int p = 0; p < m_nSpherocyls; p++) {
    int i = h_pnMemID[p];
    fwrite(h_pdX+i, sizeof(double), 1, pOutfPos);
    fwrite(h_pdY+i, sizeof(double), 1, pOutfPos);
    fwrite(h_pdPhi+i, sizeof(double), 1, pOutfPos);
    fwrite(h_pdR+i, sizeof(double), 1, pOutfPos);
    fwrite(h_pdA+i, sizeof(double), 1, pOutfPos);
  }

  fclose(pOutfPos);
}


////////////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////////////
void Spherocyl_Box::run_strain(double dStartGamma, double dStopGamma, double dSvStressGamma, double dSvPosGamma)
{
  if (m_dStrainRate == 0.0)
    {
      fprintf(stderr, "Cannot strain with zero strain rate\n");
      exit(1);
    }

  printf("Beginnig strain run with strain rate: %g and step %g\n", m_dStrainRate, m_dStep);
  fflush(stdout);

  if (dSvStressGamma < m_dStrainRate * m_dStep)
    dSvStressGamma = m_dStrainRate * m_dStep;
  if (dSvPosGamma < m_dStrainRate)
    dSvPosGamma = m_dStrainRate;

  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = (unsigned long)(dStartGamma / m_dStrainRate + 0.5);
  unsigned long int nStop = (unsigned long)(dStopGamma / m_dStrainRate + 0.5);
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = (unsigned int)(dSvStressGamma / (m_dStrainRate * m_dStep) + 0.5);
  unsigned int nSvPosInterval = (unsigned int)(dSvPosGamma / m_dStrainRate + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Strain run configured\n");
  printf("Start: %lu, Stop: %lu, Int step: %lu\n", nTime, nStop, nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      unsigned long int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atol(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }

  // Run strain for specified number of steps
  while (nTime < nStop)
    {
      bool bSvPos = (nTime % nSvPosInterval == 0);
      if (bSvPos) {
	strain_step(nTime, 1, 1);
	fflush(m_pOutfSE);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  strain_step(nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  strain_step(nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  strain_step(nTime, 1, 1);
  fflush(m_pOutfSE);
  fclose(m_pOutfSE);
}


void Spherocyl_Box::run_strain(unsigned long int nStart, double dRunGamma, double dSvStressGamma, double dSvPosGamma)
{
  if (m_dStrainRate == 0.0)
    {
      fprintf(stderr, "Cannot strain with zero strain rate\n");
      exit(1);
    }

  printf("Beginnig strain run with strain rate: %g and step %g\n", m_dStrainRate, m_dStep);
  fflush(stdout);

  if (dSvStressGamma < m_dStrainRate * m_dStep)
    dSvStressGamma = m_dStrainRate * m_dStep;
  if (dSvPosGamma < m_dStrainRate)
    dSvPosGamma = m_dStrainRate;

  // +0.5 to cast to nearest integer rather than rounding down
  unsigned long int nTime = nStart;
  unsigned long int nStop = nStart + (unsigned long)(dRunGamma / m_dStrainRate + 0.5);
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = (unsigned int)(dSvStressGamma / (m_dStrainRate * m_dStep) + 0.5);
  unsigned int nSvPosInterval = (unsigned int)(dSvPosGamma / m_dStrainRate + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Strain run configured\n");
  printf("Start: %lu, Stop: %lu, Int step: %lu\n", nTime, nStop, nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_pOutfSE = fopen(szPathSE, "w");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
    }
  else
    {  
      m_pOutfSE = fopen(szPathSE, "r+");
      if (m_pOutfSE == NULL)
	{
	  fprintf(stderr, "Could not open file for writing");
	  exit(1);
	}
      
      unsigned long int nTpos = 0;
      while (nTpos != nTime)
	{
	  if (fgets(szBuf, 200, m_pOutfSE) != NULL)
	    {
	      int nPos = strcspn(szBuf, " ");
	      char szTime[20];
	      strncpy(szTime, szBuf, nPos);
	      szTime[nPos] = '\0';
	      nTpos = atol(szTime);
	    }
	  else
	    {
	      fprintf(stderr, "Reached end of file without finding start position");
	      exit(1);
	    }
	}
    }

  // Run strain for specified number of steps
  while (nTime < nStop)
    {
      bool bSvPos = (nTime % nSvPosInterval == 0);
      if (bSvPos) {
	strain_step(nTime, 1, 1);
	fflush(m_pOutfSE);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  strain_step(nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  strain_step(nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  strain_step(nTime, 1, 1);
  fflush(m_pOutfSE);
  fclose(m_pOutfSE);
}

void Spherocyl_Box::run_strain(long unsigned int nSteps)
{
  // Run strain for specified number of steps
  long unsigned int nTime = 0;
  while (nTime < nSteps)
    {
      strain_step(nTime, 0, 0);
      nTime += 1;
    }

}

void Spherocyl_Box::run_strain(double dGammaMax)
{
  reconfigure_cells(dGammaMax);
  find_neighbors();

  long unsigned int nTime = 0;
  while (m_dGamma < dGammaMax) {
    strain_step_ngl(nTime, 0, 0);
    nTime += 1;
  }

}


__global__ void resize_coords(int nParticles, double dEpsilon, double *pdX, double *pdY)
{
	int nPID = threadIdx.x + blockIdx.x*blockDim.x;
	int nThreads = blockDim.x*gridDim.x;

	while (nPID < nParticles) {
		pdX[nPID] += dEpsilon*pdX[nPID];
		pdY[nPID] += dEpsilon*pdY[nPID];

		nPID += nThreads;
	}
}

void Spherocyl_Box::resize_step(long unsigned int tTime, double dEpsilon, bool bSvStress, bool bSvPos)
{
	resize_coords <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, dEpsilon, d_pdX, d_pdY);
	m_dLx += dEpsilon*m_dLx;
	m_dLy += dEpsilon*m_dLy;
	m_dPacking = calculate_packing();
	cudaThreadSynchronize();
	checkCudaError("Resizing box");

	if (bSvStress)
	    {
	      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

	      switch (m_ePotential)
		{
		case HARMONIC:
		  euler_est <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		}
	      cudaThreadSynchronize();
	      checkCudaError("Estimating new particle positions, calculating stresses");

	      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
	      if (bSvPos)
		{
		  cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
		  cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
		  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
		}
	      cudaThreadSynchronize();
	    }
	  else
	    {
	      switch (m_ePotential)
		{
		case HARMONIC:
		  euler_est <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		}
	      cudaThreadSynchronize();
	      checkCudaError("Estimating new particle positions");
	    }

	  switch (m_ePotential)
	    {
	    case HARMONIC:
	      heun_corr <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
		 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
		 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
	      break;
	    case HERTZIAN:
	      heun_corr <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, 
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
		 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
		 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
	    }

	  if (bSvStress)
	    {
	      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
	      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g %.7g\n",
		      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
	      if (bSvPos)
	    	save_positions(tTime);
	    }

	cudaThreadSynchronize();
	checkCudaError("Updating estimates, moving particles");

	cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
	reconfigure_cells();
	cudaThreadSynchronize();

	if (*h_bNewNbrs)
	  find_neighbors();
}

void Spherocyl_Box::resize_box(long unsigned int nStart, double dEpsilon, double dFinalPacking, double dSvStressRate, double dSvPosRate)
{
	assert(dEpsilon != 0);

	m_dPacking = calculate_packing();
	if (dFinalPacking > m_dPacking && dEpsilon > 0)
		dEpsilon = -dEpsilon;
	else if (dFinalPacking < m_dPacking && dEpsilon < 0)
		dEpsilon = -dEpsilon;
	dSvStressRate = int(dEpsilon/fabs(dEpsilon))*fabs(dSvStressRate);
	dSvPosRate = int(dEpsilon/fabs(dEpsilon))*fabs(dSvPosRate);
	printf("Beginning resize with packing fraction: %f\n", m_dPacking);

	unsigned long int nTime = nStart;

	char szBuf[200];
	sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
	const char *szPathSE = szBuf;
	if (nTime == 0) {
		m_pOutfSE = fopen(szPathSE, "w");
		if (m_pOutfSE == NULL) {
			fprintf(stderr, "Could not open file for writing");
			exit(1);
		}
	}
	else {
		m_pOutfSE = fopen(szPathSE, "r+");
		if (m_pOutfSE == NULL) {
			fprintf(stderr, "Could not open file for writing");
			exit(1);
		}

		long unsigned int nTpos = 0;
		while (nTpos != nTime) {
			if (fgets(szBuf, 200, m_pOutfSE) != NULL) {
				int nPos = strcspn(szBuf, " ");
				char szTime[20];
				strncpy(szTime, szBuf, nPos);
				szTime[nPos] = '\0';
				nTpos = atol(szTime);
			}
		  else {
			  fprintf(stderr, "Reached end of file without finding start position");
		      exit(1);
		  }
		}
	}

	sprintf(szBuf, "%s/rs_stress_energy.dat", m_szDataDir);
	const char *szRsSE = szBuf;
	FILE *pOutfRs;
	if (nTime == 0) {
		pOutfRs = fopen(szRsSE, "w");
		if (pOutfRs == NULL) {
			fprintf(stderr, "Could not open file for writing");
			exit(1);
		}
	}
	else {
		pOutfRs = fopen(szRsSE, "r+");
		if (pOutfRs == NULL) {
			fprintf(stderr, "Could not open file for writing");
			exit(1);
		}
		for (unsigned long int t = 0; t < nTime; t++) {
			fgets(szBuf, 200, pOutfRs);
		}
		fgets(szBuf, 200, pOutfRs);
		int nPos = strcspn(szBuf, " ");
		char szPack[20];
		strncpy(szPack, szBuf, nPos);
		szPack[nPos] = '\0';
		double dPack = atof(szPack);
		/*
	    if (dPack - (1 + dEpsilon) * m_dPacking || dPack < (1 - dShrinkRate) * m_dPacking) {
	  	  fprintf(stderr, "System packing fraction %g does not match with time %lu", dPack, nTime);
	   	  exit(1);
     	}
		*/
	}

	int nSaveStressInt = int(log(1+dSvStressRate)/log(1+dEpsilon));
	int nSavePosInt = int(dSvPosRate/dSvStressRate+0.5)*nSaveStressInt;
	printf("Starting resize with rate: %g\nStress save rate: %g (%d)\nPosition save rate: %g (%d)\n",
			dEpsilon, dSvStressRate, nSaveStressInt, dSvPosRate, nSavePosInt);
	while (dEpsilon*(m_dPacking - dFinalPacking) > dEpsilon*dEpsilon) {
		bool bSavePos = (nTime % nSavePosInt == 0);
		bool bSaveStress = (nTime % nSaveStressInt == 0);
		//printf("Step %l\n", nTime);
		//fflush(stdout);
		resize_step(nTime, dEpsilon, bSaveStress, bSavePos);
		if (bSaveStress) {
			fprintf(pOutfRs, "%.6g %.7g %.7g %.7g %.7g %.7g %.7g\n",
					m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
		}

		if (bSavePos) {
			fflush(stdout);
			fflush(pOutfRs);
			fflush(m_pOutfSE);
		}
		nTime += 1;
		//if (nTime % nReorderInterval == 0)
		//reorder_particles();
	}
	resize_step(nTime, dEpsilon, 1, 1);
	fprintf(pOutfRs, "%.6g %.7g %.7g %.7g %.7g %.7g %.7g\n", m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
	fclose(m_pOutfSE);
	fclose(pOutfRs);

}



/////////////////////////////////////////
//
//  Relax to minimum
//
///////////////////////////////////////


void Spherocyl_Box::relax_step(long unsigned int tTime, bool bSvStress, bool bSvPos)
{

	if (bSvStress)
	    {
	      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

	      switch (m_ePotential)
		{
		case HARMONIC:
		  euler_est <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		}
	      cudaThreadSynchronize();
	      checkCudaError("Estimating new particle positions, calculating stresses");

	      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
	      if (bSvPos)
		{
		  cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
		  cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
		  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
		}
	      cudaThreadSynchronize();
	    }
	  else
	    {
	      switch (m_ePotential)
		{
		case HARMONIC:
		  euler_est <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		}
	      cudaThreadSynchronize();
	      checkCudaError("Estimating new particle positions");
	    }

	  switch (m_ePotential)
	    {
	    case HARMONIC:
	      heun_corr <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
		 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
		 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
	      break;
	    case HERTZIAN:
	      heun_corr <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
		 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
		 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
	    }

	  if (bSvStress)
	    {
	      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
	      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g %.7g\n",
		      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
	      if (bSvPos)
	    	save_positions(tTime);
	    }

	cudaThreadSynchronize();
	checkCudaError("Updating estimates, moving particles");

	cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
	reconfigure_cells();
	cudaThreadSynchronize();

	if (*h_bNewNbrs)
	  find_neighbors();
}

void Spherocyl_Box::relax_box(long unsigned int nSteps, double dMaxStep, double dMinStep, int nSaveStressInt, int nSavePosInt)
{

	unsigned long int nTime = 0;

	char szBuf[200];
	sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
	const char *szPathSE = szBuf;
	if (nTime == 0) {
		m_pOutfSE = fopen(szPathSE, "w");
		if (m_pOutfSE == NULL) {
			fprintf(stderr, "Could not open file for writing");
			exit(1);
		}
	}
	else {
		m_pOutfSE = fopen(szPathSE, "r+");
		if (m_pOutfSE == NULL) {
			fprintf(stderr, "Could not open file for writing");
			exit(1);
		}

		long unsigned int nTpos = 0;
		while (nTpos != nTime) {
			if (fgets(szBuf, 200, m_pOutfSE) != NULL) {
				int nPos = strcspn(szBuf, " ");
				char szTime[20];
				strncpy(szTime, szBuf, nPos);
				szTime[nPos] = '\0';
				nTpos = atol(szTime);
			}
		  else {
			  fprintf(stderr, "Reached end of file without finding start position");
		      exit(1);
		  }
		}
	}
	
	m_dStep = dMaxStep;
	printf("Starting relax with step size: %g\nStress save rate: %d\nPosition save rate: %d\n", m_dStep, nSaveStressInt, nSavePosInt);
	while (nTime < nSteps) {
		bool bSavePos = (nTime % nSavePosInt == 0);
		bool bSaveStress = (nTime % nSaveStressInt == 0);
		//printf("Step %l\n", nTime);
		//fflush(stdout);
		relax_step(nTime, bSaveStress, bSavePos);
		if (bSavePos) {
			fflush(stdout);
			fflush(m_pOutfSE);
		}
		nTime += 1;
		//if (nTime % nReorderInterval == 0)
		//reorder_particles();
	}
	relax_step(nTime, 1, 1);
	fclose(m_pOutfSE);

}

