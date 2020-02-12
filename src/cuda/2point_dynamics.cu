
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
// Deprecated method
//
///////////////////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void euler_est_sc_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy,
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

	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (b < 0)
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    else 
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    if (b < 0)
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    else
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ >= 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg >= 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    double dPfx = dDx1 * dDVij / dDij;
	    double dPfy = dDy1 * dDVij / dDij;
	    dFx += dPfx;
	    dFy += dPfy;
	    dFt += s1*nxA * dPfy - s1*nyA * dPfx;
	    if (bCalcStress)
	      {
		double dCx = 0.5*dDx1 - s1*nxA;
		double dCy = 0.5*dDy1 - s1*nyA;
		sData[thid] += dCx * dPfx / (dLx * dLy);
		sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		sData[thid + 4*offset] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      }
	  }
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dDVij, dAlpha;
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
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
	      if (bCalcStress)
		{
		  double dCx = 0.5*dDx2 - s2*nxA;
		  double dCy = 0.5*dDy2 - s2*nyA;
		  sData[thid] += dCx * dPfx / (dLx * dLy);
		  sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		  sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		  sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		  sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
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


////////////////////////////////////////////////
//
// First step (Newtonian estimate) for integration of e.o.m.
//  Uses two points of contact along flat sides when applicable
//
////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void euler_est_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
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

	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (fabs(s1) == 1)
	      t1 = fmin(fmax( -(b*s1+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    else 
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    if (fabs(s2) == 1)
	      t2 = fmin(fmax( -(b*s2+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    else
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ > 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg > 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    double dPfx = dDx1 * dDVij / dDij;
	    double dPfy = dDy1 * dDVij / dDij;
	    dFx += dPfx;
	    dFy += dPfy;
	    //  Find the point of contact (with respect to the center of the spherocyl)
	    //double dCx = s*nxA - 0.5*dDx;
	    //double dCy = s*nyA - 0.5*dDy; 
	    //double dCx = s*nxA;
	    //double dCy = s*nyA;
	    dFt += s1*nxA * dPfy - s1*nyA * dPfx;
	    if (bCalcStress)
	      {
		double dCx = 0.5*dDx1 - s1*nxA;
		double dCy = 0.5*dDy1 - s1*nyA;
		sData[thid] += dCx * dPfx / (dLx * dLy);
		sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      }
	  }
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dDVij, dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 -dDij / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
	      if (bCalcStress) {
		double dCx = 0.5*dDx2 - s2*nxA;
		double dCy = 0.5*dDy2 - s2*nyA;
		sData[thid] += dCx * dPfx / (dLx * dLy);
		sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
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
__global__ void heun_corr_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			     double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, double *pdR, 
			     double *pdA, double dKd, double *pdArea, double *pdMOI, double *pdIsoC, double *pdFx, 
			     double *pdFy, double *pdFt, double *pdTempX, double *pdTempY, double *pdTempPhi, 
			     double *pdXMoved, double *pdYMoved, double dEpsilon, int *bNewNbrs)
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

	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision; 
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (fabs(s1) == 1)
	      t1 = fmin(fmax( -(b*s1+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    else 
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    if (fabs(s2) == 1)
	      t2 = fmin(fmax( -(b*s2+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    else
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ > 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg > 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    double dPfx = dDx1 * dDVij / dDij;
	    double dPfy = dDy1 * dDVij / dDij;
	    dFx += dPfx;
	    dFy += dPfy;
	    //double dCx = s*nxA - 0.5*dDx;
	    //double dCy = s*nyA - 0.5*dDy;
	    double dCx = s1*nxA;
	    double dCy = s1*nyA;
	    dFt += dCx * dPfy - dCy * dPfx;
	  }
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dDVij;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 -dDij / dSigma) / dSigma;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		}
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
	    }     
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
void Spherocyl_Box::strain_step_2p(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

      switch (m_ePotential)
	{
	case HARMONIC:
	  euler_est_2p <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est_2p <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
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
	  euler_est_2p <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est_2p <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
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
      heun_corr_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, 
	 d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      heun_corr_2p <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
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
}




////////////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////////////
void Spherocyl_Box::run_strain_2p(double dStartGamma, double dStopGamma, double dSvStressGamma, double dSvPosGamma)
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
	strain_step_2p(nTime, 1, 1);
	fflush(m_pOutfSE);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  strain_step_2p(nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  strain_step_2p(nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  strain_step_2p(nTime, 1, 1);
  fflush(m_pOutfSE);
  fclose(m_pOutfSE);
}


void Spherocyl_Box::run_strain_2p(unsigned long int nStart, double dRunGamma, double dSvStressGamma, double dSvPosGamma)
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
	strain_step_2p(nTime, 1, 1);
	fflush(m_pOutfSE);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  strain_step_2p(nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  strain_step_2p(nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  strain_step_2p(nTime, 1, 1);
  fflush(m_pOutfSE);
  fclose(m_pOutfSE);
}

void Spherocyl_Box::run_strain_2p(long unsigned int nSteps)
{
  // Run strain for specified number of steps
  long unsigned int nTime = 0;
  while (nTime < nSteps)
    {
      strain_step_2p(nTime, 0, 0);
      nTime += 1;
    }

}


__global__ void resize_coords_2p(int nParticles, double dEpsilon, double *pdX, double *pdY)
{
	int nPID = threadIdx.x + blockIdx.x*blockDim.x;
	int nThreads = blockDim.x*gridDim.x;

	while (nPID < nParticles) {
		pdX[nPID] += dEpsilon*pdX[nPID];
		pdY[nPID] += dEpsilon*pdY[nPID];

		nPID += nThreads;
	}
}

__global__ void resize_y_coords_2p(int nParticles, double dEpsilon, double *pdY)
{
  int nPID = threadIdx.x + blockIdx.x*blockDim.x;
  int nThreads = blockDim.x*gridDim.x;

  while (nPID < nParticles) {
    pdY[nPID] += dEpsilon*pdY[nPID];

    nPID += nThreads;
  }
}


void Spherocyl_Box::resize_step_2p(long unsigned int tTime, double dEpsilon, bool bSvStress, bool bSvPos)
{
  if (dEpsilon > 0) {
    dEpsilon = 1./(1.-dEpsilon) - 1.;
  }

	resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, dEpsilon, d_pdX, d_pdY);
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
		  euler_est_2p <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est_2p <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
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
		  euler_est_2p <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est_2p <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
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
	      heun_corr_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
		 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
		 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
	      break;
	    case HERTZIAN:
	      heun_corr_2p <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd,d_pdArea, 
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

void Spherocyl_Box::y_resize_step_2p(long unsigned int tTime, double dEpsilon, bool bSvStress, bool bSvPos)
{
  if (dEpsilon > 0) {
    dEpsilon = 1./(1.-dEpsilon) - 1.;
  }

  resize_y_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, dEpsilon, d_pdY);
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
	  euler_est_2p <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
	     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est_2p <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
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
	  euler_est_2p <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
	     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  euler_est_2p <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
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
      heun_corr_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
	 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
	 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
	 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      heun_corr_2p <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
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

void Spherocyl_Box::resize_box_2p(long unsigned int nStart, double dEpsilon, double dFinalPacking, double dSvStressRate, double dSvPosRate)
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
		resize_step_2p(nTime, dEpsilon, bSaveStress, bSavePos);
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
	resize_step_2p(nTime, dEpsilon, 1, 1);
	fprintf(pOutfRs, "%.6g %.7g %.7g %.7g %.7g %.7g %.7g\n", m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
	fclose(m_pOutfSE);
	fclose(pOutfRs);

}



/////////////////////////////////////////
//
//  Relax to minimum
//
///////////////////////////////////////


void Spherocyl_Box::relax_step_2p(long unsigned int tTime, bool bSvStress, bool bSvPos)
{

	if (bSvStress)
	    {
	      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

	      switch (m_ePotential)
		{
		case HARMONIC:
		  euler_est_2p <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est_2p <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
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
		  euler_est_2p <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
		    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0, m_dStep, 
		     d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC,
		     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
		  break;
		case HERTZIAN:
		  euler_est_2p <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
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
	      heun_corr_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
		(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, 0,
		 m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, 
		 d_pdMOI, d_pdIsoC, d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, 
		 d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
	      break;
	    case HERTZIAN:
	      heun_corr_2p <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
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

void Spherocyl_Box::relax_box_2p(long unsigned int nSteps, double dMaxStep, double dMinStep, int nSaveStressInt, int nSavePosInt)
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
    relax_step_2p(nTime, bSaveStress, bSavePos);
    if (bSavePos) {
      fflush(stdout);
      fflush(m_pOutfSE);
    }
    nTime += 1;
    //if (nTime % nReorderInterval == 0)
    //reorder_particles();
  }
  relax_step_2p(nTime, 1, 1);
  fclose(m_pOutfSE);
  
}



//////////////////////////////////////////////////////////
//
//   Pure shear (compression y - expansion x)
//
///////////////////////////////////////////////////
template<Potential ePot, int bCalcStress>
__global__ void pure_euler_est_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
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

	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (fabs(s1) == 1)
	      t1 = fmin(fmax( -(b*s1+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    else 
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    if (fabs(s2) == 1)
	      t2 = fmin(fmax( -(b*s2+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    else
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ > 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg > 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    double dPfx = dDx1 * dDVij / dDij;
	    double dPfy = dDy1 * dDVij / dDij;
	    dFx += dPfx;
	    dFy += dPfy;
	    //  Find the point of contact (with respect to the center of the spherocyl)
	    //double dCx = s*nxA - 0.5*dDx;
	    //double dCy = s*nyA - 0.5*dDy; 
	    //double dCx = s*nxA;
	    //double dCy = s*nyA;
	    dFt += s1*nxA * dPfy - s1*nyA * dPfx;
	    if (bCalcStress)
	      {
		double dCx = 0.5*dDx1 - s1*nxA;
		double dCy = 0.5*dDy1 - s1*nyA;
		sData[thid] += dCx * dPfx / (dLx * dLy);
		sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      }
	  }
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dDVij, dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 -dDij / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
	      if (bCalcStress) {
		double dCx = 0.5*dDx2 - s2*nxA;
		double dCy = 0.5*dDy2 - s2*nyA;
		sData[thid] += dCx * dPfx / (dLx * dLy);
		sData[thid + offset] += dCy * dPfy / (dLx * dLy);
		sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
		sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
		sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
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

      pdTempX[nPID] = dX + dStep * (dFx + 0.5 * dStrain * dX);
      pdTempY[nPID] = dY + dStep * (dFy - 0.5 * dStrain * dY);
      double dRIso = -0.5 * pdIsoC[nPID] * sin(2*dPhi);
      pdTempPhi[nPID] = dPhi + dStep * (dFt + dStrain * dRIso);
      
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
__global__ void pure_heun_corr_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
				  double dStrain, double dStep, double *pdX, double *pdY, double *pdPhi, double *pdR, 
				  double *pdA, double dKd, double *pdArea, double *pdMOI, double *pdIsoC, double *pdFx, 
				  double *pdFy, double *pdFt, double *pdTempX, double *pdTempY, double *pdTempPhi, 
				  double *pdXMoved, double *pdYMoved, double dEpsilon, int *bNewNbrs)
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
	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision; 
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (fabs(s1) == 1)
	      t1 = fmin(fmax( -(b*s1+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    else 
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    if (fabs(s2) == 1)
	      t2 = fmin(fmax( -(b*s2+e)/c, -1.0), 1.0);
	    else if (b < 0)
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    else
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ > 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg > 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    double dPfx = dDx1 * dDVij / dDij;
	    double dPfy = dDy1 * dDVij / dDij;
	    dFx += dPfx;
	    dFy += dPfy;
	    //double dCx = s*nxA - 0.5*dDx;
	    //double dCy = s*nyA - 0.5*dDy;
	    double dCx = s1*nxA;
	    double dCy = s1*nyA;
	    dFt += dCx * dPfy - dCy * dPfx;
	  }
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dDVij;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 -dDij / dSigma) / dSigma;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		}
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
	    }     
	  }
	  
	}
      double dMOI = pdMOI[nPID];
      double dArea = pdArea[nPID];
      dFx /= (dKd*dArea);
      dFy /= (dKd*dArea);
      dFt = dFt / (dKd*dArea*dMOI); // - dStrain * 0.5 * dIsoC*sin(2*dPhi);

      double dFy0 = pdFy[nPID] / (dKd*dArea);
      double dFx0 = pdFx[nPID] / (dKd*dArea);
      //double dPhi0 = pdPhi[nPID];
      
      double dFt0 = pdFt[nPID] / (dKd*dArea*dMOI); // - dStrain * 0.5 * dIsoC*sin(2*dPhi0);

      double dDx = 0.5 * dStep * (dFx - dFx0);
      //double dDsx = 0.5 * dStep * dStrain * (dX + pdX[nPID]);
      double dDy = 0.5 * dStep * (dFy - dFy0);
      //double dDsy = -0.5 * dStep * dStrain * (dY + pdY[nPID]);
      double dDt = 0.5 * dStep * (dFt - dFt0);

      pdXMoved[nPID] += dX + dDx - pdX[nPID];
      pdYMoved[nPID] += dY + dDy - pdY[nPID];
      if (fabs(pdXMoved[nPID]) + fabs(pdYMoved[nPID]) > 0.5*dEpsilon)
	*bNewNbrs = 1;

      /*
      if (thid == 0) {
	printf("ID: %d, x0: %g, y0: %g, t0: %g, xt: %g, yt: %g, tt: %g\n", nPID, pdX[nPID], pdY[nPID], pdPhi[nPID], dX, dY, dPhi);
	printf("ID: %d, fx0: %g, fy0: %g, ft0: %g, fxt: %g, fyt: %g, ftt: %g\n", nPID, dFx0, dFy0, dFt0, dFx, dFy, dFt);
	printf("ID: %d, Dx: %g, Dy: %g, Dt: %g\n", nPID, dDx, dDy, dDt);
      }
      */

      pdX[nPID] = dX + dDx;
      pdY[nPID] = dY + dDy;
      pdPhi[nPID] = dPhi + dDt;

      nPID += nThreads;
    }
}


void Spherocyl_Box::pure_strain_step_2p(long unsigned int tTime, bool bSvStress, bool bSvPos)
{
  if (bSvStress)
    {
      cudaMemset((void *) d_pfSE, 0, 5*sizeof(float));

      switch (m_ePotential)
	{
	case HARMONIC:
	  pure_euler_est_2p <HARMONIC, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  pure_euler_est_2p <HERTZIAN, 1> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
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
	  pure_euler_est_2p <HARMONIC, 0> <<<m_nGridSize, m_nBlockSize>>>
	    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, m_dStrainRate,
	     m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	     d_pdFx, d_pdFy, d_pdFt, d_pfSE, d_pdTempX, d_pdTempY, d_pdTempPhi);
	  break;
	case HERTZIAN:
	  pure_euler_est_2p <HERTZIAN, 0> <<<m_nGridSize, m_nBlockSize>>>
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
      pure_heun_corr_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx*(1+0.5*m_dStep*m_dStrainRate), m_dLy*(1-0.5*m_dStep*m_dStrainRate), 
	 m_dGamma, m_dStrainRate, m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
      break;
    case HERTZIAN:
      pure_heun_corr_2p <HERTZIAN> <<<m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx*(1+0.5*m_dStep*m_dStrainRate), m_dLy*(1-0.5*m_dStep*m_dStrainRate),
	 m_dGamma, m_dStrainRate, m_dStep, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, m_dKd, d_pdArea, d_pdMOI, d_pdIsoC, 
	 d_pdFx, d_pdFy, d_pdFt, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdXMoved, d_pdYMoved, m_dEpsilon, d_bNewNbrs);
    }

  if (bSvStress)
    {
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      fprintf(m_pOutfSE, "%lu %.7g %.7g %.7g %.7g %.7g %.7g\n", 
	      tTime, *m_pfEnergy, *m_pfPxx, *m_pfPyy, m_fP, *m_pfPxy, *m_pfPyx);
      if (bSvPos)
	save_positions(tTime);
    }
  m_dLx += 0.5*m_dStep*m_dStrainRate*m_dLx;
  m_dLy -= 0.5*m_dStep*m_dStrainRate*m_dLy;

  cudaThreadSynchronize();
  checkCudaError("Updating estimates, moving particles");

  /*
  cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  for (int p=0; p < m_nSpherocyls; p++) {
    printf("%d: %g %g %g\n", p, h_pdX[p], h_pdY[p], h_pdPhi[p]);
  }
  */
  
  cudaMemcpy(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
  if (tTime % 100000 == 0) {
    *h_bNewNbrs = 1;
  }
  cudaThreadSynchronize();
  
  reconfigure_cells(0.0);
  m_dTotalGamma += m_dStep*m_dStrainRate;
  //cudaThreadSynchronize();

  if (*h_bNewNbrs)
    find_neighbors();
}


void Spherocyl_Box::run_pure_strain_2p(unsigned long int nStart, double dStopAspect, double dSvStressGamma, double dSvPosGamma)
{
  if (m_dStrainRate == 0.0)
    {
      fprintf(stderr, "Cannot strain with zero strain rate\n");
      exit(1);
    }
  else if (m_dStrainRate < 0.0)
    {
      m_dStrainRate = -m_dStrainRate;
    }
  
  double dAspect = m_dLx / m_dLy;
  if (dAspect > dStopAspect) {
    fprintf(stderr, "Cannot strain since starting aspect ratio is beyond ending aspect ratio\n");
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
  //unsigned long int nStop = (unsigned long)(dStopGamma / m_dStrainRate + 0.5);
  unsigned int nIntStep = (unsigned int)(1.0 / m_dStep + 0.5);
  unsigned int nSvStressInterval = (unsigned int)(dSvStressGamma / (m_dStrainRate * m_dStep) + 0.5);
  unsigned int nSvPosInterval = (unsigned int)(dSvPosGamma / m_dStrainRate + 0.5);
  unsigned long int nTotalStep = nTime * nIntStep;
  //unsigned int nReorderInterval = (unsigned int)(1.0 / m_dStrainRate + 0.5);
  
  printf("Strain run configured\n");
  printf("Time: %lu, Start: %g, Stop: %g, Int step: %lu\n", nTime, dAspect, dStopAspect, nIntStep);
  printf("Stress save int: %lu, Pos save int: %lu\n", nSvStressInterval, nSvPosInterval);
  fflush(stdout);

  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  if (nTime == 0)
    {
      m_dTotalGamma = 0;
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
  while (m_dLx / m_dLy < dStopAspect)
    {
      bool bSvPos = (nTime % nSvPosInterval == 0);
      if (bSvPos) {
	pure_strain_step_2p(nTime, 1, 1);
	fflush(m_pOutfSE);
      }
      else
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0);
	  pure_strain_step_2p(nTime, bSvStress, 0);
	}
      nTotalStep += 1;
      for (unsigned int nI = 1; nI < nIntStep; nI++)
	{
	  bool bSvStress = (nTotalStep % nSvStressInterval == 0); 
	  pure_strain_step_2p(nTime, bSvStress, 0);
	  nTotalStep += 1;
	}
      nTime += 1;
      //printf("Time: %lu, Gamma: %g, Aspect: %g\n", nTime, m_dTotalGamma, m_dLx/m_dLy);
      //if (nTime % nReorderInterval == 0)
      //reorder_particles();
    }
  
  // Save final configuration
  strain_step_2p(nTime, 1, 1);
  fflush(m_pOutfSE);
  fclose(m_pOutfSE);
}



///////////////////////////////////////////////
////////////////////////////////////////////////
//                                             //
//   Quasistatic transformations            //
//                                       //
////////////////////////////////////////
template<Potential ePot>
__global__ void calc_se_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, 
			   double dGamma, double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, 
			   double *pdFx, double *pdFy, double *pdFt, double *pdEnergies, float *pfSE)
{
  // Declare shared memory pointer, the size is determined at the kernel launch
  extern __shared__ double sData[];
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  int offset = blockDim.x + 8; // +8 helps to avoid bank conflicts
  for (int i = 0; i < 5; i++)
    sData[i*offset + thid] = 0.0;
  __syncthreads();  // synchronizes every thread in the block before going on
  
  while (nPID < nSpherocyls)
    {
      double dFx = 0.0;
      double dFy = 0.0;
      double dFt = 0.0;
      double dEnergy = 0.0;
      
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
	  
	  double a = dA * dA;
	  double b = -(nxA * nxB + nyA * nyB);
	  double c = dB * dB;
	  double d = nxA * dDeltaX + nyA * dDeltaY;
	  double e = -nxB * dDeltaX - nyB * dDeltaY;
	  double delta = a * c - b * b;
	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (b < 0) {
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    }
	    else {
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    }
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ > 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg > 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	
	  // Check if they overlap and calculate forces
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
		  dDVij = (1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDij / dSigma) * sqrt(1.0 - dDij / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDx1 * dDVij / dDij;
	      double dPfy = dDy1 * dDVij / dDij;
	      double dCx = 0.5 * dDx1 - s1*nxA;
	      double dCy = 0.5 * dDy1 - s1*nyA;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s1*nxA * dPfy - s1*nyA * dPfx;

	      sData[thid] += dCx * dPfx / (dLx * dLy);
	      sData[thid + offset] += dCy * dPfy / (dLx * dLy);
	      sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
	      sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
	      sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      dEnergy += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	    }
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dAlpha, dDVij;
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
	      //printf("Found second contact: %d %d %f %f %f %f %g %g %.8g %.8g %g %g\n", nPID, nAdjPID, s2, t2, s1, t1, dPhi, dPhiB, dDij1, dDij, dDVij1, dDVij);
	      
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
	      
	      double dCx = 0.5*dDx2 - s2*nxA;
	      double dCy = 0.5*dDy2 - s2*nyA;
	      sData[thid] += dCx * dPfx / (dLx * dLy);
	      sData[thid + offset] += dCy * dPfy / (dLx * dLy);
	      sData[thid + 2*offset] += dCx * dPfy / (dLx * dLy);
	      sData[thid + 3*offset] += dCy * dPfx / (dLx * dLy);
	      sData[thid + 4*offset] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      dEnergy += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	      
	    }
	  }

	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      pdFt[nPID] = dFt;
      pdEnergies[nPID] = dEnergy;
      
      nPID += nThreads;
    }
  __syncthreads();
  
  /*
  double dEnergyCheck = 0;
  if (thid == 0) {
    //double dEnergyCheck = 0;
    for (int i = 0; i < blockDim.x; i++) {
      dEnergyCheck += sData[4*offset + i];
    }
    }
  __syncthreads();
  */

  // Now we do a parallel reduction sum to find the total number of contacts
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
	      //if (thid == 0) {
	      //printf("Energy Check (blid %d): %.8g %.8g\n", blockIdx.x, dEnergyCheck, sData[4*offset]);
	      //}
	    }
	}
    } 
}


void Spherocyl_Box::calculate_stress_energy_2p()
{
  cudaMemset((void*) d_pfSE, 0, 5*sizeof(float));
  
  //dim3 grid(m_nGridSize);
  //dim3 block(m_nBlockSize);
  //size_t smem = m_nSM_CalcSE;
  //printf("Configuration: %d x %d x %d\n", m_nGridSize, m_nBlockSize, m_nSM_CalcSE);

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
	 d_pdPhi, d_pdR, d_pdA, d_pdFx, d_pdFy, d_pdFt, d_pdEnergies, d_pfSE);
      break;
    case HERTZIAN:
      calc_se_2p <HERTZIAN> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
	 d_pdPhi, d_pdR, d_pdA, d_pdFx, d_pdFy, d_pdFt, d_pdEnergies, d_pfSE);
    }
  cudaThreadSynchronize();
  checkCudaError("Calculating stresses and energy");
}




template<Potential ePot>
__global__ void calc_fe_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, 
			   double dGamma, double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, 
			   double *pdMOI, double *pdFx, double *pdFy, double *pdFt, double *pdEnergyBlocks)
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

	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (b < 0) {
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    }
	    else {
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    }
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ >= 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg >= 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    double dPfx = dDx1 * dDVij / dDij;
	    double dPfy = dDy1 * dDVij / dDij;
	    dFx += dPfx;
	    dFy += dPfy;
	    //  Find the point of contact (with respect to the center of the spherocyl)
	    //double dCx = s*nxA - 0.5*dDx;
	    //double dCy = s*nyA - 0.5*dDy; 
	    //double dCx = s*nxA;
	    //double dCy = s*nyA;
	    dFt += s1*nxA * dPfy - s1*nyA * dPfx;
	    sData[thid] += 0.5 * dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dLx * dLy);
	  }
	    
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dDVij, dAlpha;
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
	      double dPfx = dDx2 * dDVij / dDij;
	      double dPfy = dDy2 * dDVij / dDij;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s2*nxA * dPfy - s2*nyA * dPfx;
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

    /*
    double dEnergyCheck = 0;
    if (thid == 0) {
      //double dEnergyCheck = 0;
      for (int i = 0; i < blockDim.x; i++) {
	dEnergyCheck += sData[i];
      }
    }
    __syncthreads();
    */
    
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
		//printf("Energy Check (blid: %d): %.8g %.8g\n", blockIdx.x, dEnergyCheck, sData[0]);
	      }
	    }
	  }
	}  
      }
    }
}

template<Potential ePot>
__global__ void calc_energy_2p(int nSpherocyls, int *pnNPP, int *pnNbrList, double dLx, double dLy, double dGamma, 
			       double *pdX, double *pdY, double *pdPhi, double *pdR, double *pdA, double *pdEnergyBlocks)
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

	  double s1, t1, s2, t2, s3, t3;
	  if (delta <= 0) {  //delta should never be negative but I'm using <= in case it comes out negative due to precision
	    s1 = fmin(fmax( -(b+d)/a, -1.0), 1.0);
	    s2 = fmin(fmax( -(d-b)/a, -1.0), 1.0);
	    if (b < 0) {
	      t1 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	      t2 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	    }
	    else {
	      t1 = fmin(fmax( -(e-b)/c, -1.0), 1.0);
	      t2 = fmin(fmax( -(b+e)/c, -1.0), 1.0);
	    }
	    s3 = s2;
	    t3 = t2;
	  }
	  else {
	    t1 = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	    s1 = -(b*t1+d)/a;
	    double sarg = fabs(s1);
	    s1 = fmin( fmax(s1,-1.), 1. );
	    if (sarg >= 1) {
	      t1 = fmin( fmax( -(b*s1+e)/c, -1.), 1.);
	      s2 = -s1;
	      t2 = -(b*s2+e)/c;
	      double targ = fabs(t2);
	      t2 = fmin( fmax(t2, -1.), 1.);
	      if (targ >= 1) {
		s2 = fmin( fmax( -(b*t2+d)/a, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  t3 = -s1;
		else
		  t3 = s1;
		s3 = -(b*t3+d)/a;
		sarg = fabs(s3);
		s3 = fmin(fmax(s3, -1.0), 1.0);
		if (sarg >= 1)
		  t3 = fmin(fmax(-(b*s3+e)/c, -1.0), 1.0);
	      }
	    }
	    else {
	      t2 = -t1;
	      s2 = -(b*t2+d)/a;
	      sarg = fabs(s2);
	      s2 = fmin( fmax(s2, -1.), 1.);
	      if (sarg >= 1) {
		t2 = fmin( fmax( -(b*s2+e)/c, -1.), 1.);
		s3 = s2;
		t3 = t2;
	      }
	      else {
		if (b < 0)
		  s3 = -t1;
		else
		  s3 = t1;
		
		t3 = -(b*s3+e)/c;
		double targ = fabs(t3);
		t3 = fmin(fmax(t3, -1.0), 1.0);
		if (targ >= 1)
		  s3 = min(max(-(b*t3+d)/a, -1.0), 1.0);
	      } 
	    }
	  }
	  
	  // Check if they overlap and calculate forces
	  double dDx1 = dDeltaX + s1*nxA - t1*nxB;
	  double dDy1 = dDeltaY + s1*nyA - t1*nyB;
	  double dDSqr1 = dDx1 * dDx1 + dDy1 * dDy1;
	  if (dDSqr1 < dSigma*dSigma) {
	    double dDij = sqrt(dDSqr1);
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
	    
	  if (s1 != s2 && t1 != t2) {
	    double dDx2 = dDeltaX + s2*nxA - t2*nxB;
	    double dDy2 = dDeltaY + s2*nyA - t2*nyB;
	    double dDSqr2 = dDx2 * dDx2 + dDy2 * dDy2;
	    if (s3 != s1 && s3 != s2) {
	      double dDx3 = dDeltaX + s3*nxA - t3*nxB;
	      double dDy3 = dDeltaY + s3*nyA - t3*nyB;
	      double dDSqr3 = dDx3 * dDx3 + dDy3 * dDy3;
	      if (dDSqr3 < dDSqr2) {
		s2 = s3;
		t2 = t3;
		dDx2 = dDx3;
		dDy2 = dDy3;
		dDSqr2 = dDSqr3;
	      }
	    }
	    if (dDSqr2 < dSigma*dSigma) {
	      double dDij = sqrt(dDSqr2);
	      double dAlpha, dDVij;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 -dDij / dSigma) / dSigma;
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

    /*
    double dEnergyCheck = 0;
    if (thid == 0) {
      //double dEnergyCheck = 0;
      for (int i = 0; i < blockDim.x; i++) {
	dEnergyCheck += sData[i];
      }
    }
    __syncthreads();
    */

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
		//printf("Energy Check (blid: %d): %.8g %.8g\n", blockIdx.x, dEnergyCheck, sData[0]); 
	      }
	    }
	  }
	}  
      }
    }
}

__global__ void square_forces_2p(int nDim, double *pdFx, double *pdFy, double *pdFt, double *pdMOI, float *pfSquare)
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

__global__ void mult_forces_2p(int nDim, double *pdFx0, double *pdFy0, double *pdFt0, double *pdFx, 
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

__global__ void sqrt_grad_dir_2p(int nSpherocyls, double *pdFx, double *pdFy, double *pdFt,
				 double *pdDx, double *pdDy, double *pdDt)
{
  int thid = threadIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  int nPID = thid + (blockIdx.x / 3) * blockDim.x;
  
  if (blockIdx.x % 3 == 0) {
    while (nPID < nSpherocyls) {
      double dFx = pdFx[nPID];
      double dFy = pdFy[nPID];
      double dFmag = sqrt(dFx*dFx + dFy*dFy);
      if (dFmag == 0) {
	pdDx[nPID] = 0;
      }
      else {
	pdDx[nPID] = dFx / sqrt(dFmag);
      }
      //printf("%d %g %g %g\n", nPID, dFx, dFmag, pdDx[nPID]);
      nPID += nThreads;
    }
  }
  else if (blockIdx.x % 3 == 1) {
     while (nPID < nSpherocyls) {
      double dFx = pdFx[nPID];
      double dFy = pdFy[nPID];
      double dFmag = sqrt(dFx*dFx + dFy*dFy);
      if (dFmag == 0) {
	pdDy[nPID] = 0;
      }
      else {
	pdDy[nPID] = dFy / sqrt(dFmag);
      }
      //printf("%d %g %g %g\n", nPID, dFy, dFmag, pdDy[nPID]);
      nPID += nThreads;
    }
  }
  else {
    while (nPID < nSpherocyls) {
      double dFt = pdFt[nPID];
      double dFmag = fabs(dFt);
      if (dFmag == 0) {
	pdDt[nPID] = 0;
      }
      else {
	pdDt[nPID] = dFt / sqrt(dFmag);
      }
      //printf("%d %g %g %g\n", nPID, dFt, dFmag, pdDt[nPID]);
      nPID += nThreads;
    }
  }
}

__global__ void new_pr_conj_dir_2p(int nSpherocyls, double *pdFx, double *pdFy, double *pdFt, 
				double *pdMOI, double *pdDx, double *pdDy, double *pdDt, 
				float *pfF0Sq, float *pfFSq, float *pfFF0)
{
  int thid = threadIdx.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  int nPID = thid + (blockIdx.x / 3) * blockDim.x;
  float fBeta = fmax(0., (*pfFSq - *pfFF0) / (*pfF0Sq));

  if (blockIdx.x % 3 == 0) {
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
      pdDt[nPID] = dNewDt;
      nPID += nThreads;
    }
  }

}


__global__ void temp_move_step_2p(int nSpherocyls, double *pdX, double *pdY, double *pdPhi, 
			       double *pdTempX, double *pdTempY, double *pdTempPhi, 
			       double *pdDx, double *pdDy, double *pdDt, double dStep)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nPID = thid + (blid / 3) * blockDim.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  
  if (blid % 3 == 0) {
    while (nPID < nSpherocyls) {
      pdTempX[nPID] = pdX[nPID] + dStep * pdDx[nPID];
      nPID += nThreads;
    }
  }
  else if (blid % 3 == 1) {
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

__global__ void move_step_2p(int nSpherocyls, double *pdX, double *pdY, double *pdPhi,
			     double *pdDx, double *pdDy, double *pdDt, double dStep, 
			     double *pdXMoved, double *pdYMoved, int *bNewNbrs, double dDR)
{
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int nPID = thid + (blid / 3) * blockDim.x;
  int nThreads = blockDim.x * gridDim.x / 3;
  
  if (blid % 3 == 0) {
    while (nPID < nSpherocyls) {
      pdX[nPID] = pdX[nPID] + dStep * pdDx[nPID];
      pdXMoved[nPID] += dStep * pdDx[nPID];
      if (fabs(pdXMoved[nPID]) > 0.5*dDR)
	  *bNewNbrs = 1;
      nPID += nThreads;
    }
  }
  else if (blid % 3 == 1) {
    while (nPID < nSpherocyls) {
      pdY[nPID] = pdY[nPID] + dStep * pdDy[nPID];
      pdYMoved[nPID] += dStep * pdDy[nPID];
      if (fabs(pdYMoved[nPID]) > 0.5*dDR)
	  *bNewNbrs = 1;
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

int Spherocyl_Box::new_line_search_2p(double dMinStep, double dMaxStep)
{
  cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
 
  temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>>  
    (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
     d_pdDx, d_pdDy, d_pdDt, m_dStep);
  cudaDeviceSynchronize();
  checkCudaError("Moving Step 1");

  calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating energy");
  
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
  h_pdLineEnergy[1] = 0;
  cudaDeviceSynchronize();
  for(int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[1] += h_pdBlockSums[j];
  }

  if (h_pdLineEnergy[1] > h_pdLineEnergy[0]) {
    while (h_pdLineEnergy[1] >= h_pdLineEnergy[0]) {
      m_dStep /= 2;
      if (m_dStep < dMinStep) {
	printf("Line search stopped due to step size %g %g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1]);
	return 2;
      }
      h_pdLineEnergy[2] = h_pdLineEnergy[1];
      
      cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
 
      temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>>  
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
	 d_pdDx, d_pdDy, d_pdDt, m_dStep);
      cudaDeviceSynchronize();
      checkCudaError("Moving Step 1");
      
      calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
      cudaDeviceSynchronize();
      checkCudaError("Calculating energy");
      
      cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
      h_pdLineEnergy[1] = 0;
      cudaDeviceSynchronize();
      for(int j = 0; j < m_nGridSize; j++) {
	h_pdLineEnergy[1] += h_pdBlockSums[j];
      }

      //printf("Step size %g , line energies: %.10g %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
    }
  }
  else {
    cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
    
    temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>>  
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, 2*m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    h_pdLineEnergy[2] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[2] += h_pdBlockSums[j];
    }
    //printf("Step size %g , line energies: %.10g %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
    
    while (h_pdLineEnergy[2] <= h_pdLineEnergy[1]) {
      m_dStep *= 2;
      h_pdLineEnergy[1] = h_pdLineEnergy[2];
      
      cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
    
      temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>>  
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
	 d_pdDx, d_pdDy, d_pdDt, 2*m_dStep);
      cudaDeviceSynchronize();
      checkCudaError("Moving Step 2");
      
      calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
      cudaDeviceSynchronize();
      checkCudaError("Calculating energy");
      
      cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
      h_pdLineEnergy[2] = 0;
      cudaDeviceSynchronize();
      for(int j = 0; j < m_nGridSize; j++) {
	h_pdLineEnergy[2] += h_pdBlockSums[j];
      }
      //printf("Step size %g , line energies: %.10g %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);    
      if (m_dStep > dMaxStep) {
	break;
      }
    }
  }

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

    move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, dLineMin*m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
    cudaDeviceSynchronize();
    checkCudaError("Moving Spherocyls");

    calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    h_pdLineEnergy[5] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[5] += h_pdBlockSums[j];
    }
    
    if (h_pdLineEnergy[5] > h_pdLineEnergy[1]) {
      move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, (1.-dLineMin)*m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
      cudaDeviceSynchronize();
      checkCudaError("Moving Spherocyls");
      dLineMin = 1;
    }

    printf("Line search result, step %g, line energies %.12g %.12g %.12g, line min %g\n", 
	   m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2], dLineMin);
    
    
    cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if (*h_bNewNbrs) {
      find_neighbors();
    }
  }

  return 0;
}

int Spherocyl_Box::line_search_2p(bool bFirstStep, bool bSecondStep, double dMinStep, double dMaxStep)
{
  bool bFindMin = true;
  bool bMinStep = false;
  if (bFirstStep) {
    cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
 
    temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>>  
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
    cudaDeviceSynchronize();
    checkCudaError("Calculating energy");
    
    cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    h_pdLineEnergy[1] = 0;
    cudaDeviceSynchronize();
    for(int j = 0; j < m_nGridSize; j++) {
      h_pdLineEnergy[1] += h_pdBlockSums[j];
    }
    //printf("Step Energy: %.10g\n", dEnergy); 
    //printf("First step %g, line energies: %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1]); 
    if (h_pdLineEnergy[1] > h_pdLineEnergy[0]) {   
      if (m_dStep > dMinStep) {
	m_dStep /= 2;
	h_pdLineEnergy[2] = h_pdLineEnergy[1];
	h_pdLineEnergy[1] = 0;
	//cudaMemcpyAsync(d_pfSE+1, h_pfSE+1, 2*sizeof(float), cudaMemcpyHostToDevice);
	int ret = line_search_2p(1,0, dMinStep, dMaxStep);
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

    temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
       d_pdDx, d_pdDy, d_pdDt, 2*m_dStep);
    cudaDeviceSynchronize();
    checkCudaError("Moving Step 1");
    
    calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
      (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
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
    //printf("Second step %g , line energies: %.10g %.10g %.10g\n", m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2]);
    if (h_pdLineEnergy[2] <= h_pdLineEnergy[1]) {
      
      if (m_dStep < dMaxStep) {
	m_dStep *= 2;
	h_pdLineEnergy[1] = h_pdLineEnergy[2];
	h_pdLineEnergy[2] = 0;
	//cudaMemcpyAsync(d_pfSE+1, h_pfSE+1, 2*sizeof(float), cudaMemcpyHostToDevice);
	int ret = line_search_2p(0,1, dMinStep, dMaxStep);
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
      cudaMemset((void*) d_pdBlockSums, 0, m_nGridSize*sizeof(double));

      temp_move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdTempX, d_pdTempY, d_pdTempPhi,
	 d_pdDx, d_pdDy, d_pdDt, dLineMin*m_dStep);
      cudaDeviceSynchronize();
      checkCudaError("Moving Step 1");
      
      calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdTempX, d_pdTempY, d_pdTempPhi, d_pdR, d_pdA, d_pdBlockSums);
      cudaDeviceSynchronize();
      checkCudaError("Calculating energy");
      cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_pfSE+2, d_pfSE+2, sizeof(float), cudaMemcpyDeviceToHost);
      h_pdLineEnergy[2] = 0;
      cudaDeviceSynchronize();
      for(int j = 0; j < m_nGridSize; j++) {
	h_pdLineEnergy[2] += h_pdBlockSums[j];
      }
      if (h_pdLineEnergy[2] > h_pdLineEnergy[1]) {
	dLineMin = 1;
      }
      move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, dLineMin*m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
      cudaDeviceSynchronize();
      checkCudaError("Moving Spherocyls");
      printf("Line search result, step %g, line energies %.14g %.14g %.14g, line min %g\n", 
      	     m_dStep, h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[2], dLineMin);
      

      cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      if (*h_bNewNbrs) {
	find_neighbors();
      }
      
      /*
      calc_energy_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, d_pdPhi, d_pdR, d_pdA, d_pdBlockSums);
      cudaDeviceSynchronize();
      checkCudaError("Calculating energy");
    
      cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, sizeof(double)*m_nGridSize, cudaMemcpyDeviceToHost);
      h_pdLineEnergy[4] = 0;
      cudaDeviceSynchronize();
      for(int j = 0; j < m_nGridSize; j++) {
	h_pdLineEnergy[4] += h_pdBlockSums[j];
      }
      
      if (h_pdLineEnergy[4] > h_pdLineEnergy[2]) {
	move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
	  (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, (1.-dLineMin)*m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
	cudaDeviceSynchronize();
	checkCudaError("Moving Spherocyls");
      }
      */
    }
  }
  
  cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  if (*h_bNewNbrs) {
    find_neighbors();
  }

  if (m_dStep < dMinStep)
    return 2;
  return 0;
}

int Spherocyl_Box::gradient_descent_step_2p(double dMinStep, double dMaxStep)
{
  //float *d_pfF0Square;
  float *d_pfFSquare;
  //cudaMalloc((void **) &d_pfF0Square, sizeof(float));
  cudaMalloc((void **) &d_pfFSquare, sizeof(float));
  //cudaMemset((void *) d_pfF0Square, 0, sizeof(float));
  cudaMemset((void *) d_pfFSquare, 0, sizeof(float));
  cudaMemset((void *) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
  checkCudaError("Setting memory");

  //cudaMemcpyAsync(d_pdTempFx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  //cudaMemcpyAsync(d_pdTempFy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  //cudaMemcpyAsync(d_pdTempFt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  //cudaDeviceSynchronize();

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums); 
  cudaDeviceSynchronize();
  checkCudaError("Calculating gradient direction");

  //square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
  //  (m_nSpherocyls, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfF0Square);
  //cudaDeviceSynchronize();
  //checkCudaError("Calculating square of forces 1");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice); 

  square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfFSquare);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces squared");

  float *h_pfFSquare;
  //h_pfF0Square = (float*) malloc(sizeof(float));
  h_pfFSquare = (float*) malloc(sizeof(float));
  //cudaMemcpy(h_pfF0Square, d_pfF0Square, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  if (*h_pfFSquare == 0) {
    printf("Zero energy found, stopping minimization\n");
    return 1;
  }

  double dStepNorm = sqrt(*h_pfFSquare);
  m_dStep = 1e-3 / dStepNorm;
  dMinStep = 1e-13 / dStepNorm;
  dMaxStep = 100 / dStepNorm;

  h_pdLineEnergy[5] = h_pdLineEnergy[4];
  h_pdLineEnergy[4] = h_pdLineEnergy[3];
  h_pdLineEnergy[3] = h_pdLineEnergy[0];
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  
  if (h_pdLineEnergy[3] <= h_pdLineEnergy[0] && h_pdLineEnergy[4] <= h_pdLineEnergy[3] && h_pdLineEnergy[5] <= h_pdLineEnergy[4]) {
    printf("Minimum Energy Found (%g %g %g %g), stopping minimization\n", h_pdLineEnergy[0], h_pdLineEnergy[3], h_pdLineEnergy[4], h_pdLineEnergy[5]);
    return 2;
  }

  
  int ret = new_line_search_2p(dMinStep, dMaxStep);
  printf("Line Search returned %d\n", ret);
  if (ret == 2) {
    m_dStep = sqrt(h_pdLineEnergy[0]) / dStepNorm;
    printf("Moving gradient step: %g %g\n", m_dStep, dStepNorm);
    move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
    cudaDeviceSynchronize();
    checkCudaError("Moving Spherocyls");
    
    cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if (*h_bNewNbrs) {
      find_neighbors();
    }
  }
  else if ((h_pdLineEnergy[0] - h_pdLineEnergy[1]) < (h_pdLineEnergy[0] * 1e-13)) {
    m_dStep = sqrt(h_pdLineEnergy[0]) / dStepNorm;
    printf("Small delta-E detected: %g %g %g\n", h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[0] - h_pdLineEnergy[1]);
    printf("Moving gradient step: %g %g\n", m_dStep, *h_pfFSquare);
    move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
      (m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
    cudaDeviceSynchronize();
    checkCudaError("Moving Spherocyls");
    
    cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if (*h_bNewNbrs) {
      find_neighbors();
    }
  }

  free(h_pfFSquare);
  cudaFree(d_pfFSquare);

  return 0;
}

int Spherocyl_Box::cjpr_relax_step_2p(double dMinStep, double dMaxStep)
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
  checkCudaError("Setting memory");

  cudaMemcpyAsync(d_pdTempFx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  
  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums); 

  square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfF0Square);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces 1");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);

  square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfFSquare);
  mult_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfFF0);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  
  float *h_pfF0Square;
  float *h_pfFSquare;
  h_pfF0Square = (float*) malloc(sizeof(float));
  h_pfFSquare = (float*) malloc(sizeof(float));
  cudaMemcpy(h_pfF0Square, d_pfF0Square, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  if (*h_pfFSquare == 0) {
    printf("Zero energy found, stopping minimization\n");
    return 1;
  }
  cudaDeviceSynchronize();
  h_pdLineEnergy[4] = h_pdLineEnergy[3];
  h_pdLineEnergy[3] = h_pdLineEnergy[0];
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  if (h_pdLineEnergy[3] == h_pdLineEnergy[0] && h_pdLineEnergy[4] == h_pdLineEnergy[3]) {
    printf("Minimum Energy Found, stopping minimization\n");
    return 2;
  }

  new_pr_conj_dir_2p <<<3*m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdDx, d_pdDy, d_pdDt, 
     d_pfF0Square, d_pfFSquare, d_pfFF0);
  cudaDeviceSynchronize();
  checkCudaError("Finding new conjugate direction");

  double dStepNorm = sqrt(*h_pfFSquare/(3*m_nSpherocyls));
  m_dStep = 2e-6 / dStepNorm;
  dMinStep = 1e-18 / dStepNorm;
  dMaxStep = 2 / dStepNorm;
  printf("Starting Line search with step size: %g %g %g %g\n", m_dStep, *h_pfFSquare, dMinStep, dMaxStep);
  int ret = line_search_2p(1,1, dMinStep, dMaxStep);
  if (ret == 1) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 1e-6 / dStepNorm;
    printf("Starting Line search with step size: %g\n", m_dStep);
    cudaDeviceSynchronize();
    ret = line_search_2p(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      sqrt_grad_dir_2p <<<3*m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdDx, d_pdDy, d_pdDt);
      cudaDeviceSynchronize();
      //cudaMemcpy(h_pdFx, d_pdDx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h_pdFy, d_pdDy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h_pdFt, d_pdDt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      m_dStep = 2e-6 / sqrt(dStepNorm);
      dMinStep = 1e-18 / sqrt(dStepNorm);
      dMaxStep = 2 / sqrt(dStepNorm);
      //printf("Starting Line search with step size: %g\n", m_dStep);
      //printf("Step: %g, Dir: %g %g %g, Move: %g %g %g\n", 
      //	     m_dStep, h_pdFx[0], h_pdFy[0], h_pdFt[0], m_dStep*h_pdFx[0], m_dStep*h_pdFy[0], m_dStep*h_pdFt[0]);
      //printf("Step: %g, Dir: %g %g %g, Move: %g %g %g\n", 
      //	     m_dStep, h_pdFx[1], h_pdFy[1], h_pdFt[1], m_dStep*h_pdFx[1], m_dStep*h_pdFy[1], m_dStep*h_pdFt[1]);
      ret = line_search_2p(1,1, dMinStep, dMaxStep);
      if (ret == 1) {
	printf("Stopping due to step size\n");
	return 1;
      }
    }
  }
  else if (ret == 2) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    m_dStep = 1e-6 / dStepNorm;
    printf("Starting Line search with step size: %g\n", m_dStep);
    cudaDeviceSynchronize();
    ret = line_search_2p(1,1, dMinStep, dMaxStep);
    if (ret == 1) {
      sqrt_grad_dir_2p <<<3*m_nGridSize, m_nBlockSize>>>
	(m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdDx, d_pdDy, d_pdDt);
      cudaDeviceSynchronize();
      m_dStep = 1e-6 / sqrt(dStepNorm);
      dMinStep = 1e-18 / sqrt(dStepNorm);
      dMaxStep = 0.01 / sqrt(dStepNorm);
      printf("Starting Line search with step size: %g\n", m_dStep);
      //printf("Step: %g, Dir: %g %g %g, Move: %g %g %g\n", 
      //       m_dStep, h_pdFx[0], h_pdFy[0], h_pdFt[0], m_dStep*h_pdFx[0], m_dStep*h_pdFy[0], m_dStep*h_pdFt[0]);
      //printf("Step: %g, Dir: %g %g %g, Move: %g %g %g\n", 
      //       m_dStep, h_pdFx[1], h_pdFy[1], h_pdFt[1], m_dStep*h_pdFx[1], m_dStep*h_pdFy[1], m_dStep*h_pdFt[1]);
      ret = line_search_2p(1,1, dMinStep, dMaxStep);
      if (ret == 1) {
	printf("Stopping due to step size\n");
	return 1;
      }
    }
  }

  free(h_pfF0Square); free(h_pfFSquare);
  cudaFree(d_pfF0Square); cudaFree(d_pfFSquare); cudaFree(d_pfFF0);

  return 0;
}

int Spherocyl_Box::new_cjpr_relax_step_2p(double dMinStep, double dMaxStep)
{
  float *d_pfF0Square;
  float *d_pfFSquare;
  float *d_pfFF0;
  float *d_pfDSquare;
  cudaMalloc((void **) &d_pfF0Square, sizeof(float));
  cudaMalloc((void **) &d_pfFSquare, sizeof(float));
  cudaMalloc((void **) &d_pfFF0, sizeof(float));
  cudaMalloc((void **) &d_pfDSquare, sizeof(float));
  cudaMemset((void *) d_pfF0Square, 0, sizeof(float));
  cudaMemset((void *) d_pfFSquare, 0, sizeof(float));
  cudaMemset((void *) d_pfFF0, 0, sizeof(float));
  cudaMemset((void *) d_pfDSquare, 0, sizeof(float));
  cudaMemset((void *) d_pdBlockSums, 0, m_nGridSize*sizeof(double));
  checkCudaError("Setting memory");

  cudaMemcpyAsync(d_pdTempFx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(d_pdTempFt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  
  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums); 

  square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfF0Square);
  cudaDeviceSynchronize();
  checkCudaError("Calculating square of forces 1");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);

  square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pfFSquare);
  mult_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdTempFx, d_pdTempFy, d_pdTempFt, d_pdMOI, d_pfFF0);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  
  
  float *h_pfF0Square;
  float *h_pfFSquare;
  h_pfF0Square = (float*) malloc(sizeof(float));
  h_pfFSquare = (float*) malloc(sizeof(float));
  cudaMemcpy(h_pfF0Square, d_pfF0Square, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pfFSquare, d_pfFSquare, sizeof(float), cudaMemcpyDeviceToHost);
  if (*h_pfFSquare == 0) {
    printf("Zero energy found, stopping minimization\n");
    return 1;
  }
  cudaDeviceSynchronize();

  new_pr_conj_dir_2p <<<3*m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, d_pdFx, d_pdFy, d_pdFt, d_pdMOI, d_pdDx, d_pdDy, d_pdDt, 
     d_pfF0Square, d_pfFSquare, d_pfFF0);
  h_pdLineEnergy[5] = h_pdLineEnergy[4];
  h_pdLineEnergy[4] = h_pdLineEnergy[3];
  h_pdLineEnergy[3] = h_pdLineEnergy[0];
  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  cudaDeviceSynchronize();
  checkCudaError("Finding new conjugate direction");
  
  square_forces_2p <<<3*m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pdDx, d_pdDy, d_pdDt, d_pdMOI, d_pfDSquare);
  float *h_pfDSquare;
  h_pfDSquare = (float*) malloc(sizeof(float));

  if (h_pdLineEnergy[3] == h_pdLineEnergy[0] && h_pdLineEnergy[4] == h_pdLineEnergy[3] && h_pdLineEnergy[5] <= h_pdLineEnergy[4]) {
    printf("Minimum Energy Found, stopping minimization\n");
    return 2;
  }
  cudaDeviceSynchronize();
  cudaMemcpy(h_pfDSquare, d_pfDSquare, sizeof(float), cudaMemcpyDeviceToHost);

  double dStepNorm = sqrt(*h_pfDSquare);
  m_dStep = 1;
  dMinStep = 1e-13 / dStepNorm;
  dMaxStep = 10 / dStepNorm;
  printf("Starting Line search with step size: %g %g %g %g\n", m_dStep, dStepNorm, dMinStep, dMaxStep);
  int ret = new_line_search_2p(dMinStep, dMaxStep);
  if (ret == 2) {
    cudaMemcpyAsync(d_pdDx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_pdDt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToDevice);
    dStepNorm = sqrt(*h_pfFSquare);
    m_dStep = 1;
    dMinStep = 1e-13 / dStepNorm;
    dMaxStep = 10 / dStepNorm;
    printf("Starting Line search with step size: %g %g %g %g\n", m_dStep, dStepNorm, dMinStep, dMaxStep);
    cudaDeviceSynchronize();
    ret = new_line_search_2p(dMinStep, dMaxStep);
    if (ret == 2) {
      m_dStep = 0.1;
      printf("Moving gradient step: %g %g\n", m_dStep, dStepNorm);
      move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
      cudaDeviceSynchronize();
      checkCudaError("Moving Spherocyls");
    
      cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      if (*h_bNewNbrs) {
	find_neighbors();
      }
    }
    else if ((h_pdLineEnergy[0] - h_pdLineEnergy[1]) < (h_pdLineEnergy[0] * 1e-13)) {
      m_dStep = 0.1;
      printf("Small delta-E detected: %g %g %g\n", h_pdLineEnergy[0], h_pdLineEnergy[1], h_pdLineEnergy[0] - h_pdLineEnergy[1]);
      printf("Moving gradient step: %g %g\n", m_dStep, *h_pfFSquare);
      move_step_2p <<<3*m_nGridSize, m_nBlockSize>>> 
	(m_nSpherocyls, d_pdX, d_pdY, d_pdPhi, d_pdDx, d_pdDy, d_pdDt, m_dStep, d_pdXMoved, d_pdYMoved, d_bNewNbrs, m_dEpsilon);
      cudaDeviceSynchronize();
      checkCudaError("Moving Spherocyls");
      
      cudaMemcpyAsync(h_bNewNbrs, d_bNewNbrs, sizeof(int), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      if (*h_bNewNbrs) {
	find_neighbors();
      }
    }
    //cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
    //cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
    //cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));
  }
  

  free(h_pfF0Square); free(h_pfFSquare); free(h_pfDSquare);
  cudaFree(d_pfF0Square); cudaFree(d_pfFSquare); cudaFree(d_pfFF0); cudaFree(d_pfDSquare);

  return 0;
}


void Spherocyl_Box::gradient_relax_2p(double dMinStep, double dMaxStep, int nMaxSteps = 100000)
{
  cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));
  for (int i = 0; i < 4; i++) {
    h_pdLineEnergy[i] = 0;
  }

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  printf("Relaxation start energy: %g\n", h_pdLineEnergy[0]);
  find_neighbors();
  h_pdLineEnergy[5] = h_pdLineEnergy[0];
  h_pdLineEnergy[3] = h_pdLineEnergy[0];
  h_pdLineEnergy[4] = h_pdLineEnergy[0];
  h_pdLineEnergy[0] = 0;

  long unsigned int nTime = 0;
  while (nTime < nMaxSteps) {
    printf("Step %lu\n", nTime);
    int ret = new_cjpr_relax_step_2p(dMinStep, dMaxStep);
    //find_neighbors();
    if (ret == 1) {
      cudaMemcpyAsync(h_pdX, d_pdX, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      //save_positions(nTime);
      break;
    }
    else if (ret == 2) {
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

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }

  printf("Relaxation stopped after %lu steps with energy %g\n", nTime, h_pdLineEnergy[0]);
  
}


void Spherocyl_Box::cjpr_relax_2p(double dMinStep, double dMaxStep, int nMaxSteps = 100000)
{
  cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
  cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));
  for (int i = 0; i < 4; i++) {
    h_pdLineEnergy[i] = 0;
  }

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }
  printf("Relaxation start energy: %g\n", h_pdLineEnergy[0]);
  find_neighbors();
  h_pdLineEnergy[0] = 0;

  long unsigned int nTime = 0;
  while (nTime < nMaxSteps) {
    int ret = cjpr_relax_step_2p(dMinStep, dMaxStep);
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
    if (nTime % (3*m_nSpherocyls) == 0) {
      cudaMemset((void*) d_pdDx, 0, m_nSpherocyls*sizeof(double));
      cudaMemset((void*) d_pdDy, 0, m_nSpherocyls*sizeof(double));
      cudaMemset((void*) d_pdDt, 0, m_nSpherocyls*sizeof(double));
    }
  }

  find_neighbors();

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");

  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  h_pdLineEnergy[0] = 0;
  for (int j = 0; j < m_nGridSize; j++) {
    h_pdLineEnergy[0] += h_pdBlockSums[j];
  }

  printf("Relaxation stopped after %lu steps with energy %g\n", nTime, h_pdLineEnergy[0]);
  
}



void Spherocyl_Box::compress_qs2p(double dMaxPacking, double dResizeStep)
{
  assert(dResizeStep != 0);
  dResizeStep = fabs(dResizeStep);
  calculate_packing();
  assert(dMaxPacking >= m_dPacking);

  int nCompressTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }

  double dCJMinStep = 1e-12;
  double dCJMaxStep = 10;
  int nCJMaxSteps = 1000000;
  cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
  calculate_stress_energy_2p();
  cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
  save_positions(nCompressTime);
  cudaDeviceSynchronize();
  fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nCompressTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
  while (m_dPacking < dMaxPacking) {
    nCompressTime += 1;
    m_dStep = 0.01;
    resize_step_2p(nCompressTime, -dResizeStep, 0, 0);
    //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, -dResizeStep, d_pdX, d_pdY);
    //m_dLx -= dResizeStep*m_dLx;
    //m_dLy -= dResizeStep*m_dLy;
    cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
    
    calculate_stress_energy_2p();
    cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    calculate_packing();
    save_positions(nCompressTime);
    cudaDeviceSynchronize();
    fprintf(m_pOutfSE, "%d %.9f %g %g %g %g %g\n", nCompressTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
    fflush(m_pOutfSE);
  }

  fclose(m_pOutfSE);
}

void Spherocyl_Box::expand_qs2p(double dMinPacking, double dResizeStep)
{
  assert(dResizeStep != 0);
  dResizeStep = fabs(dResizeStep);
  calculate_packing();
  assert(dMinPacking <= m_dPacking);

  int nCompressTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }

  double dCJMinStep = 1e-12;
  double dCJMaxStep = 10;
  int nCJMaxSteps = 1000000;
  cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
  calculate_stress_energy_2p();
  cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
  save_positions(nCompressTime);
  cudaDeviceSynchronize();
  fprintf(m_pOutfSE, "%d %f %g %g %g %g %g\n", nCompressTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
  while (m_dPacking > dMinPacking) {
    nCompressTime += 1;
    m_dStep = 0.01;
    resize_step_2p(nCompressTime, dResizeStep, 0, 0);
    //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, -dResizeStep, d_pdX, d_pdY);
    //m_dLx -= dResizeStep*m_dLx;
    //m_dLy -= dResizeStep*m_dLy;
    cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
    
    calculate_stress_energy_2p();
    cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    calculate_packing();
    save_positions(nCompressTime);
    cudaDeviceSynchronize();
    fprintf(m_pOutfSE, "%d %.9f %g %g %g %g %g\n", nCompressTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
    fflush(m_pOutfSE);
  }

  fclose(m_pOutfSE);
}

int Spherocyl_Box::resize_to_energy_2p(int nTime, double dEnergy, double dStep)
{
  dStep = fabs(dStep);
  double dCJMinStep = 1e-15;
  double dCJMaxStep = 10;
  int nCJMaxSteps = 200000;

  find_neighbors();

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
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
      resize_step_2p(nTime, -dStep, 0, 0);
      //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, -dStep, d_pdX, d_pdY);
      //m_dLx -= dStep*m_dLx;
      //m_dLy -= dStep*m_dLy;
      cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
      
      calculate_stress_energy_2p();
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      calculate_packing();
      save_positions(nTime);
      cudaDeviceSynchronize();
      fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
      fflush(m_pOutfSE);
      dStep *= 1.1;
    }
  }
  else {
    while (h_pdLineEnergy[0] > dEnergy) {
      nTime += 1;
      m_dStep = 0.01;
      resize_step_2p(nTime, dStep, 0, 0);
      //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, dStep, d_pdX, d_pdY);
      //m_dLx += dStep*m_dLx;
      //m_dLy += dStep*m_dLy;
      cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
      
      calculate_stress_energy_2p();
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      calculate_packing();
      save_positions(nTime);
      cudaDeviceSynchronize();
      fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
      fflush(m_pOutfSE);
      dStep *= 1.1;
    }
  }
  
  return nTime;  
}

int Spherocyl_Box::y_resize_to_energy_2p(int nTime, double dEnergy, double dStep)
{
  dStep = fabs(dStep);
  double dCJMinStep = 1e-15;
  double dCJMaxStep = 10;
  int nCJMaxSteps = 200000;

  find_neighbors();

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
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
      y_resize_step_2p(nTime, -dStep, 0, 0);
      //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, -dStep, d_pdX, d_pdY);
      //m_dLx -= dStep*m_dLx;
      //m_dLy -= dStep*m_dLy;
      cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
      
      calculate_stress_energy_2p();
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      calculate_packing();
      save_positions(nTime);
      cudaDeviceSynchronize();
      fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
      fflush(m_pOutfSE);
      dStep *= 1.1;
    }
  }
  else {
    while (h_pdLineEnergy[0] > dEnergy) {
      nTime += 1;
      m_dStep = 0.01;
      y_resize_step_2p(nTime, dStep, 0, 0);
      //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, dStep, d_pdX, d_pdY);
      //m_dLx += dStep*m_dLx;
      //m_dLy += dStep*m_dLy;
      cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
      
      calculate_stress_energy_2p();
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      calculate_packing();
      save_positions(nTime);
      cudaDeviceSynchronize();
      fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
      fflush(m_pOutfSE);
      dStep *= 1.1;
    }
  }
  
  return nTime;  
}

int Spherocyl_Box::resize_relax_step_2p(int nTime, double dStep, int nCJMaxSteps)
{
  dStep = fabs(dStep);
  double dCJMinStep = 1e-15;
  double dCJMaxStep = 100;
  //int nCJMaxSteps = 2000000;

  find_neighbors();

  calc_fe_2p <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nBlockSize*sizeof(double)>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dLx, m_dLy, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, d_pdMOI, d_pdFx, d_pdFy, d_pdFt, d_pdBlockSums);
  cudaDeviceSynchronize();
  checkCudaError("Calculating forces and energy");
  cudaMemcpyAsync(h_pdBlockSums, d_pdBlockSums, m_nGridSize*sizeof(double), cudaMemcpyDeviceToHost);
  h_pdLineEnergy[0] = 0;
  cudaDeviceSynchronize();
  for (int i = 0; i < m_nGridSize; i++) {
    h_pdLineEnergy[0] += h_pdBlockSums[i];
  }
  
 
  m_dStep = 0.01;
  resize_step_2p(nTime, -dStep, 0, 0);
  //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, -dStep, d_pdX, d_pdY);
  //m_dLx -= dStep*m_dLx;
  //m_dLy -= dStep*m_dLy;
  cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
  nTime += 1;
      
  calculate_stress_energy_2p();
  cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
  calculate_packing();
  save_positions(nTime);
  cudaDeviceSynchronize();
  fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
  fflush(m_pOutfSE);

  
  m_dStep = 0.01;
  resize_step_2p(nTime, dStep, 0, 0);
  //resize_coords_2p <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, dStep, d_pdX, d_pdY);
  //m_dLx += dStep*m_dLx;
  //m_dLy += dStep*m_dLy;
  cjpr_relax_2p(dCJMinStep, dCJMaxStep, nCJMaxSteps);
  nTime += 1;
      
  calculate_stress_energy_2p();
  cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
  calculate_packing();
  save_positions(nTime);
  cudaDeviceSynchronize();
  fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
  fflush(m_pOutfSE);
  
  
  return nTime;  
}

void Spherocyl_Box::find_jam_2p(double dJamE, double dResizeStep)
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

  double dMinResizeStep = 1e-15;
  int nOldTime = 0;
  while (dResizeStep > dMinResizeStep) {
    nTime = resize_to_energy_2p(nTime, dJamE, dResizeStep);
    dResizeStep = dResizeStep*pow(1.1, (nTime-nOldTime-1))/2;
    nOldTime = nTime;
  }
  save_positions_bin(nTime);

  fclose(m_pOutfSE);

}

void Spherocyl_Box::y_compress_energy_2p(double dTargetE, double dResizeStep)
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

  double dMinResizeStep = 1e-14;
  int nOldTime = 0;
  while (dResizeStep > dMinResizeStep) {
    nTime = y_resize_to_energy_2p(nTime, dTargetE, dResizeStep);
    dResizeStep = dResizeStep*pow(1.1, (nTime-nOldTime-1))/2;
    nOldTime = nTime;
  }
  save_positions_bin(nTime);

  fclose(m_pOutfSE);
}


void Spherocyl_Box::find_energy_2p(double dResizeStep, double dMinResizeStep, int nMaxSteps)
{
  int nTime = 0;
  char szBuf[200];
  sprintf(szBuf, "%s/%s", m_szDataDir, m_szFileSE);
  const char *szPathSE = szBuf;
  printf("%s\n", szBuf);
  m_pOutfSE = fopen(szPathSE, "w");
  if (m_pOutfSE == NULL) {
    fprintf(stderr, "Could not open file for writing");
    exit(1);
  }
  save_positions(0);

  double L0 = m_dLx;

  //double dMinResizeStep = 1e-15;
  //nTime = resize_relax_step_2p(nTime, dResizeStep, nMaxSteps);
  //dResizeStep *= -0.75;
  while (fabs(dResizeStep) > dMinResizeStep) {
    printf("Relaxing with step size: %g\n", dResizeStep);
    //nTime = resize_relax_step_2p(nTime, dResizeStep, nMaxSteps);
    nTime = resize_relax_step_2p(nTime, dResizeStep, nMaxSteps);
    dResizeStep /= 2;
  }
  double dErr = 1 - m_dLx / L0;
  if (dErr != 0) {
    m_dStep = 0.01;
    resize_step_2p(nTime, dErr, 0, 0);
    cjpr_relax_2p(1e-15, 100., nMaxSteps);
    nTime += 1;

    calculate_stress_energy_2p();
    cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
    calculate_packing();
    save_positions(nTime);
    cudaDeviceSynchronize();
    fprintf(m_pOutfSE, "%d %.10f %.9g %.9g %.9g %.9g %.9g\n", nTime, m_dPacking, *m_pfEnergy, *m_pfPxx, *m_pfPyy, *m_pfPxy, *m_pfPyx);
    fflush(m_pOutfSE);
  }

  save_positions_bin(nTime);

  fclose(m_pOutfSE);

}

//void Spherocyl_Box::find_energy_2p2(double dResizeStep, double dMinResize
