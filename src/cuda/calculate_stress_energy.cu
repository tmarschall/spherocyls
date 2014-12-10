// -*- c++ -*-
/*
* calculate_stress_energy.cu
*
*
*/

#include <cuda.h>
#include "spherocyl_box.h"
#include "cudaErr.h"
#include <math.h>
#include <sm_20_atomic_functions.h>
#include <stdio.h>


using namespace std;


////////////////////////////////////////////////////////
// Calculates energy, stress tensor, forces:
//  Returns returns energy, pxx, pyy and pxy to pn
//  Returns 
//
// The neighbor list pnNbrList has a list of possible contacts
//  for each particle and is found in find_neighbors.cu
//
/////////////////////////////////////////////////////////
template<Potential ePot>
__global__ void calc_se(int nSpherocyls, int *pnNPP, int *pnNbrList, double dL,
			double dGamma, double *pdX, double *pdY, double *pdPhi,
			double *pdR, double *pdA, double *pdFx, double *pdFy, 
			double *pdFt, float *pfSE)
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
	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
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

	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	  double s = -(b*t+d)/a;
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
	      double dCx = 0.5 * dDx - s*nxA;
	      double dCy = 0.5 * dDy - s*nyA;
	      dFx += dPfx;
	      dFy += dPfy;
	      dFt += s*nxA * dPfy - s*nyA * dPfx;

	      sData[thid] += dCx * dPfx / (dL * dL);
	      sData[thid + offset] += dCy * dPfy / (dL * dL);
	      sData[thid + 2*offset] += dCx * dPfy / (dL * dL);
	      sData[thid + 3*offset] += dCy * dPfx / (dL * dL); 
	      if (nAdjPID > nPID)
		{
		  sData[thid + 4*offset] += dDVij * dSigma * (1.0 - dDij / dSigma) / (dAlpha * dL * dL);
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


void Spherocyl_Box::calculate_stress_energy()
{
  cudaMemset((void*) d_pfSE, 0, 5*sizeof(float));
  
  //dim3 grid(m_nGridSize);
  //dim3 block(m_nBlockSize);
  //size_t smem = m_nSM_CalcSE;
  //printf("Configuration: %d x %d x %d\n", m_nGridSize, m_nBlockSize, m_nSM_CalcSE);

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se <HARMONIC> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdX, d_pdY, 
	 d_pdPhi, d_pdR, d_pdA, d_pdFx, d_pdFy, d_pdFt, d_pfSE);
      break;
    case HERTZIAN:
      calc_se <HERTZIAN> <<<m_nGridSize, m_nBlockSize, m_nSM_CalcSE>>>
	(m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdX, d_pdY, 
	 d_pdPhi, d_pdR, d_pdA, d_pdFx, d_pdFy, d_pdFt, d_pfSE);
    }
  cudaThreadSynchronize();
  checkCudaError("Calculating stresses and energy");
}


__global__ void find_contact(int nSpherocyls, int *pnNPP, int *pnNbrList, double dL,
			     double dGamma, double *pdX, double *pdY, double *pdPhi,
			     double *pdR, double *pdA, int *nContacts)
{
  // Declare shared memory pointer, the size is determined at the kernel launch
  extern __shared__ int sCont[];
  int thid = threadIdx.x;
  int nPID = thid + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;
  sCont[thid] = 0;

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
	  
	  double dDeltaX = dX - pdX[nAdjPID];
	  double dDeltaY = dY - pdY[nAdjPID];
	  double dPhiB = pdPhi[nAdjPID];
	  double dSigma = dR + pdR[nAdjPID];
	  double dB = pdA[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
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

	  double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	  double s = -(b*t+d)/a;
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
	      sCont[thid] += 1;
	    }
	}	
      
      nPID += nThreads;
    }
  __syncthreads();
  
  // Now we do a parallel reduction sum to find the total number of contacts
  int stride = blockDim.x / 2;
  while (stride > 32)
    {
      if (thid < stride)
	{
	  sCont[thid] += sCont[thid + stride];
	}
      stride /= 2;  
      __syncthreads();
    }
  if (thid < 32) //unroll end of loop
    {
      sCont[thid] += sCont[thid + 32];
      if (thid < 16)
	{
	  sCont[thid] += sCont[thid + 16];
	  if (thid < 8)
	    {
	      sCont[thid] += sCont[thid + 8];
	      if (thid < 4)
		{
		  sCont[thid] += sCont[thid + 4];
		  if (thid < 2) {
		    sCont[thid] += sCont[thid + 2];
		    if (thid == 0) {
		      sCont[0] += sCont[1];
		      int nBContacts = atomicAdd(nContacts, sCont[0]);
		    }
		  }
		}
	    }
	}
    } 
 
}

bool Spherocyl_Box::check_for_contacts()
{
  int *pnContacts;
  cudaMalloc((void**) &pnContacts, sizeof(int));
  cudaMemset((void*) pnContacts, 0, sizeof(int));
  
  int nSMSize = m_nBlockSize * sizeof(int);
  find_contact <<<m_nGridSize, m_nBlockSize, nSMSize>>>
    (m_nSpherocyls, d_pnNPP, d_pnNbrList, m_dL, m_dGamma, d_pdX, d_pdY, 
     d_pdPhi, d_pdR, d_pdA, pnContacts);
  cudaThreadSynchronize();

  int nContacts;
  cudaMemcpy(&nContacts, pnContacts, sizeof(int), cudaMemcpyHostToDevice);
  cudaFree(pnContacts);

  return (bool)nContacts;
}











////////////////////////////////////////////////////////////////////////////////////
#if GOLD_FUNCS == 1
void calc_se_gold(Potential ePot, int nGridDim, int nBlockDim, int sMemSize, 
		  int nSpherocyls, int *pnNPP, int *pnNbrList, double dL,
		  double dGamma, double *pdX, double *pdY, double *pdR,
		  double *pdFx, double *pdFy, float *pfSE)
{
for (int b = 0; b < nGridDim; b++)
  {
    printf("Entering loop, block %d\n", b);
for (int thid = 0; thid < nBlockDim; thid++)
  {
    printf("Entering loop, thread %d\n", thid);
  // Declare shared memory pointer, the size is determined at the kernel launch
    double *sData = new double[sMemSize / sizeof(double)];
  int nPID = thid + b * nBlockDim;
  int nThreads = nBlockDim * nGridDim;
  int offset = nBlockDim + 8; // +8 helps to avoid bank conflicts (I think)
  for (int i = 0; i < 5; i++)
    sData[thid + i*offset] = 0.0;
  double dFx = 0.0;
  double dFy = 0.0;

  while (nPID < nSpherocyls)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dR = pdR[nPID];
      
      int nNbrs = pnNPP[nPID];
      for (int p = 0; p < nNbrs; p++)
	{
	  int nAdjPID = pnNbrList[nPID + p * nSpherocyls];
	  
	  double dDeltaX = dX - pdX[nAdjPID];
	  double dDeltaY = dY - pdY[nAdjPID];
	  double dSigma = dR + pdR[nAdjPID];
	  // Make sure we take the closest distance considering boundary conditions
	  dDeltaX += dL * ((dDeltaX < -0.5*dL) - (dDeltaX > 0.5*dL));
	  dDeltaY += dL * ((dDeltaY < -0.5*dL) - (dDeltaY > 0.5*dL));
	  // Transform from shear coordinates to lab coordinates
	  dDeltaX += dGamma * dDeltaY;
	  
	  // Check if they overlap
	  double dRSqr = dDeltaX*dDeltaX + dDeltaY*dDeltaY;
	  if (dRSqr < dSigma*dSigma)
	    {
	      double dDelR = sqrt(dRSqr);
	      double dDVij;
	      double dAlpha;
	      if (ePot == HARMONIC)
		{
		  dDVij = (1.0 - dDelR / dSigma) / dSigma;
		  dAlpha = 2.0;
		}
	      else if (ePot == HERTZIAN)
		{
		  dDVij = (1.0 - dDelR / dSigma) * sqrt(1.0 - dDelR / dSigma) / dSigma;
		  dAlpha = 2.5;
		}
	      double dPfx = dDeltaX * dDVij / dDelR;
	      double dPfy = dDeltaY * dDVij / dDelR;
	      dFx += dPfx;
	      dFy += dPfy;
	      if (nAdjPID > nPID)
		{
		  sData[thid] += dDVij * dSigma * (1.0 - dDelR / dSigma) / (dAlpha * dL * dL);
		  sData[thid + offset] += dPfx * dDeltaX / (dL * dL);
		  sData[thid + 2*offset] += dPfy * dDeltaY / (dL * dL);
		  sData[thid + 3*offset] += dPfy * dDeltaX / (dL * dL);
		  sData[thid + 4*offset] += dPfx * dDeltaY / (dL * dL);
		} 
	    }
	}
      pdFx[nPID] = dFx;
      pdFy[nPID] = dFy;
      dFx = 0.0;
      dFy = 0.0;
      
      nPID += nThreads;
    }
  
  // Now we do a parallel reduction sum to find the total number of contacts
  for (int s = 0; s < 5; s++)
    pfSE[s] += sData[thid + s*offset];
	 
  }
  }
}

void Spherocyl_Box::calculate_stress_energy_gold()
{
  printf("Calculating streeses and energy");
  cudaMemcpy(g_pnNPP, d_pnNPP, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pnNbrList, d_pnNbrList, sizeof(int)*m_nSpherocyls*m_nMaxNbrs, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdX, d_pdX, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdY, d_pdY, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_pdR, d_pdR, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++)
    g_pfSE[i] = 0.0;

  switch (m_ePotential)
    {
    case HARMONIC:
      calc_se_gold (HARMONIC, m_nGridSize, m_nBlockSize, m_nSM_CalcSE,
			  m_nSpherocyls, g_pnNPP, g_pnNbrList, m_dL, m_dGamma, 
			  g_pdX, g_pdY, g_pdR, g_pdFx, g_pdFy, g_pfSE);
      break;
    case HERTZIAN:
      calc_se_gold (HERTZIAN, m_nGridSize, m_nBlockSize, m_nSM_CalcSE,
			  m_nSpherocyls, g_pnNPP, g_pnNbrList, m_dL, m_dGamma, 
			  g_pdX, g_pdY, g_pdR, g_pdFx, g_pdFy, g_pfSE);
    }

  for (int p = 0; p < m_nSpherocyls; p++)
    {
      printf("Particle %d:  (%g, %g)\n", p, g_pdFx[p], g_pdFy[p]);
    }
  printf("Energy: %g\n", g_pfSE[0]);
  printf("Pxx: %g\n", g_pfSE[1]);
  printf("Pyy: %g\n", g_pfSE[2]);
  printf("P: %g\n", g_pfSE[1] + g_pfSE[2]);
  printf("Pxy: %g\n", g_pfSE[3]);
}

#endif
