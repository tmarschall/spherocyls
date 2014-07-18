// -*- c++ -*-
/*
* find_neighbors.cu
*
*  
*
*
*/

#include <cuda.h>
#include "spherocyl_box.h"
#include "cudaErr.h"
#include "data_primitives.h"
#include <math.h>

using namespace std;

const double D_PI = 3.14159265358979;


__global__ void find_moi(int nSpherocyls, double *pdMOI, double *pdA)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;

  while (thid < nSpherocyls) {
    double dA = pdA[thid];

    pdMOI[thid] = dA*dA / 3;
    
    thid += blockDim.x * gridDim.x;
  }
}

__global__ void find_rot_consts(int nSpherocyls, double *pdMOI, double *pdIsoCoeff, double *pdR, double *pdA)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;

  while (thid < nSpherocyls) {
    double dR = pdR[thid];
    double dA = pdA[thid];
    double dAlpha = dA/dR;
    
    double dC = 3*D_PI + 24*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha;
    double dB = D_PI*dR*dR + 4*dR*dA;
    pdMOI[thid] = dR*dR*dR*dR*dC/(6*dB);
    pdIsoCoeff[thid] = (8*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha)/dC;

    thid += blockDim.x*gridDim.x;
  }
}



///////////////////////////////////////////////////////////////
// Find the Cell ID for each particle:
//  The list of cell IDs for each particle is returned to pnCellID
//  A list of which particles are in each cell is returned to pnCellList
//
// *NOTE* if there are more than nMaxPPC particles in a given cell,
//  not all of these particles will get added to the cell list
///////////////////////////////////////////////////////////////
__global__ void find_cells(int nSpherocyls, int nMaxPPC, double dCellW, double dCellH,
			   int nCellCols, double dL, double *pdX, double *pdY, 
			   int *pnCellID, int *pnPPC, int *pnCellList)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nSpherocyls) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // I often allow the stored coordinates to drift slightly outside the box limits
    //  until 
    if (dY > dL)
      {
	dY -= dL;
	pdY[nPID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nPID] = dY;
      }
    if (dX > dL)
      {
	dX -= dL;
	pdX[nPID] = dX;
      }
    else if (dX < 0)
      {
	dX += dL;
	pdX[nPID] = dX;
      }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    pnCellID[nPID] = nCellID;

    // Add 1 particle to a cell safely (only allows one thread to access the memory
    //  address at a time). nPPC is the original value, not the result of addition 
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    // only add particle to cell if there is not already the maximum number in cell
    if (nPPC < nMaxPPC)
      pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
    else
      nPPC = atomicAdd(pnPPC + nCellID, -1);

    nPID += nThreads;
  }
}


////////////////////////////////////////////////////////////////
// Here a list of possible contacts is created for each particle
//  The list of neighbors is returned to pnNbrList
//
// This is one function that I may target for optimization in
//  the future because I know it is slowed down by "branch divergence"
////////////////////////////////////////////////////////////////
__global__ void find_nbrs(int nSpherocyls, int nMaxPPC, int *pnCellID, int *pnPPC, 
			  int *pnCellList, int *pnAdjCells, int nMaxNbrs, int *pnNPP, 
			  int *pnNbrList, double *pdX, double *pdY, double *pdR, 
			  double *pdA, double dEpsilon, double dL, double dGamma)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = gridDim.x * blockDim.x;

  while (nPID < nSpherocyls)
    {
      double dX = pdX[nPID];
      double dY = pdY[nPID];
      double dR = pdR[nPID];
      double dA = pdA[nPID];
      int nNbrs = 0;

      // Particles in adjacent cells are added if they are close enough to 
      //  interact without each moving by more than dEpsilon/2
      int nCellID = pnCellID[nPID];
      int nP = pnPPC[nCellID];
      for (int p = 0; p < nP; p++)
	{
	  int nAdjPID = pnCellList[nCellID*nMaxPPC + p];
	  if (nAdjPID != nPID)
	    {
	      double dSigma = dR + dA + pdR[nAdjPID] + pdA[nAdjPID] + dEpsilon;
	      double dDeltaY = dY - pdY[nAdjPID];
	      dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	      
	      if (fabs(dDeltaY) < dSigma)
		{
		  double dDeltaX = dX - pdX[nAdjPID];
		  dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		  double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		  double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		  if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		    {
		      // This indexing makes global memory accesses more coalesced
		      if (nNbrs < nMaxNbrs)
			{
			  pnNbrList[nSpherocyls * nNbrs + nPID] = nAdjPID;
			  nNbrs += 1;
			}
		    }
		}
	    }
	}

      for (int nc = 0; nc < 8; nc++)
	{
	  int nAdjCID = pnAdjCells[8 * nCellID + nc];
	  nP = pnPPC[nAdjCID];
	  for (int p = 0; p < nP; p++)
	    {
	      int nAdjPID = pnCellList[nAdjCID*nMaxPPC + p];
	      // The maximum distance at which two particles could contact
	      //  plus a little bit of moving room - dEpsilon 
	      double dSigma = dR + dA + pdA[nAdjPID] + pdR[nAdjPID] + dEpsilon;
	      double dDeltaY = dY - pdY[nAdjPID];
		
	      // Make sure were finding the closest separation
	      dDeltaY += dL * ((dDeltaY < -0.5 * dL) - (dDeltaY > 0.5 * dL));
	      
	      if (fabs(dDeltaY) < dSigma)
		{
		  double dDeltaX = dX - pdX[nAdjPID];
		  dDeltaX += dL * ((dDeltaX < -0.5 * dL) - (dDeltaX > 0.5 * dL));
		  
		  // Go to unsheared coordinates
		  double dDeltaRx = dDeltaX + dGamma * dDeltaY;
		  // Also look at distance when the strain parameter is at its max (0.5)
		  double dDeltaRx2 = dDeltaX + 0.5 * dDeltaY;
		  if (fabs(dDeltaRx) < dSigma || fabs(dDeltaRx2) < dSigma)
		    {
		      if (nNbrs < nMaxNbrs)
			{
			  pnNbrList[nSpherocyls * nNbrs + nPID] = nAdjPID;
			  nNbrs += 1;
			}
		    }
		}
	    }   
	}
      
      pnNPP[nPID] = nNbrs;
      nPID += nThreads;
    }
}



///////////////////////////////////////////////////////////////
// Finds a list of possible contacts for each particle
//
// Usually when things are moving I keep track of an Xmoved and Ymoved
//  and only call this to make a new list of neighbors if some particle
//  has moved more than (dEpsilon / 2) in some direction
///////////////////////////////////////////////////////////////
void Spherocyl_Box::find_neighbors()
{
  // reset each byte to 0
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nSpherocyls);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nSpherocyls);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  if (!m_bMOI)
    find_rot_consts <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, d_pdMOI, d_pdIsoC, d_pdR, d_pdA);

  find_cells <<<m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, m_nMaxPPC, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdX, d_pdY, d_pnCellID, d_pnPPC, d_pnCellList);
  cudaThreadSynchronize();
  checkCudaError("Finding cells");


  find_nbrs <<<m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, m_nMaxPPC, d_pnCellID, d_pnPPC, d_pnCellList, d_pnAdjCells,
     m_nMaxNbrs, d_pnNPP, d_pnNbrList, d_pdX, d_pdY, d_pdR, d_pdA, 
     m_dEpsilon, m_dL, m_dGamma);
  cudaThreadSynchronize();
  checkCudaError("Finding neighbors");

  /*
  int *h_pnCellID = (int*) malloc(sizeof(int)*3*m_nSpherocyls);
  int *h_pnNPP = (int*) malloc(sizeof(int)*3*m_nSpherocyls);
  int *h_pnNbrList = (int*) malloc(sizeof(int)*3*m_nSpherocyls*m_nMaxNbrs);
  cudaMemcpy(h_pnCellID, d_pnCellID, sizeof(int)*3*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNPP,d_pnNPP, sizeof(int)*3*m_nSpherocyls,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*3*m_nSpherocyls*m_nMaxNbrs, cudaMemcpyDeviceToHost);

  for (int p = 0; p < 3*m_nSpherocyls; p++) {
    printf("Spherocyl: %d, Cell: %d, neighbors: %d\n", 
	   p, h_pnCellID[p], h_pnNPP[p]);
    for (int n = 0; n < h_pnNPP[p]; n++) {
      printf("%d ", h_pnNbrList[n*3*m_nSpherocyls + p]);
    }
    printf("\n");
    fflush(stdout);
  }

  free(h_pnCellID); free(h_pnNPP); free(h_pnNbrList);
  */
}


////////////////////////////////////////////////////////////////////////////////////
// Sets gamma back by 1 (used when gamma > 0.5)
//  also finds the cells in the process
//
///////////////////////////////////////////////////////////////////////////////////
__global__ void set_back_coords(int nSpherocyls, double dL, double *pdX, double *pdY, double *pdPhi)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nSpherocyls) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    double dPhi = pdPhi[nPID];
    
    // I often allow the stored coordinates to drift slightly outside the box limits
    if (dPhi > D_PI)
      {
	dPhi -= 2*D_PI;
	pdPhi[nPID] = dPhi;
      }
    else if (dPhi < -D_PI)
      {
	dPhi += 2*D_PI;
	pdPhi[nPID] = dPhi;
      }
    if (dY > dL)
      {
	dY -= dL;
	pdY[nPID] = dY;
      }
    else if (dY < 0)
      {
	dY += dL;
	pdY[nPID] = dY;
      }
    
    // When gamma -> gamma-1, Xi -> Xi + Yi
    dX += dY;
    if (dX < 0)
      {
	dX += dL;
      }
    while (dX > dL)
      {
	dX -= dL;
      }
    pdX[nPID] = dX;

    nPID += nThreads;
  }
  

}


void Spherocyl_Box::set_back_gamma()
{
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);
  cudaMemset((void *) d_pdXMoved, 0, sizeof(double)*m_nSpherocyls);
  cudaMemset((void *) d_pdYMoved, 0, sizeof(double)*m_nSpherocyls);
  cudaMemset((void *) d_bNewNbrs, 0, sizeof(int));

  /*
  int *h_pnCellID = (int*) malloc(sizeof(int)*m_nSpherocyls);
  int *h_pnNPP = (int*) malloc(sizeof(int)*m_nSpherocyls);
  int *h_pnNbrList = (int*) malloc(sizeof(int)*m_nSpherocyls*m_nMaxNbrs);
  cudaMemcpy(h_pnCellID, d_pnCellID, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNPP,d_pnNPP, sizeof(int)*m_nSpherocyls,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*m_nSpherocyls*m_nMaxNbrs, cudaMemcpyDeviceToHost);

  printf("\nSetting coordinate system back by gamma\n\nOld neighbors:");
  for (int p = 0; p < m_nSpherocyls; p++) {
    printf("Spherocyl: %d, Cell: %d, neighbors: %d\n", 
	   p, h_pnCellID[p], h_pnNPP[p]);
    for (int n = 0; n < h_pnNPP[p]; n++) {
      printf("%d ", h_pnNbrList[n*m_nSpherocyls + p]);
    }
    printf("\n");
    fflush(stdout);
  }
*/

  set_back_coords <<<m_nGridSize, m_nBlockSize>>> 
    (m_nSpherocyls, m_dL, d_pdX, d_pdY, d_pdPhi);
  cudaThreadSynchronize();
  checkCudaError("Finding new coordinates, cells");
  m_dGamma -= 1;
  m_dTotalGamma = int(m_dTotalGamma+1) + m_dGamma;  // Gamma total will have diverged slightly due to differences in precision with gamma

  find_neighbors();

  /*
  cudaMemcpy(h_pnCellID, d_pnCellID, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNPP,d_pnNPP, sizeof(int)*m_nSpherocyls,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*m_nSpherocyls*m_nMaxNbrs, cudaMemcpyDeviceToHost);
  printf("\nNew Neighbors:\n");
  for (int p = 0; p < m_nSpherocyls; p++) {
    printf("Spherocyl: %d, Cell: %d, neighbors: %d\n", 
	   p, h_pnCellID[p], h_pnNPP[p]);
    for (int n = 0; n < h_pnNPP[p]; n++) {
      printf("%d ", h_pnNbrList[n*m_nSpherocyls + p]);
    }
    printf("\n");
    fflush(stdout);
  }
  
  free(h_pnCellID); free(h_pnNPP); free(h_pnNbrList);
  */
}


////////////////////////////////////////////////////////////////////////////
// Finds cells for all particles regardless of maximum particle per cell
//  used for reordering particles
/////////////////////////////////////////////////////////////////////////
__global__ void find_cells_nomax(int nSpherocyls, double dCellW, double dCellH,
				 int nCellCols, double dL, double *pdX, double *pdY, 
				 int *pnCellID, int *pnPPC)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nSpherocyls) {
    double dX = pdX[nPID];
    double dY = pdY[nPID];
    
    // Particles are allowed to drift slightly outside the box limits
    //  until cells are reassigned due to a particle drift of dEpsilon/2 
    if (dY > dL) {
      dY -= dL; 
      pdY[nPID] = dY; }
    else if (dY < 0) {
      dY += dL;
      pdY[nPID] = dY; }
    if (dX > dL) {
      dX -= dL; 
      pdX[nPID] = dX; }
    else if (dX < 0) {
      dX += dL;
      pdX[nPID] = dX; }

    //find the cell ID, add a particle to that cell 
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH); 
    int nCellID = nCol + nRow * nCellCols;
    
    pnCellID[nPID] = nCellID;
    int nPPC = atomicAdd(pnPPC + nCellID, 1);
    
    nPID += nThreads; }
}

__global__ void reorder_part(int nSpherocyls, double *pdTempX, double *pdTempY,
			     double *pdTempR, double *pdTempA, 
			     int *pnTempInitID, double *pdX, double *pdY, 
			     double *pdR, double *pdA, int *pnInitID, 
			     int *pnMemID, int *pnCellID, int *pnCellSID)
{
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nSpherocyls) {
    double dX = pdTempX[nPID];
    double dY = pdTempY[nPID];
    double dR = pdTempR[nPID];
    double dA = pdTempA[nPID];
    int nInitID = pnTempInitID[nPID];

    int nCellID = pnCellID[nPID];
    int nNewID = atomicAdd(pnCellSID + nCellID, 1);
    
    pdX[nNewID] = dX;
    pdY[nNewID] = dY;
    pdR[nNewID] = dR;
    pdA[nNewID] = dA;
    pnMemID[nInitID] = nNewID;
    pnInitID[nNewID] = nInitID;

    nPID += nThreads; 
  }
}

__global__ void invert_IDs(int nIDs, int *pnIn, int *pnOut)
{
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (thid < nIDs) {
    int i = pnIn[thid];
    pnOut[i] = thid; 
    thid += nThreads; }
    
}

void Spherocyl_Box::reorder_particles()
{
  cudaMemset((void *) d_pnPPC, 0, sizeof(int)*m_nCells);

  //find particle cell IDs and number of particles in each cell
  find_cells_nomax <<<m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, m_dCellW, m_dCellH, m_nCellCols, 
     m_dL, d_pdX, d_pdY, d_pnCellID, d_pnPPC);
  cudaThreadSynchronize();
  checkCudaError("Reordering particles: Finding cells");

  int *d_pnCellSID;
  int *d_pnTempInitID;
  double *d_pdTempR; 
  double *d_pdTempA; 
  cudaMalloc((void **) &d_pnCellSID, sizeof(int) * m_nCells);
  cudaMalloc((void **) &d_pdTempR, sizeof(double) * m_nSpherocyls);
  cudaMalloc((void **) &d_pdTempA, sizeof(double) * m_nSpherocyls);
  cudaMalloc((void **) &d_pnTempInitID, sizeof(int) * m_nSpherocyls);
  cudaMemcpy(d_pdTempX, d_pdX, sizeof(double) * m_nSpherocyls, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempY, d_pdY, sizeof(double) * m_nSpherocyls, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempR, d_pdR, sizeof(double) * m_nSpherocyls, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdTempA, d_pdA, sizeof(double) * m_nSpherocyls, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pnTempInitID, d_pnInitID, sizeof(int) * m_nSpherocyls, cudaMemcpyDeviceToDevice);

  exclusive_scan(d_pnPPC, d_pnCellSID, m_nCells);

  /*
  int *h_pnCellSID = (int*) malloc(m_nCells * sizeof(int));
  int *h_pnCellNPart = (int*) malloc(m_nCells * sizeof(int));
  cudaMemcpy(h_pnCellNPart, d_pnCellNPart, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pnCellSID, d_pnCellSID, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost);
  for (int c = 0; c < m_nCells; c++)
    {
      printf("%d %d\n", h_pnCellNPart[c], h_pnCellSID[c]);
    }
  free(h_pnCellSID);
  free(h_pnCellNPart);
  */

  //reorder particles based on cell ID (first by Y direction)
  reorder_part <<<m_nGridSize, m_nBlockSize>>>
    (m_nSpherocyls, d_pdTempX, d_pdTempY, d_pdTempR, d_pdTempA, d_pnTempInitID,
     d_pdX, d_pdY, d_pdR, d_pdA, d_pnInitID, d_pnMemID, d_pnCellID, d_pnCellSID);
  cudaThreadSynchronize();
  checkCudaError("Reordering particles: changing order");

  //invert_IDs <<<m_nGridSize, m_nBlockSize>>> (m_nSpherocyls, d_pnMemID, d_pnInitID);
  cudaMemcpyAsync(h_pnMemID, d_pnMemID, m_nSpherocyls*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdR, d_pdR, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_pdA, d_pdA, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  cudaFree(d_pnCellSID); cudaFree(d_pnTempInitID);
  cudaFree(d_pdTempR); cudaFree(d_pdTempA);

  m_bMOI = 0;
  find_neighbors();
}


////////////////////////////////////////////////////////////////////////
// Sets the particle IDs to their order in memory
//  so the current IDs become the initial IDs
/////////////////////////////////////////////////////////////////////
void Spherocyl_Box::reset_IDs()
{
  ordered_array(d_pnInitID, m_nSpherocyls, m_nGridSize, m_nBlockSize);
  cudaMemcpyAsync(h_pnMemID, d_pnInitID, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_pnMemID, d_pnInitID, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToDevice);
  
}
