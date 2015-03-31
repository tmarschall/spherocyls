/*  spherocyl_box.cpp
 *
 *
 */

#include "spherocyl_box.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>
#include "cudaErr.h"
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

const double D_PI = 3.14159265358979;

// Just setting things up here
//  configure_cells() decides how the space should be divided into cells
//  and which cells are next to each other
void Spherocyl_Box::configure_cells()
{
  assert(m_dRMax > 0.0);
  assert(m_dAMax >= 0.0);

  // Minimum height & width of cells
  //  Width is set so that it is only possible for particles in 
  //  adjacent cells to interact as long as |gamma| < 0.5
  double dWMin = 2.24 * (m_dRMax + m_dAMax) + m_dEpsilon;
  double dHMin = 2 * (m_dRMax + m_dAMax) + m_dEpsilon;

  m_nCellRows = max(static_cast<int>(m_dL / dHMin), 1);
  m_nCellCols = max(static_cast<int>(m_dL / dWMin), 1);
  m_nCells = m_nCellRows * m_nCellCols;
  cout << "Cells: " << m_nCells << ": " << m_nCellRows << " x " << m_nCellCols << endl;

  m_dCellW = m_dL / m_nCellCols;
  m_dCellH = m_dL / m_nCellRows;
  cout << "Cell dimensions: " << m_dCellW << " x " << m_dCellH << endl;

  h_pnPPC = new int[m_nCells];
  h_pnCellList = new int[m_nCells * m_nMaxPPC];
  h_pnAdjCells = new int[8 * m_nCells];
  cudaMalloc((void **) &d_pnPPC, m_nCells * sizeof(int));
  cudaMalloc((void **) &d_pnCellList, m_nCells*m_nMaxPPC*sizeof(int));
  cudaMalloc((void **) &d_pnAdjCells, 8*m_nCells*sizeof(int));
  m_nDeviceMem += m_nCells*(9+m_nMaxPPC)*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnPPC = new int[m_nCells];
  g_pnCellList = new int[m_nCells * m_nMaxPPC];
  g_pnAdjCells = new int[8 * m_nCells];
#endif
  
  // Make a list of which cells are next to each cell
  // This is done once for convinience since the boundary conditions
  //  make this more than trivial
  for (int c = 0; c < m_nCells; c++)
    {
      int nRow = c / m_nCellCols; 
      int nCol = c % m_nCellCols;

      int nAdjCol1 = (nCol + 1) % m_nCellCols;
      int nAdjCol2 = (m_nCellCols + nCol - 1) % m_nCellCols;
      h_pnAdjCells[8 * c] = nRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 1] = nRow * m_nCellCols + nAdjCol2;

      int nAdjRow = (nRow + 1) % m_nCellRows;
      h_pnAdjCells[8 * c + 2] = nAdjRow * m_nCellCols + nCol;
      h_pnAdjCells[8 * c + 3] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 4] = nAdjRow * m_nCellCols + nAdjCol2;
      
      nAdjRow = (m_nCellRows + nRow - 1) % m_nCellRows;
      h_pnAdjCells[8 * c + 5] = nAdjRow * m_nCellCols + nCol;
      h_pnAdjCells[8 * c + 6] = nAdjRow * m_nCellCols + nAdjCol1;
      h_pnAdjCells[8 * c + 7] = nAdjRow * m_nCellCols + nAdjCol2;
    }
  cudaMemcpy(d_pnAdjCells, h_pnAdjCells, 8*m_nCells*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError("Configuring cells");
}

// Set the thread configuration for kernel launches
void Spherocyl_Box::set_kernel_configs()
{
  switch (m_nSpherocyls)
    {
    case 512:
      m_nGridSize = 4;
      m_nBlockSize = 128;
      m_nSM_CalcF = 3*128*sizeof(double);
      m_nSM_CalcSE = 5*136*sizeof(double);
    case 1024:
      m_nGridSize = 8;  // Grid size (# of thread blocks)
      m_nBlockSize = 128; // Block size (# of threads per block)
      m_nSM_CalcF = 3*128*sizeof(double);
      m_nSM_CalcSE = 5*136*sizeof(double); // Size of shared memory per block
      break;
    case 1280:
      m_nGridSize = 10;
      m_nBlockSize = 128;
      m_nSM_CalcF = 3*128*sizeof(double);
      m_nSM_CalcSE = 5*136*sizeof(double);
      break;
    case 2048:
      m_nGridSize = 16;  // Grid size (# of thread blocks)
      m_nBlockSize = 128; // Block size (# of threads per block)
      m_nSM_CalcF = 3*128*sizeof(double);
      m_nSM_CalcSE = 5*136*sizeof(double); // Size of shared memory per block
      break;
    case 4096:
      m_nGridSize = 16;
      m_nBlockSize = 256;
      m_nSM_CalcF = 3*256*sizeof(double);
      m_nSM_CalcSE = 5*264*sizeof(double);
      break;
    default:
      m_nGridSize = m_nSpherocyls / 256 + (0 ? (m_nSpherocyls % 256 == 0) : 1);
      m_nBlockSize = 256;
      m_nSM_CalcF = 3*256*sizeof(double);
      m_nSM_CalcSE = 5*264*sizeof(double);
    };
  cout << "Kernel config (spherocyls):\n";
  cout << m_nGridSize << " x " << m_nBlockSize << endl;
  cout << "Shared memory allocation (calculating forces):\n";
  cout << (float)m_nSM_CalcF / 1024. << "KB" << endl;
  cout << "Shared memory allocation (calculating S-E):\n";
  cout << (float)m_nSM_CalcSE / 1024. << " KB" << endl; 
}

void Spherocyl_Box::construct_defaults()
{
  m_dGamma = 0.0;
  m_dTotalGamma = 0.0;
  m_dStep = 1;
  m_dStrainRate = 1e-3;
  m_szDataDir = "./";
  m_szFileSE = "sd_stress_energy.dat";

  cudaHostAlloc((void**) &h_bNewNbrs, sizeof(int), 0);
  *h_bNewNbrs = 1;
  cudaMalloc((void**) &d_bNewNbrs, sizeof(int));
  cudaMemcpyAsync(d_bNewNbrs, h_bNewNbrs, sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_pdTempX, sizeof(double)*m_nSpherocyls);
  cudaMalloc((void**) &d_pdTempY, sizeof(double)*m_nSpherocyls);
  cudaMalloc((void**) &d_pdTempPhi, sizeof(double)*m_nSpherocyls);
  cudaMalloc((void**) &d_pdXMoved, sizeof(double)*m_nSpherocyls);
  cudaMalloc((void**) &d_pdYMoved, sizeof(double)*m_nSpherocyls);
  cudaMalloc((void**) &d_pdMOI, sizeof(double)*m_nSpherocyls);
  cudaMalloc((void**) &d_pdIsoC, sizeof(double)*m_nSpherocyls);
  m_bMOI = 0;
  m_nDeviceMem += 6*m_nSpherocyls*sizeof(double);
#if GOLD_FUNCS == 1
  *g_bNewNbrs = 1;
  g_pdTempX = new double[3*m_nSpherocyls];
  g_pdTempY = new double[3*m_nSpherocyls];
  g_pdTempPhi = new double[3*m_nSpherocyls];
  g_pdXMoved = new double[3*m_nSpherocyls];
  g_pdYMoved = new double[3*m_nSpherocyls];
#endif
  
  // Stress, energy, & force data
  cudaHostAlloc((void **)&h_pfSE, 5*sizeof(float), 0);
  m_pfEnergy = h_pfSE+4;
  m_pfPxx = h_pfSE;
  m_pfPyy = h_pfSE+1;
  m_pfPxy = h_pfSE+2;
  m_pfPyx = h_pfSE+3;
  m_fP = 0;
  h_pdFx = new double[m_nSpherocyls];
  h_pdFy = new double[m_nSpherocyls];
  h_pdFt = new double[m_nSpherocyls];
  // GPU
  cudaMalloc((void**) &d_pfSE, 5*sizeof(float));
  cudaMalloc((void**) &d_pdFx, m_nSpherocyls*sizeof(double));
  cudaMalloc((void**) &d_pdFy, m_nSpherocyls*sizeof(double));
  cudaMalloc((void**) &d_pdFt, m_nSpherocyls*sizeof(double));
  m_nDeviceMem += 5*sizeof(float) + 3*m_nSpherocyls*sizeof(double);
 #if GOLD_FUNCS == 1
  g_pfSE = new float[5];
  g_pdFx = new double[m_nSpherocyls];
  g_pdFy = new double[m_nSpherocyls];
  g_pdFt = new double[m_nSpherocyls];
#endif

  // Cell & neighbor data
  h_pnCellID = new int[m_nSpherocyls];
  cudaMalloc((void**) &d_pnCellID, sizeof(int)*m_nSpherocyls);
  m_nDeviceMem += m_nSpherocyls*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnCellID = new int[m_nSpherocyls];
#endif
  configure_cells();
  
  h_pnNPP = new int[m_nSpherocyls];
  h_pnNbrList = new int[m_nSpherocyls*m_nMaxNbrs];
  cudaMalloc((void**) &d_pnNPP, sizeof(int)*m_nSpherocyls);
  cudaMalloc((void**) &d_pnNbrList, sizeof(int)*m_nSpherocyls*m_nMaxNbrs);
  m_nDeviceMem += m_nSpherocyls*(1+m_nMaxNbrs)*sizeof(int);
#if GOLD_FUNCS == 1
  g_pnNPP = new int[m_nSpherocyls];
  g_pnNbrList = new int[m_nSpherocyls*m_nMaxNbrs];
#endif

  set_kernel_configs();	
}

double Spherocyl_Box::calculate_packing()
{
  double dParticleArea = 0.0;
  for (int p = 0; p < m_nSpherocyls; p++)
    {
      dParticleArea += (4 * h_pdA[p] + D_PI * h_pdR[p]) * h_pdR[p];
    }
  return dParticleArea / (m_dL * m_dL);
}

// Creates the class
// See spherocyl_box.h for default values of parameters
Spherocyl_Box::Spherocyl_Box(int nSpherocyls, double dL, double dAspect, double dBidispersity, Config config, double dEpsilon, int nMaxPPC, int nMaxNbrs, Potential ePotential)
{
  assert(nSpherocyls > 0);
  m_nSpherocyls = nSpherocyls;
  assert(dL > 0.0);
  m_dL = dL;
  m_ePotential = ePotential;

  m_dEpsilon = dEpsilon;
  m_nMaxPPC = nMaxPPC;
  m_nMaxNbrs = nMaxNbrs;
  if (dBidispersity >= 1) {
    m_dRMax = 0.5*dBidispersity;
  }
  else {
    m_dRMax = 0.5;
  }
  m_dAMax = m_dRMax*dAspect;
  m_nDeviceMem = 0;

  // This allocates the coordinate data as page-locked memory, which
  //  transfers faster, since they are likely to be transferred often
  cudaHostAlloc((void**)&h_pdX, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdY, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdPhi, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdR, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdA, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pnMemID, nSpherocyls*sizeof(int), 0);
  m_dPacking = 0;
  // This initializes the arrays on the GPU
  cudaMalloc((void**) &d_pdX, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdY, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdPhi, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdR, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdA, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pnInitID, sizeof(int)*nSpherocyls);
  cudaMalloc((void**) &d_pnMemID, sizeof(int)*nSpherocyls);
  // Spherocylinders
  m_nDeviceMem += nSpherocyls*(5*sizeof(double) + 2*sizeof(int));
#if GOLD_FUNCS == 1
  g_pdX = new double[nSpherocyls];
  g_pdY = new double[nSpherocyls];
  g_pdPhi = new double[nSpherocyls];
  g_pdR = new double[nSpherocyls];
  g_pdA = new double[nSpherocyls];
  g_pnInitID = new int[nSpherocyls];
  g_pnMemID = new int[nSpherocyls];
#endif

  construct_defaults();
  cout << "Memory allocated on device (MB): " << (double)m_nDeviceMem / (1024.*1024.) << endl;
  /*
  switch (config) {
  case RANDOM:
    place_random_spherocyls(0,1,dBidispersity);
    break;
  case RANDOM_ALIGNED:
    place_random_spherocyls(0,0,dBidispersity);
    break;
  case ZERO_E:
    place_random_0e_spherocyls(0,1,dBidispersity);
    break;
  case ZERO_E_ALIGNED:
    place_random_0e_spherocyls(0,0,dBidispersity);
    break;
  case GRID:
    place_spherocyl_grid(0,1);
    break;
  case GRID_ALIGNED:
    place_spherocyl_grid(0,0);
    break;
  }
  */
  place_spherocyls(config,0,dBidispersity);
    
  m_dPacking = calculate_packing();
  cout << "Random spherocyls placed" << endl;
  //display(0,0,0,0);
  
}
// Create class with coordinate arrays provided
Spherocyl_Box::Spherocyl_Box(int nSpherocyls, double dL, double *pdX, 
			     double *pdY, double *pdPhi, double *pdR, 
			     double *pdA, double dEpsilon, int nMaxPPC, 
			     int nMaxNbrs, Potential ePotential)
{
  assert(nSpherocyls > 0);
  m_nSpherocyls = nSpherocyls;
  assert(dL > 0);
  m_dL = dL;
  m_ePotential = ePotential;

  m_dEpsilon = dEpsilon;
  m_nMaxPPC = nMaxPPC;
  m_nMaxNbrs = nMaxNbrs;

  // This allocates the coordinate data as page-locked memory, which 
  //  transfers faster, since they are likely to be transferred often
  cudaHostAlloc((void**)&h_pdX, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdY, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdPhi, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdR, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pdA, nSpherocyls*sizeof(double), 0);
  cudaHostAlloc((void**)&h_pnMemID, nSpherocyls*sizeof(int), 0);
  m_dRMax = 0.0;
  m_dAMax = 0.0;
  cout << "Loading positions" << endl;
  for (int p = 0; p < nSpherocyls; p++)
    {
	  cout << "loading p=" << p << endl;
      h_pdX[p] = pdX[p];
      h_pdY[p] = pdY[p];
      h_pdPhi[p] = pdPhi[p];
      h_pdR[p] = pdR[p];
      h_pdA[p] = pdA[p];
      h_pnMemID[p] = p;
      if (pdR[p] > m_dRMax)
	m_dRMax = pdR[p];
      if (pdA[p] > m_dAMax)
	m_dAMax = pdA[p];
      while (h_pdX[p] > dL)
	h_pdX[p] -= dL;
      while (h_pdX[p] < 0)
	h_pdX[p] += dL;
      while (h_pdY[p] > dL)
	h_pdY[p] -= dL;
      while (h_pdY[p] < 0)
	h_pdY[p] += dL;
    }
  m_dPacking = calculate_packing();
  cout << "Positions loaded" << endl;

  // This initializes the arrays on the GPU
  cudaMalloc((void**) &d_pdX, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdY, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdPhi, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdR, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pdA, sizeof(double)*nSpherocyls);
  cudaMalloc((void**) &d_pnInitID, sizeof(int)*nSpherocyls);
  cudaMalloc((void**) &d_pnMemID, sizeof(int)*nSpherocyls);
  // This copies the values to the GPU asynchronously, which allows the
  //  CPU to go on and process further instructions while the GPU copies.
  //  Only workes on page-locked memory (allocated with cudaHostAlloc)
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdR, h_pdR, sizeof(double)*nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdA, h_pdA, sizeof(double)*nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, d_pnMemID, sizeof(int)*nSpherocyls, cudaMemcpyDeviceToDevice);

  m_nDeviceMem += nSpherocyls*(5*sizeof(double)+2*sizeof(int));

#if GOLD_FUNCS == 1
  g_pdX = new double[nSpherocyls];
  g_pdY = new double[nSpherocyls];
  g_pdPhi = new double[nSpherocyls];
  g_pdR = new double[nSpherocyls];
  g_pdA = new double[nSpherocyls];
  g_pnInitID = new int[nSpherocyls];
  g_pnMemID = new int[nSpherocyls];
  for (int p = 0; p < nSpherocyls; p++)
    {
      g_pdX[p] = h_pdX[p];
      g_pdY[p] = h_pdY[p];
      g_pdPhi[p] = h_pdPhi[p];
      g_pdR[p] = h_pdR[p];
      g_pdA[p] = h_pdA[p];
      g_pnMemID[p] = h_pnMemID[p];
      g_pnInitID[p] = g_pnMemID[p];
    }
#endif

  cout << "Positions transfered to device" << endl;
  construct_defaults();
  cout << "Memory allocated on device (MB): " << (double)m_nDeviceMem / (1024.*1024.) << endl;
  // Get spheocyl coordinates from spherocyls

  cudaThreadSynchronize();
  //display(0,0,0,0);
}

//Cleans up arrays when class is destroyed
Spherocyl_Box::~Spherocyl_Box()
{
  // Host arrays
  cudaFreeHost(h_pdX);
  cudaFreeHost(h_pdY);
  cudaFreeHost(h_pdPhi);
  cudaFreeHost(h_pdR);
  cudaFreeHost(h_pdA);
  cudaFreeHost(h_pnMemID);
  cudaFreeHost(h_bNewNbrs);
  cudaFreeHost(h_pfSE);
  delete[] h_pdFx;
  delete[] h_pdFy;
  delete[] h_pdFt;
  delete[] h_pnCellID;
  delete[] h_pnPPC;
  delete[] h_pnCellList;
  delete[] h_pnAdjCells;
  delete[] h_pnNPP;
  delete[] h_pnNbrList;
  
  // Device arrays
  cudaFree(d_pdX);
  cudaFree(d_pdY);
  cudaFree(d_pdPhi);
  cudaFree(d_pdR);
  cudaFree(d_pdA);
  cudaFree(d_pdTempX);
  cudaFree(d_pdTempY);
  cudaFree(d_pdTempPhi);
  cudaFree(d_pnInitID);
  cudaFree(d_pnMemID);
  cudaFree(d_pdMOI);
  cudaFree(d_pdIsoC);
  cudaFree(d_pdXMoved);
  cudaFree(d_pdYMoved);
  cudaFree(d_bNewNbrs);
  cudaFree(d_pfSE);
  cudaFree(d_pdFx);
  cudaFree(d_pdFy);
  cudaFree(d_pdFt);
  cudaFree(d_pnCellID);
  cudaFree(d_pnPPC);
  cudaFree(d_pnCellList);
  cudaFree(d_pnAdjCells);
  cudaFree(d_pnNPP);
  cudaFree(d_pnNbrList);
#if GOLD_FUNCS == 1
  delete[] g_pdX;
  delete[] g_pdY;
  delete[] g_pdPhi;
  delete[] g_pdR;
  delete[] g_pdA;
  delete[] g_pdTempX;
  delete[] g_pdTempY;
  delete[] g_pdTempPhi;
  delete[] g_pdXMoved;
  delete[] g_pdYMoved;
  delete[] g_pnInitID;
  delete[] g_pnMemID;
  delete[] g_pfSE;
  delete[] g_pdFx;
  delete[] g_pdFy;
  delete[] g_pdFt;
  delete[] g_pnCellID;
  delete[] g_pnPPC;
  delete[] g_pnCellList;
  delete[] g_pnAdjCells;
  delete[] g_pnNPP;
  delete[] g_pnNbrList;
#endif 
}

// Display various info about the configuration which has been calculated
// Mostly used to make sure things are working right
void Spherocyl_Box::display(bool bParticles, bool bCells, bool bNeighbors, bool bStress)
{
  if (bParticles)
    {
      cudaMemcpyAsync(h_pdX, d_pdX, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdY, d_pdY, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdPhi, d_pdPhi, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdR, d_pdR, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pdA, d_pdA, sizeof(double)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_pnMemID, d_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
      checkCudaError("Display: copying particle data to host");
      
      cout << endl << "Box dimension: " << m_dL << endl;;
      for (int p = 0; p < m_nSpherocyls; p++)
	{
	  int m = h_pnMemID[p];
	  cout << "Particle " << p << " (" << m  << "): (" 
	       << h_pdX[m] << ", " << h_pdY[m] << ", " << h_pdPhi[m] 
	       << ") R = " << h_pdR[m] << " A = " << h_pdA[m] << endl;
	}
    }
  if (bCells)
    {
      cudaMemcpy(h_pnPPC, d_pnPPC, sizeof(int)*m_nCells, cudaMemcpyDeviceToHost); 
      cudaMemcpy(h_pnCellList, d_pnCellList, sizeof(int)*m_nCells*m_nMaxPPC, cudaMemcpyDeviceToHost);
      checkCudaError("Display: copying cell data to host");

      cout << endl;
      int nTotal = 0;
      int nMaxPPC = 0;
      for (int c = 0; c < m_nCells; c++)
	{
	  nTotal += h_pnPPC[c];
	  nMaxPPC = max(nMaxPPC, h_pnPPC[c]);
	  cout << "Cell " << c << ": " << h_pnPPC[c] << " particles\n";
	  for (int p = 0; p < h_pnPPC[c]; p++)
	    {
	      cout << h_pnCellList[c*m_nMaxPPC + p] << " ";
	    }
	  cout << endl;
	}
      cout << "Total particles in cells: " << nTotal << endl;
      cout << "Maximum particles in any cell: " << nMaxPPC << endl;
    }
  if (bNeighbors)
    {
      cudaMemcpy(h_pnNPP, d_pnNPP, sizeof(int)*m_nSpherocyls, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pnNbrList, d_pnNbrList, sizeof(int)*m_nSpherocyls*m_nMaxNbrs, cudaMemcpyDeviceToHost);
      checkCudaError("Display: copying neighbor data to host");

      cout << endl;
      int nMaxNPP = 0;
      for (int p = 0; p < m_nSpherocyls; p++)
	{
	  nMaxNPP = max(nMaxNPP, h_pnNPP[p]);
	  cout << "Particle " << p << ": " << h_pnNPP[p] << " neighbors\n";
	  for (int n = 0; n < h_pnNPP[p]; n++)
	    {
	      cout << h_pnNbrList[n*m_nSpherocyls + p] << " ";
	    }
	  cout << endl;
	}
      cout << "Maximum neighbors of any particle: " << nMaxNPP << endl;
    }
  if (bStress)
    {
      cudaMemcpyAsync(h_pfSE, d_pfSE, 5*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFx, d_pdFx, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFy, d_pdFy, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_pdFt, d_pdFt, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
      m_fP = 0.5 * (*m_pfPxx + *m_pfPyy);
      cout << endl;
      for (int p = 0; p < m_nSpherocyls; p++)
	{
	  cout << "Particle " << p << ":  (" << h_pdFx[p] << ", " 
	       << h_pdFy[p] << ", " << h_pdFt[p] << ")\n";
	}
      cout << endl << "Energy: " << *m_pfEnergy << endl;
      cout << "Pxx: " << *m_pfPxx << endl;
      cout << "Pyy: " << *m_pfPyy << endl;
      cout << "Total P: " << m_fP << endl;
      cout << "Pxy: " << *m_pfPxy << endl;
      cout << "Pyx: " << *m_pfPyx << endl;
    }
}

bool Spherocyl_Box::check_for_contacts(int nIndex, double dTol)
{
  double dS = 2 * ( m_dAMax + m_dRMax );

  double dX = h_pdX[nIndex];
  double dY = h_pdY[nIndex];
  for (int p = 0; p < nIndex; p++) {
    //cout << "Checking: " << nIndex << " vs " << p << endl;
    double dXj = h_pdX[p];
    double dYj = h_pdY[p];

    double dDelX = dX - dXj;
    double dDelY = dY - dYj;
    dDelX += m_dL * ((dDelX < -0.5*m_dL) - (dDelX > 0.5*m_dL));
    dDelY += m_dL * ((dDelY < -0.5*m_dL) - (dDelY > 0.5*m_dL));
    dDelX += m_dGamma * dDelY;
    double dDelRSqr = dDelX * dDelX + dDelY * dDelY;
    if (dDelRSqr < dS*dS) {
	double dPhi = h_pdPhi[nIndex];
	double dR = h_pdR[nIndex];
	double dA = h_pdA[nIndex];
    
	double dDeltaX = dX - dXj;
	double dDeltaY = dY - dYj;
	double dPhiB = h_pdPhi[p];
	double dSigma = dR + h_pdR[p];
	double dB = h_pdA[p];
	// Make sure we take the closest distance considering boundary conditions
	dDeltaX += m_dL * ((dDeltaX < -0.5*m_dL) - (dDeltaX > 0.5*m_dL));
	dDeltaY += m_dL * ((dDeltaY < -0.5*m_dL) - (dDeltaY > 0.5*m_dL));
	// Transform from shear coordinates to lab coordinates
	dDeltaX += m_dGamma * dDeltaX;
	
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
	if (dDSqr < (1-dTol)*dSigma*dSigma || dDSqr != dDSqr)
	  return 1;
    }
  }

  return 0;
}

bool Spherocyl_Box::check_for_crosses(int nIndex, double dEpsilon)
{
  double dX = h_pdX[nIndex];
  double dY = h_pdY[nIndex];
  for (int p = 0; p < nIndex; p++) {
    //cout << "Checking: " << nIndex << " vs " << p << endl;
    double dXj = h_pdX[p];
    double dYj = h_pdY[p];
    double dPhi = h_pdPhi[nIndex];
    double dR = h_pdR[nIndex];
    double dA = h_pdA[nIndex];
    
    double dDeltaX = dX - dXj;
    double dDeltaY = dY - dYj;
    double dPhiB = h_pdPhi[p];
    double dSigma = dR + h_pdR[p];
    double dB = h_pdA[p];
    // Make sure we take the closest distance considering boundary conditions
    dDeltaX += m_dL * ((dDeltaX < -0.5*m_dL) - (dDeltaX > 0.5*m_dL));
    dDeltaY += m_dL * ((dDeltaY < -0.5*m_dL) - (dDeltaY > 0.5*m_dL));
    // Transform from shear coordinates to lab coordinates
    dDeltaX += m_dGamma * dDeltaX;
    
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
    if (dDSqr < dEpsilon*dSigma*dSigma || dDSqr != dDSqr)
      return 1;
  }

  return 0;
}

void Spherocyl_Box::place_random_0e_spherocyls(int seed, bool bRandAngle, double dBidispersity)
{
  srand(time(0) + seed);

  double dAspect = m_dAMax / m_dRMax;
  double dR = 0.5;
  double dA = dAspect * dR;
  double dRDiff = (dBidispersity-1) * dR;
  double dADiff = (dBidispersity-1) * dA;
  for (int p = 0; p < m_nSpherocyls; p++) {
    h_pdR[p] = dR + (p%2)*dRDiff;
    h_pdA[p] = dA + (p%2)*dADiff;
    h_pnMemID[p] = p;
  }
  cudaMemcpy(d_pdR, h_pdR, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdA, h_pdA, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  h_pdX[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdY[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  if (bRandAngle)
    h_pdPhi[0] = 2*D_PI * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  else
    h_pdPhi[0] = 0;

  for (int p = 1; p < m_nSpherocyls; p++) {
    bool bContact = 1;
    int nTries = 0;

    while (bContact) {
      h_pdX[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdY[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      if (bRandAngle)
	h_pdPhi[p] = 2*D_PI * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      else
	h_pdPhi[p] = 0;
	

      bContact = check_for_contacts(p);
      nTries += 1;
    }
    cout << "Spherocyl " << p << " placed in " << nTries << " attempts." << endl;
  }
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  cout << "Data copied to device" << endl;

}

void Spherocyl_Box::place_random_spherocyls(int seed, bool bRandAngle, double dBidispersity)
{
  srand(time(0) + seed);

  double dAspect = m_dAMax / m_dRMax;
  double dR = 0.5;
  double dA = dAspect * dR;
  double dRDiff = (dBidispersity-1) * dR;
  double dADiff = (dBidispersity-1) * dA;
  for (int p = 0; p < m_nSpherocyls; p++) {
    h_pdR[p] = dR + (p%2)*dRDiff;
    h_pdA[p] = dA + (p%2)*dADiff;
    h_pnMemID[p] = p;
  }
  cudaMemcpy(d_pdR, h_pdR, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdA, h_pdA, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  h_pdX[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdY[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  if (bRandAngle)
    h_pdPhi[0] = 2*D_PI * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  else
    h_pdPhi[0] = 0;

  for (int p = 1; p < m_nSpherocyls; p++) {
    bool bContact = 1;
    int nTries = 0;

    while (bContact) {
      h_pdX[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdY[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      if (bRandAngle)
	h_pdPhi[p] = 2*D_PI * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      else
	h_pdPhi[p] = 0;
	

      bContact = check_for_crosses(p);
      nTries += 1;
    }
    cout << "Spherocylinder " << p << " placed in " << nTries << " attempts." << endl;
  }
  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  cout << "Data copied to device" << endl;

}


void Spherocyl_Box::place_spherocyl_grid(int seed, bool bRandAngle) 
{
  double dAspect = (m_dAMax + m_dRMax) / m_dRMax;
  double dCols = sqrt(m_nSpherocyls/dAspect);
  int nRows = int(dAspect*dCols);
  int nCols = int(dCols);
  double dWidth = m_dL/nCols;
  double dHeight = m_dL/nRows;
  if (dWidth < 2*(m_dAMax+m_dRMax) || dHeight < 2*m_dRMax) {
    cerr << "Error: Spherocylinders will not fit into square grid, change the number or size of box" << endl;
    exit(1);
  }

  srand(time(0) + seed);
  
  for (int p = 0; p < m_nSpherocyls; p++) {
    h_pdR[p] = m_dRMax;
    h_pdA[p] = m_dAMax;
    h_pnMemID[p] = p;
  }
  cudaMemcpy(d_pdR, h_pdR, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdA, h_pdA, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  
  double dXOffset = m_dAMax + m_dRMax;
  double dRWidth = dWidth - 2*dXOffset;
  double dYOffset = m_dRMax;
  double dRHeight = dHeight - 2*dYOffset;

  h_pdX[0] = dXOffset + dRWidth* static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  h_pdY[0] = dYOffset + dRHeight * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  if (bRandAngle)
    h_pdPhi[0] = 2*D_PI * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
  else
    h_pdPhi[0] = 0;

  for (int p = 1; p < m_nSpherocyls; p++) {
    bool bContact = 1;
    int nTries = 0;
    int nR = p / nCols;
    int nC = p % nCols;

    while (bContact) {
      h_pdX[p] = nC*dWidth + dXOffset + dRWidth * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      h_pdY[p] = nR*dHeight + dYOffset + dRHeight * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      if (bRandAngle)
	h_pdPhi[p] = 2*D_PI * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
      else
	h_pdPhi[p] = 0;
	

      bContact = check_for_crosses(p);
      nTries += 1;
    }
    cout << "Spherocylinder " << p << " placed in " << nTries << " attempts." << endl;
  }

  cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  cout << "Data copied to device" << endl;

}

void Spherocyl_Box::place_spherocyls(Config config, int seed, double dBidispersity)
{
  //TODO: implement grid placement with bidisperse particles
  if (config.type == GRID) {
    bool bRandAngle = bool(config.maxAngle - config.minAngle);
    place_spherocyl_grid(seed, bRandAngle);
  }
  else {
     srand(time(0) + seed);

     double dAspect = m_dAMax / m_dRMax;
     double dR = 0.5;
     double dA = dAspect * dR;
     double dRDiff = (dBidispersity-1) * dR;
     double dADiff = (dBidispersity-1) * dA;
     for (int p = 0; p < m_nSpherocyls; p++) {
       h_pdR[p] = dR + (p%2)*dRDiff;
       h_pdA[p] = dA + (p%2)*dADiff;
       h_pnMemID[p] = p;
     }
     cudaMemcpy(d_pdR, h_pdR, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaMemcpyAsync(d_pdA, h_pdA, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaMemcpyAsync(d_pnInitID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaMemcpyAsync(d_pnMemID, h_pnMemID, sizeof(int)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaThreadSynchronize();
     
     h_pdX[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
     h_pdY[0] = m_dL * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
     h_pdPhi[0] = config.minAngle + (config.maxAngle - config.minAngle) * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
     
     for (int p = 1; p < m_nSpherocyls; p++) {
       bool bContact = 1;
       int nTries = 0;
       
       while (bContact) {
	 h_pdX[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	 h_pdY[p] = m_dL * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	 h_pdPhi[p] = config.minAngle + (config.maxAngle - config.minAngle) * static_cast<double>(rand())/static_cast<double>(RAND_MAX);
	 
	 if (config.overlap >= 1)
	   bContact = check_for_crosses(p);
	 else
	   bContact = check_for_contacts(p, config.overlap);
	 nTries += 1;
       }
       cout << "Spherocylinder " << p << " placed in " << nTries << " attempts." << endl;
     }
     cudaMemcpyAsync(d_pdX, h_pdX, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaMemcpyAsync(d_pdY, h_pdY, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaMemcpyAsync(d_pdPhi, h_pdPhi, sizeof(double)*m_nSpherocyls, cudaMemcpyHostToDevice);
     cudaThreadSynchronize();
     cout << "Data copied to device" << endl;
  }
}
