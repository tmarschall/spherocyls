#include <vector>
#include <string>
#include <assert.h>
#include "spherocyl_box.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <limits>

const double D_PI = 3.14159265358979;
using std::cout;
using std::endl;

void SpherocylBox::set_defaults()
{
  m_dLx = 0.0;
  m_dLy = 0.0;
  m_dGamma = 0.0;
  m_dGammaTotal = 0.0;
  m_dStrainRate = 0.0;
  m_dResizeRate = 0.0;
  m_dStep = 0.05;
  m_dPosSaveStep = 0.01;
  m_dSESaveStep = 1e-5;
  m_dPacking = 0.0;
  m_dAmax = 0.0;
  m_dRmax = 0.0;

  m_dPadding = 0.0;
  m_dMaxMoved = 0.0;

  m_dEnergy = 0.0;
  m_dPxx = 0.0;
  m_dPyy = 0.0;
  m_dPxy = 0.0;

  m_psParticles = 0;
  m_psTempParticles = 0;
  m_nParticles = 0;
  m_nCellCols = 1;
  m_nCellRows = 1;
  m_nCells = 1;
  m_pvCells = new std::vector<int>[1];
  m_dCellWidth = 0;
  m_dCellHeight = 0;
  m_pnCellNeighbors = 0;
  m_pvNeighbors = 0;

  m_strOutputDir = ".";
  m_strSEOutput = "sp_se.dat";
}

SpherocylBox::SpherocylBox()
{
  set_defaults();
}
SpherocylBox::SpherocylBox(double dL, int nParticles)
{
  set_defaults();
  m_dLx = dL;
  m_dLy = dL;
  m_nParticles = nParticles;
  delete[] m_psParticles;
  m_psParticles = new Spherocyl[nParticles];
  delete[] m_psTempParticles;
  m_psTempParticles = new Spherocyl[nParticles];
  delete[] m_pvNeighbors;
  m_pvNeighbors = new std::vector<int>[nParticles];
}
SpherocylBox::SpherocylBox(double dL, int nParticles, double dPadding)
{
  set_defaults();
  m_dLx = dL;
  m_dLy = dL;
  m_nParticles = nParticles;
  m_dPadding = dPadding;
  delete[] m_psParticles;
  m_psParticles = new Spherocyl[nParticles];
  delete[] m_psTempParticles;
  m_psTempParticles = new Spherocyl[nParticles];
  delete[] m_pvNeighbors;
  m_pvNeighbors = new std::vector<int>[nParticles];
}
SpherocylBox::SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double dA, double dR)
{
  set_defaults();
  m_dLx = dL;
  m_dLy = dL;
  m_nParticles = nParticles;
  delete[] m_psParticles;
  m_psParticles = new Spherocyl[nParticles];
  delete[] m_psTempParticles;
  m_psTempParticles = new Spherocyl[nParticles];
  delete[] m_pvNeighbors;
  m_pvNeighbors = new std::vector<int>[nParticles];
  m_dAmax = dA;
  m_dRmax = dR;
  for (int p = 0; p < nParticles; p++) {
    set_particle(p, pdX[p], pdY[p], pdPhi[p], dA, dR);
  }
  m_dPacking = calculate_packing_fraction();
  m_dPadding = 0.0;
  configure_cells();
  find_neighbors();
}
SpherocylBox::SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double dA, double dR, double dPadding)
{
  set_defaults();
  m_dLx = dL;
  m_dLy = dL;
  m_nParticles = nParticles;
  delete[] m_psParticles;
  m_psParticles = new Spherocyl[nParticles];
  delete[] m_psTempParticles;
  m_psTempParticles = new Spherocyl[nParticles];
  delete[] m_pvNeighbors;
  m_pvNeighbors = new std::vector<int>[nParticles];
  m_dAmax = dA;
  m_dRmax = dR;
  for (int p = 0; p < nParticles; p++) {
    set_particle(p, pdX[p], pdY[p], pdPhi[p], dA, dR);
  }
  m_dPacking = calculate_packing_fraction();
  m_dPadding = dPadding;
  configure_cells();
  find_neighbors();
}
SpherocylBox::SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double *pdA, double *pdR)
{
  set_defaults();
  m_dLx = dL;
  m_dLy = dL;
  m_nParticles = nParticles;
  delete[] m_psParticles;
  m_psParticles = new Spherocyl[nParticles];
  delete[] m_psTempParticles;
  m_psTempParticles = new Spherocyl[nParticles];
  delete[] m_pvNeighbors;
  m_pvNeighbors = new std::vector<int>[nParticles];
  m_dAmax = 0.0;
  m_dRmax = 0.0;
  for (int p = 0; p < nParticles; p++) {
    set_particle(p, pdX[p], pdY[p], pdPhi[p], pdA[p], pdR[p]);
  }
  m_dPacking = calculate_packing_fraction();
  m_dPadding = 0.0;
  configure_cells();
  find_neighbors();
}
SpherocylBox::SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double *pdA, double *pdR, double dPadding)
{
  set_defaults();
  m_dLx = dL;
  m_dLy = dL;
  m_nParticles = nParticles;
  delete[] m_psParticles;
  m_psParticles = new Spherocyl[nParticles];
  delete[] m_psTempParticles;
  m_psTempParticles = new Spherocyl[nParticles];
  delete[] m_pvNeighbors;
  m_pvNeighbors = new std::vector<int>[nParticles];
  m_dAmax = 0.0;
  m_dRmax = 0.0;
  for (int p = 0; p < nParticles; p++) {
    set_particle(p, pdX[p], pdY[p], pdPhi[p], pdA[p], pdR[p]);
  }
  m_dPacking = calculate_packing_fraction();
  m_dPadding = dPadding;
  configure_cells();
  find_neighbors();
}
SpherocylBox::~SpherocylBox()
{
  if (m_nCells > 1) {
    for (int c = 0; c < m_nCells; c++) {
  	  delete[] m_pnCellNeighbors[c];
    }
  }
  delete[] m_pnCellNeighbors;
  delete[] m_psParticles;
  delete[] m_psTempParticles;
  delete[] m_pvCells;
  delete[] m_pvNeighbors;
}

void SpherocylBox::set_particle(int p, double dX, double dY, double dPhi, double dA, double dR)
{
  assert(p < m_nParticles);
  assert(dA >= 0.0);
  assert(dR > 0.0);
  m_psParticles[p].m_dX = dX;
  m_psParticles[p].m_dY = dY;
  m_psParticles[p].m_dPhi = dPhi;
  m_psParticles[p].m_dA = dA;
  m_psParticles[p].m_dR = dR;
  double dAlpha = dA/dR;
  double dC = (3*D_PI + 24*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha);
  double dArea = dR*(D_PI*dR + 4*dA);
  m_psParticles[p].m_dI = dR*dR*dR*dR*dC/(6*dArea);
  m_psParticles[p].m_dRC = (8*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha)/dC;
  
  while (m_psParticles[p].m_dX > m_dLx)
    m_psParticles[p].m_dX -= m_dLx;
  while (m_psParticles[p].m_dX < 0.0)
    m_psParticles[p].m_dX += m_dLx;
  while (m_psParticles[p].m_dY > m_dLy)
    m_psParticles[p].m_dY -= m_dLy;
  while (m_psParticles[p].m_dY < 0.0)
    m_psParticles[p].m_dY += m_dLy;
  while (m_psParticles[p].m_dPhi > D_PI)
    m_psParticles[p].m_dPhi -= 2*D_PI;
  while (m_psParticles[p].m_dPhi < -D_PI)
    m_psParticles[p].m_dPhi += 2*D_PI;
  if (m_psParticles[p].m_dA > m_dAmax)
  	m_dAmax = m_psParticles[p].m_dA;
  if (m_psParticles[p].m_dR > m_dRmax)
  	m_dRmax = m_psParticles[p].m_dR;
    
}

void SpherocylBox::configure_cells()
{
  assert(m_dRmax > 0.0);
  assert(m_dAmax >= 0.0);
  
  double dWmin = 2.24 * (m_dRmax + m_dAmax) + m_dPadding;
  double dHmin = 2 * (m_dRmax + m_dAmax) + m_dPadding;
  
  m_nCellRows = std::max(static_cast<int>(m_dLy / dHmin), 1);
  m_nCellCols = std::max(static_cast<int>(m_dLx / dWmin), 1);
  m_nCells = m_nCellRows * m_nCellCols;
  cout << "Cells: " << m_nCells << ": " << m_nCellRows << " x " << m_nCellCols << endl;

  m_dCellWidth = m_dLx / m_nCellCols;
  m_dCellHeight = m_dLy / m_nCellRows;
  cout << "Cell dimensions: " << m_dCellWidth << " x " << m_dCellHeight << endl;
  
  if (m_dPadding == 0)
  	m_dPadding = std::max(m_dCellWidth - dWmin, m_dCellHeight - dHmin);
  
  delete[] m_pvCells;
  m_pvCells = new std::vector<int>[m_nCells];
  delete[] m_pnCellNeighbors;
  m_pnCellNeighbors = new int*[m_nCells];
  for (int c = 0; c < m_nCells; c++) {
  	m_pnCellNeighbors[c] = new int[8];
  	
  	int nRow = c / m_nCellCols; 
    int nCol = c % m_nCellCols;

	int nAdjCol1 = (nCol + 1) % m_nCellCols;
    int nAdjCol2 = (m_nCellCols + nCol - 1) % m_nCellCols;
    m_pnCellNeighbors[c][0] = nRow * m_nCellCols + nAdjCol1;
    m_pnCellNeighbors[c][1] = nRow * m_nCellCols + nAdjCol2;

    int nAdjRow = (nRow + 1) % m_nCellRows;
    m_pnCellNeighbors[c][2] = nAdjRow * m_nCellCols + nCol;
    m_pnCellNeighbors[c][3] = nAdjRow * m_nCellCols + nAdjCol1;
    m_pnCellNeighbors[c][4] = nAdjRow * m_nCellCols + nAdjCol2;
      
    nAdjRow = (m_nCellRows + nRow - 1) % m_nCellRows;
    m_pnCellNeighbors[c][5] = nAdjRow * m_nCellCols + nCol;
    m_pnCellNeighbors[c][6] = nAdjRow * m_nCellCols + nAdjCol1;
    m_pnCellNeighbors[c][7] = nAdjRow * m_nCellCols + nAdjCol2;
  }
}

void SpherocylBox::find_cells()
{
	//cout << "finding cells" << endl;
	for (int c = 0; c < m_nCells; c++) {
	  m_pvCells[c].clear();
	}
	//cout << "cells cleared" << endl;
	for (int p = 0; p < m_nParticles; p++) {
	  //cout << m_psParticles[p].m_dX << " " << m_psParticles[p].m_dY << endl;
	  while (m_psParticles[p].m_dX > m_dLx)
	    m_psParticles[p].m_dX -= m_dLx;
  	  while (m_psParticles[p].m_dX < 0.0)
	    m_psParticles[p].m_dX += m_dLx;
  	  while (m_psParticles[p].m_dY > m_dLy)
	    m_psParticles[p].m_dY -= m_dLy;
  	  while (m_psParticles[p].m_dY < 0.0)
	    m_psParticles[p].m_dY += m_dLy;
  	  while (m_psParticles[p].m_dPhi > D_PI)
	    m_psParticles[p].m_dPhi -= 2*D_PI;
	  while (m_psParticles[p].m_dPhi < -D_PI)
	    m_psParticles[p].m_dPhi += 2*D_PI;
    //cout << m_psParticles[p].m_dX << " " << m_psParticles[p].m_dY << endl;
    
      int nCol = (int)(m_psParticles[p].m_dX / m_dCellWidth);
      int nRow = (int)(m_psParticles[p].m_dY / m_dCellHeight); 
      int nCellID = nCol + nRow * m_nCellCols;
      //cout << "cell id " << nCellID << " (" << m_nCells << "): " << nRow << " " << nCol << endl;
	  m_psParticles[p].m_nCell = nCellID;
	  m_pvCells[nCellID].push_back(p); 
	}
	//cout << " done" << endl;
}

bool SpherocylBox::check_neighbors(int p, int q)
{
  double dSigma = m_psParticles[p].m_dR + m_psParticles[p].m_dA + m_psParticles[q].m_dR + m_psParticles[q].m_dA;
  double dY = m_psParticles[p].m_dY - m_psParticles[q].m_dY;
  dY += m_dLy * ((dY < -0.5 * m_dLy) - (dY > 0.5 * m_dLy));
  if (fabs(dY) < dSigma + m_dPadding) {
  	double dX = m_psParticles[p].m_dX - m_psParticles[q].m_dX;
  	dX += m_dLx * ((dX < -0.5 * m_dLx) - (dX > 0.5 * m_dLx));
	double dXprime = dX + 0.5 * dY;  // dX at maximum value of gamma
  	dX += m_dGamma * dY;
  	if (fabs(dX) < dSigma + m_dPadding || fabs(dXprime) < dSigma + m_dPadding) {
  	  return true;
  	}
  }
  return false;
}

void SpherocylBox::find_neighbors()
{
  //cout << " finding neighbors" << endl;
  find_cells();

  for (int p = 0; p < m_nParticles; p++) {
    m_pvNeighbors[p].clear();
    m_psParticles[p].m_dXMoved = 0.0;
    m_psParticles[p].m_dYMoved = 0.0;
  }
  for (int p = 0; p < m_nParticles-1; p++) {
    int nCellID = m_psParticles[p].m_nCell;
    for (int np = 0; np < m_pvCells[nCellID].size(); np++) {
      int q = m_pvCells[nCellID][np];
  	  if (q > p) {
  	    if (check_neighbors(p, q)) {
  	      m_pvNeighbors[p].push_back(q);
  	  	  //m_pvNeighbors[q].puch_back(p);
  	    }
  	  }
    }
    for (int nc = 0; nc < 8; nc++) {
      int nNCellID = m_pnCellNeighbors[nCellID][nc];
  	  for (int np = 0; np < m_pvCells[nNCellID].size(); np++) {
  	    int q = m_pvCells[nNCellID][np];
  	    if (q > p) {
  	      if (check_neighbors(p, q)) {
  	        m_pvNeighbors[p].push_back(q);
  	  	  	//m_pvNeighbors[q].push_back(p);
  	  	  }
  	  	}
  	  }
  	}   	       
  }
  //cout << "done" << endl;
}

double SpherocylBox::calculate_packing_fraction()
{
  double dArea = m_dLx * m_dLy;
  double dFilled = 0.0;
  for (int p = 0; p < m_nParticles; p++) {
  	Spherocyl *s = &m_psParticles[p];
  	dFilled += (D_PI*(s->m_dR) + 4.0*(s->m_dA))*(s->m_dR);
  }
  return dFilled / dArea;
}

void SpherocylBox::display(bool bParticles, bool bCells)
{
  cout << "Dimensions: " << m_dLx << " x " << m_dLy << endl;
  cout << "Particles: " << m_nParticles << endl;
  cout << "Packing fraction: " << m_dPacking << endl;
  cout << "Gamma: " << m_dGamma << " (" << m_dGammaTotal << ")" << endl;
  cout << "Strain rate: " << m_dStrainRate << endl;
  cout << "Integration step size: " << m_dStep << endl;
  cout << "\nEnergy: " << m_dEnergy << endl;
  cout << "Pxx: " << m_dPxx << endl;
  cout << "Pyy: " << m_dPyy << endl;
  cout << "Pxy: " << m_dPxy << endl;
  if (bParticles) {
  	cout << "\nParticles: (x, y, phi), (A, R), (I, RC) - (Fx, Fy, Tau) - (X_moved, Y_moved)" << endl; 
    for (int p = 0; p < m_nParticles; p++) {
    	Spherocyl *s = &m_psParticles[p];
    	cout << p << ": (" << s->m_dX << ", " << s->m_dY << ", " << s->m_dPhi << "), (" << s->m_dA 
	     << ", " << s->m_dR << "), (" <<  s->m_dI << ", " << s->m_dRC << ") - (" << s->m_dFx << ", " 
	     << s->m_dFy << ", " << s->m_dTau << ") - (" << s->m_dXMoved << ", " << s->m_dYMoved << ")\n";
    	cout << "Cell: " << s->m_nCell << " Neighbors: ";
    	for (int np = 0; np < m_pvNeighbors[p].size(); np++) {
    		cout << m_pvNeighbors[p][np] << " ";
    	}
    	cout << endl;
    }
  }
  if (bCells) {
  	cout << "\nCells:" << endl;
  	for (int c = 0; c < m_nCells; c++) {
  		cout << c << ":\t";
  		for (int cp = 0; cp < m_pvCells[c].size(); cp++) {
  			cout << m_pvCells[c][cp] << " ";
  		}
  		cout << endl << "Neighbors: ";
  		for (int nc = 0; nc < 8; nc++) {
  			cout << m_pnCellNeighbors[c][nc] << " ";
  		}
  		cout << endl;
  	} 
  }
}

void SpherocylBox::save_positions(std::string strFile)
{
	std::string strPath = m_strOutputDir + "/" + strFile;
	std::ofstream outf(strPath.c_str());
	Spherocyl *s = m_psParticles;
	outf.precision(14);
	outf << m_nParticles << " " << m_dLx << " " << m_dPacking << " " << m_dGamma << " " 
	     << m_dGammaTotal << " " << m_dStrainRate << " " << m_dStep << endl;
	for (int p = 0; p < m_nParticles; p++) {
		s = m_psParticles+p;
		outf << s->m_dX << " " << s->m_dY << " " << s->m_dPhi << " " << s->m_dR << " " << s->m_dA << endl;
	}
	outf.close();
}

void SpherocylBox::calc_se_force_pair(int p, int q)
{
	double dDeltaX = m_psParticles[p].m_dX - m_psParticles[q].m_dX;
	double dDeltaY = m_psParticles[p].m_dY - m_psParticles[q].m_dY;
	double dPhiP = m_psParticles[p].m_dPhi;
	double dPhiQ = m_psParticles[q].m_dPhi;
	double dSigma = m_psParticles[p].m_dR + m_psParticles[q].m_dR;
	double dAP = m_psParticles[p].m_dA;
	double dAQ = m_psParticles[q].m_dA;
	  
	// Make sure we take the closest distance considering boundary conditions
	dDeltaX += m_dLx * ((dDeltaX < -0.5*m_dLx) - (dDeltaX > 0.5*m_dLx));
	dDeltaY += m_dLy * ((dDeltaY < -0.5*m_dLy) - (dDeltaY > 0.5*m_dLy));
	// Transform from shear coordinates to lab coordinates
	dDeltaX += m_dGamma * dDeltaY;
	  
	double nxA = dAP * cos(dPhiP);
	double nyA = dAP * sin(dPhiP);
	double nxB = dAQ * cos(dPhiQ);
	double nyB = dAQ * sin(dPhiQ);

	double a = dAP * dAP;
	double b = -(nxA * nxB + nyA * nyB);
	double c = dAQ * dAQ;
	double d = nxA * dDeltaX + nyA * dDeltaY;
	double e = -nxB * dDeltaX - nyB * dDeltaY;
	double delta = a * c - b * b;

	double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	double s = -(b*t+d)/a;
	double sarg = fabs(s);
	s = fmin( fmax(s,-1.), 1. );
	if (sarg > 1) 
		t = fmin( fmax( -(b*s+e)/a, -1.), 1.);
	  
	// Check if they overlap and calculate forces
	double dDx = dDeltaX + s*nxA - t*nxB;
	double dDy = dDeltaY + s*nyA - t*nyB;
	double dDSqr = dDx * dDx + dDy * dDy;
	if (dDSqr < dSigma*dSigma)
	{
	    double dDij = sqrt(dDSqr);
	    double dDVij = (1.0 - dDij / dSigma) / dSigma;

	    double dFx = dDx * dDVij / dDij;
	    double dFy = dDy * dDVij / dDij;
		double dTauP = s*nxA * dFy - s*nyA * dFx;
		double dTauQ = t*nyB * dFx - t*nxB * dFy;
		//double dMoiP = 4.0 * dAP * dAP / 12.0;
		//double dMoiQ = 4.0 * dAQ * dAQ / 12.0;
		
		m_psParticles[p].m_dFx += dFx;
		m_psParticles[p].m_dFy += dFy;
		m_psParticles[p].m_dTau += dTauP;
		m_psParticles[q].m_dFx -= dFx;
		m_psParticles[q].m_dFy -= dFy;
		m_psParticles[q].m_dTau += dTauQ;
		
		m_dEnergy += dDVij * dSigma * (1.0 - dDij / dSigma) / (2.0 * m_nParticles);
		m_dPxx += dFx * dDx / (m_dLx * m_dLy);
		m_dPyy += dFy * dDy / (m_dLx * m_dLy);
		m_dPxy += dFx * dDy / (m_dLx * m_dLy);
		//cout << "contact between particles " << p << " and " << q << " with forces: ";
		//cout << dFx << " " << dFy << " " << dTauP / dMoiP << " " << dTauQ / dMoiQ << endl;
	}
}

void SpherocylBox::calc_se()
{
	for (int i = 0; i < m_nParticles; i++) {
      	m_psParticles[i].m_dFx = 0.0;
      	m_psParticles[i].m_dFy = 0.0;
      	m_psParticles[i].m_dTau = 0.0;
    }
    m_dEnergy = 0.0;
    m_dPxx = 0.0;
    m_dPyy = 0.0;
    m_dPxy = 0.0;

	for (int p = 0; p < m_nParticles; p++) {
    	for (int j = 0; j < m_pvNeighbors[p].size(); j++)
    	{
      		int q = m_pvNeighbors[p][j];
      		if (q > p)
				calc_se_force_pair(p, q);
    	}
  	}
}

void SpherocylBox::set_back_gamma()
{
  //cout << "setting back gamma" << endl;
  for (int i = 0; i < m_nParticles; i++)  //set all paricles to lab frame
    {
      if (m_psParticles[i].m_dY < 0)
	m_psParticles[i].m_dY += m_dLy;
      else if (m_psParticles[i].m_dY > m_dLy)
	m_psParticles[i].m_dY -= m_dLy;

      m_psParticles[i].m_dX += m_psParticles[i].m_dY; //gamma' = gamma - 1
      if (m_psParticles[i].m_dX < 0)
	m_psParticles[i].m_dX += m_dLx;
      else {
	while (m_psParticles[i].m_dX > m_dLx)  //particles outside the box sent back through periodic BC
	  m_psParticles[i].m_dX -= m_dLx;
      }	
    }
  m_dGamma -= 1.0;
  m_dGammaTotal = int(m_dGammaTotal + 1) + m_dGamma;
  find_neighbors();
}

bool SpherocylBox::check_particle_contact(int p, int q)
{
	//cout << "with: " << q << ": " << m_psParticles[q].m_dX << " " << m_psParticles[q].m_dY << " " << m_psParticles[q].m_dPhi << endl;
	double dDeltaX = m_psParticles[p].m_dX - m_psParticles[q].m_dX;
	double dDeltaY = m_psParticles[p].m_dY - m_psParticles[q].m_dY;
	double dPhiP = m_psParticles[p].m_dPhi;
	double dPhiQ = m_psParticles[q].m_dPhi;
	double dSigma = m_psParticles[p].m_dR + m_psParticles[q].m_dR;
	double dAP = m_psParticles[p].m_dA;
	double dAQ = m_psParticles[q].m_dA;
	  
	// Make sure we take the closest distance considering boundary conditions
	dDeltaX += m_dLx * ((dDeltaX < -0.5*m_dLx) - (dDeltaX > 0.5*m_dLx));
	dDeltaY += m_dLy * ((dDeltaY < -0.5*m_dLy) - (dDeltaY > 0.5*m_dLy));
	// Transform from shear coordinates to lab coordinates
	dDeltaX += m_dGamma * dDeltaY;
	  
	double nxA = dAP * cos(dPhiP);
	double nyA = dAP * sin(dPhiP);
	double nxB = dAQ * cos(dPhiQ);
	double nyB = dAQ * sin(dPhiQ);

	double a = dAP * dAP;
	double b = -(nxA * nxB + nyA * nyB);
	double c = dAQ * dAQ;
	double d = nxA * dDeltaX + nyA * dDeltaY;
	double e = -nxB * dDeltaX - nyB * dDeltaY;
	double delta = a * c - b * b;

	double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	double s = -(b*t+d)/a;
	double sarg = fabs(s);
	s = fmin( fmax(s,-1.), 1. );
	if (sarg > 1) 
		t = fmin( fmax( -(b*s+e)/a, -1.), 1.);
	  
	// Check if they overlap and calculate forces
	double dDx = dDeltaX + s*nxA - t*nxB;
	double dDy = dDeltaY + s*nyA - t*nyB;
	double dDSqr = dDx * dDx + dDy * dDy;
	if (dDSqr < dSigma*dSigma || dDSqr != dDSqr) {
		//cout << "contact detected" << endl;
		return true;
	}
	else {
		//cout << "no contact" << endl;
		return false;
	}
}

bool SpherocylBox::check_particle_cross(int p, int q)
{
	//cout << "with: " << q << ": " << m_psParticles[q].m_dX << " " << m_psParticles[q].m_dY << " " << m_psParticles[q].m_dPhi << endl;
	double dDeltaX = m_psParticles[p].m_dX - m_psParticles[q].m_dX;
	double dDeltaY = m_psParticles[p].m_dY - m_psParticles[q].m_dY;
	double dPhiP = m_psParticles[p].m_dPhi;
	double dPhiQ = m_psParticles[q].m_dPhi;
	double dSigma = m_psParticles[p].m_dR + m_psParticles[q].m_dR;
	double dAP = m_psParticles[p].m_dA;
	double dAQ = m_psParticles[q].m_dA;
	  
	// Make sure we take the closest distance considering boundary conditions
	dDeltaX += m_dLx * ((dDeltaX < -0.5*m_dLx) - (dDeltaX > 0.5*m_dLx));
	dDeltaY += m_dLy * ((dDeltaY < -0.5*m_dLy) - (dDeltaY > 0.5*m_dLy));
	// Transform from shear coordinates to lab coordinates
	dDeltaX += m_dGamma * dDeltaY;
	  
	double nxA = dAP * cos(dPhiP);
	double nyA = dAP * sin(dPhiP);
	double nxB = dAQ * cos(dPhiQ);
	double nyB = dAQ * sin(dPhiQ);

	double a = dAP * dAP;
	double b = -(nxA * nxB + nyA * nyB);
	double c = dAQ * dAQ;
	double d = nxA * dDeltaX + nyA * dDeltaY;
	double e = -nxB * dDeltaX - nyB * dDeltaY;
	double delta = a * c - b * b;

	double t = fmin( fmax( (b*d-a*e)/delta, -1. ), 1. );
	double s = -(b*t+d)/a;
	double sarg = fabs(s);
	s = fmin( fmax(s,-1.), 1. );
	if (sarg > 1) 
		t = fmin( fmax( -(b*s+e)/a, -1.), 1.);
	  
	// Check if they overlap and calculate forces
	double dDx = dDeltaX + s*nxA - t*nxB;
	double dDy = dDeltaY + s*nyA - t*nyB;
	double dDSqr = dDx * dDx + dDy * dDy;
	if (dDSqr < 1e-5*dSigma*dSigma || dDSqr != dDSqr) {
		//cout << "contact detected" << endl;
		return true;
	}
	else {
		//cout << "no contact" << endl;
		return false;
	}
}

bool SpherocylBox::check_particle_contact(int p)
{
	//cout << "checking contact: " << p << ": " << m_psParticles[p].m_dX << " " << m_psParticles[p].m_dY << " " << m_psParticles[p].m_dPhi << endl;
	int nCellID = m_psParticles[p].m_nCell;
	for (int ap = 0; ap < m_pvCells[nCellID].size(); ap++) {
		int q = m_pvCells[nCellID][ap];
		if (check_particle_contact(p,q)) {
			return true;
		}
	}
	for (int ac = 0; ac < 8; ac++) {
		int nACellID = m_pnCellNeighbors[nCellID][ac];
		for (int ap = 0; ap < m_pvCells[nACellID].size(); ap++) {
			int q = m_pvCells[nACellID][ap];
			if (check_particle_contact(p,q)) {
				return true;
			}
		}
	}
	//cout << "no contacts detected" << endl;
	return false;
}

bool SpherocylBox::check_particle_cross(int p)
{
	//cout << "checking contact: " << p << ": " << m_psParticles[p].m_dX << " " << m_psParticles[p].m_dY << " " << m_psParticles[p].m_dPhi << endl;
	int nCellID = m_psParticles[p].m_nCell;
	for (int ap = 0; ap < m_pvCells[nCellID].size(); ap++) {
		int q = m_pvCells[nCellID][ap];
		if (check_particle_cross(p,q)) {
			return true;
		}
	}
	for (int ac = 0; ac < 8; ac++) {
		int nACellID = m_pnCellNeighbors[nCellID][ac];
		for (int ap = 0; ap < m_pvCells[nACellID].size(); ap++) {
			int q = m_pvCells[nACellID][ap];
			if (check_particle_cross(p,q)) {
				return true;
			}
		}
	}
	//cout << "no contacts detected" << endl;
	return false;
}

void SpherocylBox::random_zero_energy_configuration(double dA, double dR)
{
	assert(dA >= 0);
	assert(dR > 0);
	m_dAmax = dA;
	m_dRmax = dR;
	double dAlpha = dA/dR;
	double dC = (3*D_PI + 24*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha);
	double dArea = dR*(D_PI*dR + 4*dA);
	double dI = dR*dR*dR*dR*dC/(6*dArea);
	double dRC = (8*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha)/dC;
	configure_cells();
	for (int p = 0; p < m_nParticles; p++) {
		m_pvNeighbors[p].clear();
	}	
	
	std::srand(time(0));
	for (int p = 0; p < m_nParticles; p++) {
		bool bContact = true;
		int nTries = 0;
		while (bContact) {
			nTries += 1;
			m_psParticles[p].m_dX = m_dLx*double(std::rand())/double(std::numeric_limits<int>::max());
			m_psParticles[p].m_dY = m_dLy*double(std::rand())/double(std::numeric_limits<int>::max());
			m_psParticles[p].m_dPhi = 2*D_PI*double(std::rand())/double(std::numeric_limits<int>::max());
			m_psParticles[p].m_dA = dA;
			m_psParticles[p].m_dR = dR;
			int nCol = int(m_psParticles[p].m_dX / m_dCellWidth);
			int nRow = int(m_psParticles[p].m_dY / m_dCellHeight);
			int nCellID = nCol + nRow * m_nCellCols;
			m_psParticles[p].m_nCell = nCellID;
			bContact = check_particle_contact(p);
		}
		m_pvCells[m_psParticles[p].m_nCell].push_back(p);
		m_psParticles[p].m_dI = dI;
		m_psParticles[p].m_dRC = dRC;
		cout << "Particle " << p << " placed in " << nTries << " tries" << endl;
	}

	m_dPacking = calculate_packing_fraction();
	find_neighbors();
}

void SpherocylBox::random_configuration(double dA, double dR)
{
	assert(dA >= 0);
	assert(dR > 0);
	m_dAmax = dA;
	m_dRmax = dR;
	double dAlpha = dA/dR;
	double dC = (3*D_PI + 24*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha);
	double dArea = dR*(D_PI*dR + 4*dA);
	double dI = dR*dR*dR*dR*dC/(6*dArea);
	double dRC = (8*dAlpha + 6*D_PI*dAlpha*dAlpha + 8*dAlpha*dAlpha*dAlpha)/dC;
	configure_cells();
	for (int p = 0; p < m_nParticles; p++) {
		m_pvNeighbors[p].clear();
	}	
	
	std::srand(time(0));
	for (int p = 0; p < m_nParticles; p++) {
		bool bContact = true;
		int nTries = 0;
		while (bContact) {
			nTries += 1;
			m_psParticles[p].m_dX = m_dLx*double(std::rand())/double(std::numeric_limits<int>::max());
			m_psParticles[p].m_dY = m_dLy*double(std::rand())/double(std::numeric_limits<int>::max());
			m_psParticles[p].m_dPhi = 2*D_PI*double(std::rand())/double(std::numeric_limits<int>::max());
			m_psParticles[p].m_dA = dA;
			m_psParticles[p].m_dR = dR;
			int nCol = int(m_psParticles[p].m_dX / m_dCellWidth);
			int nRow = int(m_psParticles[p].m_dY / m_dCellHeight);
			int nCellID = nCol + nRow * m_nCellCols;
			m_psParticles[p].m_nCell = nCellID;
			bContact = check_particle_cross(p);
		}
		m_pvCells[m_psParticles[p].m_nCell].push_back(p);
		m_psParticles[p].m_dI = dI;
		m_psParticles[p].m_dRC = dRC;
		cout << "Particle " << p << " placed in " << nTries << " tries" << endl;
	}

	m_dPacking = calculate_packing_fraction();
	find_neighbors();
}
