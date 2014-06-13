#include <vector>
#include <string>
#include <assert.h>
#include "spherocyl_box.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdlib>

using std::cout;
using std::endl;
using std::cerr;
using std::ios;
using std::exit;


void SpherocylBox::calc_resized_force_pair(int p, int q, bool bShrink)
{
	double dNewLx;
	double dNewLy;
	if (bShrink) {
		dNewLx = m_dLx * pow(1.0 - m_dResizeRate, m_dStep);
		dNewLy = m_dLy * pow(1.0 - m_dResizeRate, m_dStep);
	}
	else {
		dNewLx = m_dLx / pow(1.0 - m_dResizeRate, m_dStep);
		dNewLy = m_dLy / pow(1.0 - m_dResizeRate, m_dStep);
	}
	
	double dDeltaX = m_psTempParticles[p].m_dX - m_psTempParticles[q].m_dX;
	double dDeltaY = m_psTempParticles[p].m_dY - m_psTempParticles[q].m_dY;
	double dPhiP = m_psTempParticles[p].m_dPhi;
	double dPhiQ = m_psTempParticles[q].m_dPhi;
	double dSigma = m_psTempParticles[p].m_dR + m_psTempParticles[q].m_dR;
	double dAP = m_psTempParticles[p].m_dA;
	double dAQ = m_psTempParticles[q].m_dA;
	  
	// Make sure we take the closest distance considering boundary conditions
	dDeltaX += dNewLx * ((dDeltaX < -0.5*dNewLx) - (dDeltaX > 0.5*dNewLx));
	dDeltaY += dNewLy * ((dDeltaY < -0.5*dNewLy) - (dDeltaY > 0.5*dNewLy));
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
	    
	    m_psTempParticles[p].m_dFx += dFx;
	    m_psTempParticles[p].m_dFy += dFy;
	    m_psTempParticles[p].m_dTau += dTauP;
	    m_psTempParticles[q].m_dFx -= dFx;
	    m_psTempParticles[q].m_dFy -= dFy;
	    m_psTempParticles[q].m_dTau += dTauQ; 
	}
}


void SpherocylBox::calc_resized_forces(bool bShrink)
{
  for (int i = 0; i < m_nParticles; i++) {
      	m_psTempParticles[i].m_dFx = 0.0;
      	m_psTempParticles[i].m_dFy = 0.0;
      	m_psTempParticles[i].m_dTau = 0.0;
  }

  for (int p = 0; p < m_nParticles; p++) {
    for (int j = 0; j < m_pvNeighbors[p].size(); j++) {
      int q = m_pvNeighbors[p][j];
      if (q > p)
	calc_temp_force_pair(p, q);
    }
  }
}

void SpherocylBox::reconfigure_cells()
{
	double dWmin = 2.24 * (m_dRmax + m_dAmax) + m_dPadding;  //width no less than sqrt(5)*Rmax if gamma = 0.5
  	double dHmin = 2 * (m_dRmax + m_dAmax) + m_dPadding;  //height can be no less than 2*Rmax
  
  	int nNewCellRows = std::max(static_cast<int>(m_dLx / dHmin), 1);
  	int nNewCellCols = std::max(static_cast<int>(m_dLy / dWmin), 1);
  	if (nNewCellRows != m_nCellRows || nNewCellCols != m_nCellCols) {
    	configure_cells();
    	find_neighbors();
  	}
  	else {
    	m_dCellWidth = m_dLx / m_nCellCols;
    	m_dCellHeight = m_dLy / m_nCellRows;
  	}
}

/*
void SpherocylBox::resize_step(bool bShrink, int nRelaxSteps)
{
	if (bShrink) {
		m_psTempParticles[i].m_dX *= pow(1.0 - m_dResizeRate, m_dStep);
    	m_psTempParticles[i].m_dY *= pow(1.0 - m_dResizeRate, m_dStep);
    	m_dLx *= pow(1.0 - m_dResizeRate, m_dStep);
    	m_dLy *= pow(1.0 - m_dResizeRate, m_dStep);
    }
    else {
    	m_psParticles[i].m_dX /= pow(1.0 - m_dResizeRate, m_dStep);
    	m_psParticles[i].m_dY /= pow(1.0 - m_dResizeRate, m_dStep);
    	m_dLx /= pow(1.0 - m_dResizeRate, m_dStep);
    	m_dLy /= pow(1.0 - m_dResizeRate, m_dStep);
    }
    reconfigure_cells();
    
    for (int r = 0; r < nRelaxSteps; r++) {
    	calc_forces();
    	for (int i = 0; i < m_nParticles; i++) {
    		dMoi = m_psParticles[i].m_dA * m_psParticles[i].m_dA / 3.0;
    		m_psParticles[i].m_dX += m_dStep * m_psParticles[i].m_dFx;
    		m_psParticles[i].m_dY += m_dStep * m_psParticles[i].m_dFy;
    		m_psParticles[i].m_dPhi += m_dStep * m_psParticles[i].m_dTau / dMoi;
    	}
    }
    
    for (int i = 0; i < m_nParticles; i++) {
    	m_psParticles[i].m_dXMoved += m_psTempParticles[i].m_dX - m_psParticles[i].m_dX;
    	m_psParticles[i].m_dYMoved += m_psTempParticles[i].m_dY - m_psParticles[i].m_dY;
    	if (m_psParticles[i].m_dXMoved > 0.5 * m_dPadding ||
			m_psParticles[i].m_dXMoved < -0.5 * m_dPadding)
      		find_neighbors();
    	else if (m_psParticles[i].m_dYMoved > 0.5 * m_dPadding ||
	     	m_psParticles[i].m_dYMoved < -0.5 * m_dPadding)
      		find_neighbors();
    }
}
*/


void SpherocylBox::resize_step(bool bShrink)
{
  //calc_forces();

  double *dFx0 = new double[m_nParticles];
  double *dFy0 = new double[m_nParticles];
  double *dTau0 = new double[m_nParticles];
  //double *dMoi = new double[m_nParticles];
  for (int i = 0; i < m_nParticles; i++)
  {
    dFx0[i] = m_psParticles[i].m_dFx;
    dFy0[i] = m_psParticles[i].m_dFy;
    dTau0[i] = m_psParticles[i].m_dTau;
    //dMoi[i] = m_psParticles[i].m_dA * m_psParticles[i].m_dA / 3.0;

    m_psTempParticles[i] = m_psParticles[i];
    m_psTempParticles[i].m_dX += m_dStep * dFx0[i];
    m_psTempParticles[i].m_dY += m_dStep * dFy0[i];
    m_psTempParticles[i].m_dPhi += m_dStep * dTau0[i] / m_psParticles[i].m_dI;
    if (bShrink) {
    	m_psTempParticles[i].m_dX *= pow(1.0 - m_dResizeRate, m_dStep);
    	m_psTempParticles[i].m_dY *= pow(1.0 - m_dResizeRate, m_dStep);
    }
    else {
    	m_psTempParticles[i].m_dX /= pow(1.0 - m_dResizeRate, m_dStep);
    	m_psTempParticles[i].m_dY /= pow(1.0 - m_dResizeRate, m_dStep);
    }
  }
  calc_resized_forces(bShrink);

  for (int i = 0; i < m_nParticles; i++)
  {
    double dFx1 = m_psTempParticles[i].m_dFx;
    double dFy1 = m_psTempParticles[i].m_dFy;
    double dTau1 = m_psTempParticles[i].m_dTau;

    double dDx = 0.5 * m_dStep * (dFx0[i] + dFx1);
    double dDy = 0.5 * m_dStep * (dFy0[i] + dFy1);
    double dDphi = 0.5 * m_dStep * (dTau0[i] + dTau1) / m_psParticles[i].m_dI;
    double dX = m_psParticles[i].m_dX;
    double dY = m_psParticles[i].m_dY;
    m_psParticles[i].m_dX += dDx;
    m_psParticles[i].m_dY += dDy;
    m_psParticles[i].m_dPhi += dDphi;
    if (bShrink) {
    	m_psParticles[i].m_dX *= pow(1.0 - m_dResizeRate, m_dStep);
    	m_psParticles[i].m_dY *= pow(1.0 - m_dResizeRate, m_dStep);
    }
    else {
    	m_psParticles[i].m_dX /= pow(1.0 - m_dResizeRate, m_dStep);
    	m_psParticles[i].m_dY /= pow(1.0 - m_dResizeRate, m_dStep);
    }
    
    m_psParticles[i].m_dXMoved += m_psParticles[i].m_dX - dX;
    m_psParticles[i].m_dYMoved += m_psParticles[i].m_dY - dY;
    
    reconfigure_cells();
    
    if (m_psParticles[i].m_dXMoved > 0.5 * m_dPadding ||
		m_psParticles[i].m_dXMoved < -0.5 * m_dPadding)
      find_neighbors();
    else if (m_psParticles[i].m_dYMoved > 0.5 * m_dPadding ||
	     	m_psParticles[i].m_dYMoved < -0.5 * m_dPadding)
      find_neighbors();
  }  
  
  if (bShrink) {
  	m_dLx *= pow(1.0 - m_dResizeRate, m_dStep);
    m_dLy *= pow(1.0 - m_dResizeRate, m_dStep);
  }
  else {
    m_dLx /= pow(1.0 - m_dResizeRate, m_dStep);
    m_dLy /= pow(1.0 - m_dResizeRate, m_dStep);
  
  }
  
  delete[] dTau0; 
  delete[] dFx0; delete[] dFy0;
	
}


long unsigned int SpherocylBox::resize_box(double dFinalPacking, long unsigned int nTime)
{
	assert(m_dResizeRate != 0.0);
	m_dPacking = calculate_packing_fraction();
	double dPArea = m_dPacking * (m_dLx * m_dLy);
	std::string strSEPath = m_strOutputDir + "/" + m_strSEOutput;
	std::fstream outfSE;
  	if (nTime == 0) {
		outfSE.open(strSEPath.c_str(), ios::out);
		calc_se();
		outfSE << 0 << " " << m_dPacking << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << endl;
		save_positions("sp0000000000.dat");
	}
	else {
		outfSE.open(strSEPath.c_str(), ios::out | ios::in);
		if (! outfSE) {
			cerr << "Output file could not be opened for reading" << endl;
			exit(1);
		}
		bool foundT;
		char szBuf[200];
		while (! outfSE.eof()) {
			outfSE.getline(szBuf, 200);
			std::string line = szBuf;
			std::size_t spos = line.find_first_of(' ');
			std::string strTime = line.substr(0,spos);
			if (nTime == std::atoi(strTime.c_str())) {
				foundT = true;
				break;
			}	
		}
		if (!foundT) {
			cerr << "Start time not found in output file" << endl;
			exit(1);
		}
		calc_forces();
	}
	
  int nIntSteps = int(1.0 / m_dStep + 0.5);
  int nPosSaveT = int(m_dPosSaveStep / m_dResizeRate + 0.5);
  int nSESaveT = int(m_dSESaveStep / (m_dResizeRate * m_dStep) + 0.5);
  bool bShrink;
  if (dFinalPacking > m_dPacking) {
  	bShrink = true;
  }
  else {
  	bShrink = false;
  }
  
  while (bShrink ? dFinalPacking > m_dPacking : dFinalPacking < m_dPacking) {
  	nTime += 1;
    for (int s = 1; s <= nIntSteps; s++) {
		resize_step(bShrink);
		if (s % nSESaveT == 0) {
			calc_se();
			m_dPacking = dPArea / (m_dLx * m_dLy);
			outfSE << nTime << " " <<  m_dPacking << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
	 	}
		else {
	 		calc_forces();
	 	}
	}
	m_dPacking = dPArea / (m_dLx * m_dLy);
	
	if (nTime*nIntSteps % nSESaveT == 0) {
		outfSE << nTime << " " <<  m_dPacking << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
	}
	if (nTime % nPosSaveT == 0) {
		outfSE.flush();
		char szBuf[11];
		sprintf(szBuf, "%010lu", nTime);
		std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
		save_positions(strSaveFile);
	}
  }
  outfSE.flush();
  outfSE.close();

  char szBuf[11];
  sprintf(szBuf, "%010lu", nTime);
  std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
  save_positions(strSaveFile);
  
  return nTime;
}

long unsigned int SpherocylBox::resize_box(double dFinalPacking)
{
	assert(m_dResizeRate != 0.0);
	m_dPacking = calculate_packing_fraction();
	double dPArea = m_dPacking * (m_dLx * m_dLy);
  	std::string strSEPath = m_strOutputDir + "/" + m_strSEOutput;
  	std::ifstream inf(strSEPath.c_str());
  	unsigned long int nTime;
  	if (inf) {
  		std::string line;
		char szBuf[200];
		while (! inf.eof()) {
			inf.getline(szBuf, 200);
			line = szBuf;
		}
		if (! line.empty()) {
			std::size_t spos = line.find_first_of(' ');
			std::string strTime = line.substr(0,spos);
			nTime = std::atoi(strTime.c_str());
		}
		else {
			nTime = 0;
		}
  	}
	
  	std::ofstream outfSE;
  	outfSE.precision(14);
  	if (nTime == 0) {
  		outfSE.open(strSEPath.c_str());
  		calc_se();
		outfSE << 0 << " " << m_dPacking << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << endl;
		save_positions("sp0000000000.dat");
  	}
  	else {
  		outfSE.open(strSEPath.c_str(), ios::app);
  		calc_forces();
  	}
  
  
	int nIntSteps = int(1.0 / m_dStep + 0.5);
  	int nPosSaveT = int(m_dPosSaveStep / m_dResizeRate + 0.5);
  	int nSESaveT = int(m_dSESaveStep / (m_dResizeRate * m_dStep) + 0.5);
  	bool bShrink;
  	if (dFinalPacking > m_dPacking) {
  		bShrink = true;
  	}
  	else {
  		bShrink = false;
  	}
  
  	while (bShrink ? dFinalPacking > m_dPacking : dFinalPacking < m_dPacking) {
  		nTime += 1;
    	for (int s = 1; s <= nIntSteps; s++) {
			resize_step(bShrink);
			if (s % nSESaveT == 0) {
				calc_se();
				m_dPacking = dPArea / (m_dLx * m_dLy);
				outfSE << nTime << " " <<  m_dPacking << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
		 	}
			else {
	 			calc_forces();
		 	}
		}
		m_dPacking = dPArea / (m_dLx * m_dLy);
		if (nTime*nIntSteps % nSESaveT == 0) {
			outfSE << nTime << " " <<  m_dPacking << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
		}
		if (nTime % nPosSaveT == 0) {
			outfSE.flush();
			char szBuf[11];
			sprintf(szBuf, "%010lu", nTime);
			std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
			save_positions(strSaveFile);
		}
  	}
  	outfSE.flush();
  	outfSE.close();
  	
  	char szBuf[11];
  	sprintf(szBuf, "%010lu", nTime);
  	std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
  	save_positions(strSaveFile);
  	
  	return nTime;
}
