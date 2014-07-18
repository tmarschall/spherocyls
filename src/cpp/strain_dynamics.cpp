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


void SpherocylBox::calc_force_pair(int p, int q)
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
	}
}

void SpherocylBox::calc_temp_force_pair(int p, int q)
{
	double dDeltaX = m_psTempParticles[p].m_dX - m_psTempParticles[q].m_dX;
	double dDeltaY = m_psTempParticles[p].m_dY - m_psTempParticles[q].m_dY;
	double dPhiP = m_psTempParticles[p].m_dPhi;
	double dPhiQ = m_psTempParticles[q].m_dPhi;
	double dSigma = m_psTempParticles[p].m_dR + m_psTempParticles[q].m_dR;
	double dAP = m_psTempParticles[p].m_dA;
	double dAQ = m_psTempParticles[q].m_dA;
	  
	// Make sure we take the closest distance considering boundary conditions
	dDeltaX += m_dLx * ((dDeltaX < -0.5*m_dLx) - (dDeltaX > 0.5*m_dLx));
	dDeltaY += m_dLy * ((dDeltaY < -0.5*m_dLy) - (dDeltaY > 0.5*m_dLy));
	// Transform from shear coordinates to lab coordinates
	dDeltaX += (m_dGamma + m_dStep * m_dStrainRate) * dDeltaY;
	  
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

void SpherocylBox::calc_forces()
{
	for (int i = 0; i < m_nParticles; i++) {
      	m_psParticles[i].m_dFx = 0.0;
      	m_psParticles[i].m_dFy = 0.0;
      	m_psParticles[i].m_dTau = 0.0;
    }

	for (int p = 0; p < m_nParticles; p++) {
    	for (int j = 0; j < m_pvNeighbors[p].size(); j++)
    	{
      		int q = m_pvNeighbors[p][j];
      		if (q > p)
				calc_force_pair(p, q);
    	}
  	}
}


void SpherocylBox::calc_temp_forces()
{
	for (int i = 0; i < m_nParticles; i++) {
      	m_psTempParticles[i].m_dFx = 0.0;
      	m_psTempParticles[i].m_dFy = 0.0;
      	m_psTempParticles[i].m_dTau = 0.0;
    }

	for (int p = 0; p < m_nParticles; p++) {
    	for (int j = 0; j < m_pvNeighbors[p].size(); j++)
    	{
      		int q = m_pvNeighbors[p][j];
      		if (q > p)
				calc_temp_force_pair(p, q);
    	}
  	}
}

double isolated_rotation(Spherocyl s)
{
  return 0.5*(1 - s.m_dRC*cos(2*s.m_dPhi));
}

void SpherocylBox::strain_step()
{
  //calc_forces();

  double *dFx0 = new double[m_nParticles];
  double *dFy0 = new double[m_nParticles];
  double *dTau0 = new double[m_nParticles];
  //double *dMoi = new double[m_nParticles];
  for (int i = 0; i < m_nParticles; i++)
  {
    dFx0[i] = m_psParticles[i].m_dFx - m_dGamma * m_psParticles[i].m_dFy;
    dFy0[i] = m_psParticles[i].m_dFy;
    //double dSinPhi = sin(m_psParticles[i].m_dPhi);
    //dMoi[i] = m_psParticles[i].m_dA * m_psParticles[i].m_dA / 3.0;
    dTau0[i] = m_psParticles[i].m_dTau / m_psParticles[i].m_dI - m_dStrainRate * isolated_rotation(m_psParticles[i]);

    m_psTempParticles[i] = m_psParticles[i];
    m_psTempParticles[i].m_dX += m_dStep * dFx0[i];
    m_psTempParticles[i].m_dY += m_dStep * dFy0[i];
    m_psTempParticles[i].m_dPhi += m_dStep * dTau0[i];
  }

  calc_temp_forces();

  for (int i = 0; i < m_nParticles; i++)
  {
    double dFx1 = m_psTempParticles[i].m_dFx - (m_dGamma + m_dStep * m_dStrainRate) * m_psTempParticles[i].m_dFy;
    double dFy1 = m_psTempParticles[i].m_dFy;
    double dSinPhi = sin(m_psTempParticles[i].m_dPhi);
    double dTau1 = m_psTempParticles[i].m_dTau / m_psParticles[i].m_dI - m_dStrainRate * isolated_rotation(m_psTempParticles[i]);

    double dDx = 0.5 * m_dStep * (dFx0[i] + dFx1);
    double dDy = 0.5 * m_dStep * (dFy0[i] + dFy1);
    double dDphi = 0.5 * m_dStep * (dTau0[i] + dTau1);
    m_psParticles[i].m_dX += dDx;
    m_psParticles[i].m_dY += dDy;
    m_psParticles[i].m_dPhi += dDphi;
    
    m_psParticles[i].m_dXMoved += dDx;
    m_psParticles[i].m_dYMoved += dDy;
    
    if (m_psParticles[i].m_dXMoved > 0.5 * m_dPadding ||
	m_psParticles[i].m_dXMoved < -0.5 * m_dPadding)
      find_neighbors();
    else if (m_psParticles[i].m_dYMoved > 0.5 * m_dPadding ||
	     m_psParticles[i].m_dYMoved < -0.5 * m_dPadding)
      find_neighbors();
  }
  
  delete[] dFx0; delete[] dFy0; delete[] dTau0; 


  m_dGamma += m_dStrainRate * m_dStep;
  m_dGammaTotal += m_dStrainRate * m_dStep;
  if (m_dGamma >= 0.5)
    set_back_gamma();
}


long unsigned int SpherocylBox::run_strain(double dRunLength, long unsigned int nTime)
{
	std::string strSEPath = m_strOutputDir + "/" + m_strSEOutput;
	std::fstream outfSE;
  	if (nTime == 0) {
		outfSE.open(strSEPath.c_str(), ios::out);
		calc_se();
		outfSE << 0 << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << endl;
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
	
  int nTSteps = int(dRunLength / m_dStrainRate + 0.5);
  int nIntSteps = int(1.0 / m_dStep + 0.5);
  int nPosSaveT = int(m_dPosSaveStep / m_dStrainRate + 0.5);
  int nSESaveT = int(m_dSESaveStep / (m_dStrainRate * m_dStep) + 0.5);
  for (unsigned long int t = nTime+1; t <= nTime + nTSteps; t++)
  {
    for (int s = 1; s <= nIntSteps; s++) {
		strain_step();
		if (s % nSESaveT == 0) {
			calc_se();
			outfSE << t << " " <<  m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
	 	}
	 	else {
	 		calc_forces();
	 	}
	}
	if (t*nIntSteps % nSESaveT == 0) {
		outfSE << t << " " <<  m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
	}
	if (t % nPosSaveT == 0) {
		outfSE.flush();
		char szBuf[11];
		sprintf(szBuf, "%010lu", t);
		std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
		save_positions(strSaveFile);
	}
  }
  outfSE.flush();
  outfSE.close();
  
  nTime += nTSteps;
  char szBuf[11];
  sprintf(szBuf, "%010lu", nTime);
  std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
  save_positions(strSaveFile);
  
  
  return nTime;
}

long unsigned int SpherocylBox::run_strain(double dRunLength)
{
  std::string strSEPath = m_strOutputDir + "/" + m_strSEOutput;
  std::ifstream inf(strSEPath.c_str());
  long unsigned int nTime;
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
	outfSE << 0 << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << endl;
	save_positions("sp0000000000.dat");
  }
  else {
  	outfSE.open(strSEPath.c_str(), ios::app);
  	calc_forces();
  }
  
  int nTSteps = int(dRunLength / m_dStrainRate + 0.5);
  int nIntSteps = int(1.0 / m_dStep + 0.5);
  int nPosSaveT = int(m_dPosSaveStep / m_dStrainRate + 0.5);
  int nSESaveT = int(m_dSESaveStep / (m_dStrainRate * m_dStep) + 0.5);
  for (unsigned long int t = nTime+1; t <= nTime + nTSteps; t++)
  {
    for (int s = 1; s <= nIntSteps; s++) {
		strain_step();
		if (s % nSESaveT == 0) {
			calc_se();
			outfSE << t << " " <<  m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
	 	}
	 	else {
	 		calc_forces();
	 	}
	}
	if (t*nIntSteps % nSESaveT == 0) {
		outfSE << t << " " <<  m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << "\n";
	}
	if (t % nPosSaveT == 0) {
		outfSE.flush();
		char szBuf[11];
		sprintf(szBuf, "%010lu", t);
		std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
		save_positions(strSaveFile);
	}
  }
  outfSE.flush();
  outfSE.close();
  
  nTime += nTSteps;
  char szBuf[11];
  sprintf(szBuf, "%010lu", nTime);
  std::string strSaveFile = std::string("sp") + szBuf + std::string(".dat");
  save_positions(strSaveFile);
  
  return nTime;
}
