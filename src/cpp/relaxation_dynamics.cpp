#include "spherocyl_box.h"
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;
using std::cerr;
using std::exit;


void SpherocylBox::gradient_relax_step()
{
  //calc_forces();
  for (int i = 0; i < m_nParticles; i++) {
    double dDx = m_dStep * m_psParticles[i].m_dFx;
    double dDy = m_dStep * m_psParticles[i].m_dFy;
    m_psParticles[i].m_dX += dDx;
    m_psParticles[i].m_dY += dDy;
    m_psParticles[i].m_dPhi += m_dStep * m_psParticles[i].m_dTau / m_psParticles[i].m_dI;

    m_psParticles[i].m_dXMoved += dDx;
    m_psParticles[i].m_dYMoved += dDy;

    if (m_psParticles[i].m_dXMoved > 0.5 * m_dPadding ||
	m_psParticles[i].m_dXMoved < -0.5 * m_dPadding)
      find_neighbors();
    else if (m_psParticles[i].m_dYMoved > 0.5 * m_dPadding ||
	     m_psParticles[i].m_dYMoved < -0.5 * m_dPadding)
      find_neighbors();
  }
}

void SpherocylBox::gradient_descent_minimize(int nMaxSteps, double dMinE)
{
  //m_dStep = dStep;

  std::string strSEPath = m_strOutputDir + "/" + m_strSEOutput;
  std::fstream outfSE;
  outfSE.open(strSEPath.c_str(), std::ios::out);
  int nPosSaveT = int(m_dPosSaveStep + 0.5);

  int loop_exit = 0;
  for (int s = 0; s < nMaxSteps; s++) {
    calc_se();
    outfSE << s << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << endl;
    if (m_dEnergy <= dMinE) {
      outfSE.flush();
      char szBuf[11];
      sprintf(szBuf, "%010lu", s);
      std::string strSaveFile = std::string("sr") + szBuf + std::string(".dat");
      save_positions(strSaveFile);
      loop_exit = 1;
      break;
    }
    else if (s % nPosSaveT == 0) {
      outfSE.flush();
      char szBuf[11];
      sprintf(szBuf, "%010lu", s);
      std::string strSaveFile = std::string("sr") + szBuf + std::string(".dat");
      save_positions(strSaveFile);
    }
    
    gradient_relax_step();
  }
  switch (loop_exit) 
    {
    case 0:
      {
	cout << "\nMaximum relaxation steps reached" << endl;
	calc_se();
	outfSE << nMaxSteps << " " << m_dEnergy << " " << m_dPxx << " " << m_dPyy << " " << m_dPxy << endl;
	char szBuf[11];
	sprintf(szBuf, "%010lu", nMaxSteps);
	std::string strSaveFile = std::string("sr") + szBuf + std::string(".dat");
	save_positions(strSaveFile);
	break;
      }
    case 1:
      {
	cout << "\nEncountered zero energy" << endl;
	break;
      }
    }
  outfSE.close();
}
