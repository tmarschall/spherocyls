/*
 *
 *
 */

#include "spherocyl_box.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include "file_input.h"
#include <string>

using namespace std;

const double D_PI = 3.14159265358979;

int int_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    int input = atoi(argv[argn]);
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    int input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}
double float_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    double input = atof(argv[argn]);
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    double input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}
string string_input(int argc, char* argv[], int argn, char description[] = "")
{
  if (argc > argn) {
    string input = argv[argn];
    cout << description << ": " << input << endl;
    return input;
  }
  else {
    string input;
    cout << description << ": ";
    cin >> input;
    return input;
  }
}

int main(int argc, char* argv[])
{
  int argn = 0;
  string strFile = string_input(argc, argv, ++argn, "Data file ('r' for random)");
  const char* szFile = strFile.c_str();
  cout << strFile << endl;
  string strDir = string_input(argc, argv, ++argn, "Output data directory");
  const char* szDir = strDir.c_str();
  cout << strDir << endl;
  int nSpherocyls = int_input(argc, argv, ++argn, "Number of particles");
  cout << nSpherocyls << endl;
  double dResizeRate = float_input(argc, argv, ++argn, "Resize rate (epsilon)");
  cout << dResizeRate << endl;
  double dStep = float_input(argc, argv, ++argn, "Integration step size");
  cout << dStep << endl;
  double dJamEnergy = float_input(argc, argv, ++argn, "Energy at jamming");
  cout << dJamEnergy << endl;
  double dPosSaveRate = float_input(argc, argv, ++argn, "Position data save rate");
  cout << dPosSaveRate << endl;
  double dStressSaveRate = float_input(argc, argv, ++argn, "Stress data save rate");
  cout << dStressSaveRate << endl;
  double dDR = float_input(argc, argv, ++argn, "Cell padding");
  cout << dDR << endl;

  if (fabs(dStressSaveRate) < fabs(dResizeRate)) {
	  dStressSaveRate = dResizeRate;
  }
  if (fabs(dPosSaveRate) < fabs(dStressSaveRate)) {
	  dPosSaveRate = dStressSaveRate;
  }

  double dL;
  double dLx;
  double dLy = 0;
  double *dX = new double[nSpherocyls];
  double *dY = new double[nSpherocyls];
  double *dPhi = new double[nSpherocyls];
  double *dRad = new double[nSpherocyls];
  double *dA = new double[nSpherocyls];
  double dRMax = 0.0;
  double dAMax = 0.0;
  double dAspect = 4;
  double dBidispersity = 1;
  double dGamma;
  double dTotalGamma;
  long unsigned int nTime = 0;
  double dPacking;
  initialConfig config;
  Config sConfig;
  if (strFile == "r")
  {
    double dPacking = float_input(argc, argv, ++argn, "Packing Fraction");
    cout << dPacking << endl;
    dAspect = float_input(argc, argv, ++argn, "Aspect ratio (i.e. A/R or 2A/D):");
    cout << dAspect << endl;
    dBidispersity = float_input(argc, argv, ++argn, "Bidispersity ratio: (1 for monodisperse):");
    cout << dBidispersity << endl;
    if (argc > argn) {
      config = (initialConfig)int_input(argc, argv, ++argn, "Initial configuration (0: random, 1: random-uniform, 2: random-aligned, 3: zero-energy, 4: zero-energy-uniform, 5: zero-energy-aligned, 6: grid, 7: grid-uniform, 8: grid-aligned, 9: other-specified)");
      if (config == 9) {
	sConfig.type = (configType)int_input(argc, argv, ++argn, "Configuration type (0: random, 1: grid)");
	sConfig.minAngle = float_input(argc, argv, ++argn, "Minimum angle");
	sConfig.maxAngle = float_input(argc, argv, ++argn, "Maximum angle");
	sConfig.overlap = float_input(argc, argv, ++argn, "Allowed overlap (min 0, max 1)");
	sConfig.angleType = (configType)int_input(argc, argv, ++argn, "Angle Distribution (0: random, 1: uniform");
      }
      else {
	if (config < 6)
	  sConfig.type = RANDOM;
	else
	  sConfig.type = GRID;
	sConfig.minAngle = 0;
	if (config % 3 == 2)
	  sConfig.maxAngle = 0;
	else
	  sConfig.maxAngle = 1;
	if (config >= 3 && config < 6)
	  sConfig.overlap = 0;
	else
	  sConfig.overlap = 1;
	if (config %3 == 0)
	  sConfig.angleType = RANDOM;
	else
	  sConfig.angleType = GRID;
      }
    }
    else {
      sConfig.type = RANDOM;
      sConfig.minAngle = 0;
      sConfig.maxAngle = D_PI;
      sConfig.overlap = 1;
    }
    double dR = 0.5;
    double dA = dAspect*dR;
    double dArea = 0.5*nSpherocyls*(1+dBidispersity*dBidispersity)*(4*dA + D_PI * dR)*dR;
    dL = sqrt(dArea / dPacking);
    cout << "Box length L: " << dL << endl;
    dRMax = dBidispersity*dR;
    dAMax = dBidispersity*dA;
    /*
    srand(time(0) + static_cast<int>(1000*dPacking));
    for (int p = 0; p < nSpherocyls; p++)
    {
      dX[p] = dL * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dY[p] = dL * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dPhi[p] = 2.*pi * static_cast<double>(rand() % 1000000000) / 1000000000.;
      dRad[p] = dR;
      dA[p] = dA.; 
    }
    */
    dGamma = 0.;
    dTotalGamma = 0.;
  }
  else
  {
    cout << "Loading file: " << strFile << endl;
    DatFileInput cData(szFile, 1);
    int nHeadLen = cData.getHeadLen();

    if (cData.getHeadInt(0) != nSpherocyls) {
    	cout << "Warning: Number of particles in data file may not match requested number" << endl;
    	cerr << "Warning: Number of particles in data file may not match requested number" << endl;
    }
    cData.getColumn(dX, 0);
    cData.getColumn(dY, 1);
    cData.getColumn(dPhi, 2);
    cData.getColumn(dRad, 3);
    cData.getColumn(dA, 4);
    if (nHeadLen == 8) {
    	dLx = cData.getHeadFloat(1);
    	dLy = cData.getHeadFloat(2);
    	dPacking = cData.getHeadFloat(3);
    	dGamma = cData.getHeadFloat(4);
    	dTotalGamma = cData.getHeadFloat(5);
    }
    else {
    	dL = cData.getHeadFloat(1);
    	dPacking = cData.getHeadFloat(2);
    	dGamma = cData.getHeadFloat(3);
    	dTotalGamma = cData.getHeadFloat(4);
    }
    
    int nFileLen = strFile.length();
    string strNum = strFile.substr(nFileLen-14, 10);
    nTime = atol(strNum.c_str());
    cout << "Time: " << strNum << " " << nTime << endl;
    for (int p = 0; p < nSpherocyls; p++) {
    	if (dA[p] > dAMax) { dAMax = dA[p]; }
    	if (dRad[p] > dRMax) { dRMax = dRad[p]; }
    }
  }
  
  int tStart = time(0);

  Spherocyl_Box *cSpherocyls;
  if (strFile == "r") {
    cout << "Initializing box of length " << dL << " with " << nSpherocyls << " particles.";
    cSpherocyls = new Spherocyl_Box(nSpherocyls, dL, dAspect, dBidispersity, sConfig, dDR);
  }
  else {
	  if (dLy == 0) {
	  		cout << "Initializing box from file of length " << dL << " with " << nSpherocyls << " particles." << endl;
	  		cSpherocyls = new Spherocyl_Box(nSpherocyls, dL, dX, dY, dPhi, dRad, dA, dDR);
	  	}
	  	else {
	  		cout << "Initializing box from file of length " << dLx << " x " << dLy << " with " << nSpherocyls << " particles." << endl;
	  		cSpherocyls = new Spherocyl_Box(nSpherocyls, dLx, dLy, dX, dY, dPhi, dRad, dA, dDR);
	  	}
  }
  cout << "Spherocyls initialized" << endl;

  cSpherocyls->set_gamma(dGamma);
  cSpherocyls->set_total_gamma(dTotalGamma);
  cSpherocyls->set_strain(0);
  cSpherocyls->set_step(dStep);
  cSpherocyls->set_data_dir(szDir);
  cout << "Configuration set" << endl;
  
  //cSpherocyls.place_random_spherocyls();
  //cout << "Random spherocyls placed" << endl;
  //cSpherocyls.reorder_particles();
  //cSpherocyls.reset_IDs();
  //cout << "Spherocyls reordered" << endl;
  cSpherocyls->find_neighbors();
  cout << "Neighbor lists created" << endl;
  cSpherocyls->calculate_stress_energy();
  cout << "Stresses calculated" << endl;
  cSpherocyls->display(1,1,1,1);


  cSpherocyls->find_jam_1p(dJamEnergy, dResizeRate);
  cSpherocyls->calculate_stress_energy();
  cSpherocyls->display(1,0,0,1);

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  delete[] dX; delete[] dY; delete[] dPhi; 
  delete[] dRad; delete[] dA;

  return 0;
}
