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
#include <limits>


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
  int nParticles = int_input(argc, argv, ++argn, "Number of particles");
  cout << nParticles << endl;
  double dResizeRate = float_input(argc, argv, ++argn, "Resize rate");
  cout << dResizeRate << endl;
  double dStep = float_input(argc, argv, ++argn, "Integration step size");
  cout << dStep << endl;
  double dFinalPacking = float_input(argc, argv, ++argn, "Final packing fraction");
  cout << dFinalPacking << endl;
  double dPosSaveRate = float_input(argc, argv, ++argn, "Position data save rate");
  cout << dPosSaveRate << endl;
  double dStressSaveRate = float_input(argc, argv, ++argn, "Stress data save rate");
  cout << dStressSaveRate << endl;
  double dDR = float_input(argc, argv, ++argn, "Cell padding");
  cout << dDR << endl;

  if (dStressSaveRate < dResizeRate * dStep)
    dStressSaveRate = dResizeRate * dStep;
  if (dPosSaveRate < dResizeRate)
    dPosSaveRate = dResizeRate;

  double dL;
  double *pdX = new double[nParticles];
  double *pdY = new double[nParticles];
  double *pdPhi = new double[nParticles];
  double *pdRad = new double[nParticles];
  double *pdA = new double[nParticles];
  double dRMax = 0.0;
  double dAMax = 0.0;
  double dGamma;
  double dTotalGamma;
  long unsigned int nTime = 0;
  double dPacking;
  initialConfig config;
  if (strFile == "r")
  {
  	srand(time(0));
    dPacking = float_input(argc, argv, ++argn, "Starting Packing Fraction");
    cout << dPacking << endl;
    double dA = float_input(argc, argv, ++argn, "Half-shaft length");
    cout << dA << endl;
    double dR = float_input(argc, argv, ++argn, "Radius");
    cout << dR << endl;
    if (argc > ++argn) {
      config = (initialConfig)int_input(argc, argv, argn, "Initial configuration (0: random, 1: random-aligned, 2: zero-energy, 3: zero-energy-aligned)");
    }
    else {
      config = RANDOM;
    }
    double dArea = nParticles*(4.0*dA + D_PI * dR)*dR;
    dL = sqrt(dArea / dPacking);
    cout << "Box length L: " << dL << endl;
    dAMax = dA;
    dRMax = dR;
	for (int p = 0; p < nParticles; p++) {
		pdX[p] = dL * double(rand()) / double(numeric_limits<int>::max());
		pdY[p] = dL * double(rand()) / double(numeric_limits<int>::max());
		pdPhi[p] = 2*D_PI * double(rand()) / double(numeric_limits<int>::max());
		pdA[p] = dAMax;
		pdRad[p] = dRMax;
	}
		
    dGamma = 0.0;
    dTotalGamma = 0.0;
    nTime = 0;
  }
  else
  {
    cout << "Loading file: " << strFile << endl;
    DatFileInput cData(szFile, 0);
    if (cData.getInt(0, 5) != nParticles) {
    	cout << "Warning: Number of particles in data file may not match requested number" << endl;
    	cerr << "Warning: Number of particles in data file may not match requested number" << endl;
    }
    cData.getColumn(pdX, 0);
    cData.getColumn(pdY, 1);
    cData.getColumn(pdPhi, 2);
    cData.getColumn(pdA, 3);
    cData.getColumn(pdRad, 4);
    dL = cData.getFloat(0,6);
    dPacking = cData.getFloat(0,7);
    dGamma = cData.getFloat(0,8);
    dTotalGamma = cData.getFloat(0,9);
    
    int nFileLen = strFile.length();
    string strNum = strFile.substr(nFileLen-14, 10);
    nTime = atoi(strNum.c_str());
    cout << "Time: " << strNum << " " << nTime << endl;
    for (int p = 0; p < nParticles; p++) {
    	if (pdA[p] > dAMax) { dAMax = pdA[p]; }
    	if (pdRad[p] > dRMax) { dRMax = pdRad[p]; }
    }
  }
  
  int tStart = time(0);

  SpherocylBox *cBox;
  if (strFile == "r") {
  	cout << "Initializing box of length " << dL << " with " << nParticles << " particles.";
  	cBox = new SpherocylBox(dL, nParticles, dDR);
  	cBox->random_configuration(dAMax, dRMax, config);
  }
  else {
  	cout << "Initializing box from file of length " << dL << " with " << nParticles << " particles.";
  	cBox = new SpherocylBox(dL, nParticles, pdX, pdY, pdPhi, pdA, pdRad, dDR);
  }
  
  cBox->calc_se();
  cBox->display(1,1);
  cBox->set_output_directory(strDir);
  cBox->set_se_file("sp_resize_se.dat");
  cBox->set_resize_rate(dResizeRate);
  cBox->set_step(dStep);
  nTime = cBox->resize_box(dFinalPacking, nTime);  

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  delete[] pdX; delete[] pdY; delete[] pdPhi; 
  delete[] pdRad; delete[] pdA;

  return 0;
}

