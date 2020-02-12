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
  string strDir = string_input(argc, argv, ++argn, "Output data directory");
  const char* szDir = strDir.c_str();
  cout << strDir << endl;
  int nSpherocyls = int_input(argc, argv, ++argn, "Number of particles");
  cout << nSpherocyls << endl;
  int nConfigs = int_input(argc, argv, ++argn, "Number of configurations");
  cout << nConfigs << endl;
  double dPacking = float_input(argc, argv, ++argn, "Packing Fraction");
  cout << dPacking << endl;
  double dAspect = float_input(argc, argv, ++argn, "Aspect ratio (i.e. A/R or 2A/D)");
  cout << dAspect << endl;
  double dBidispersity = float_input(argc, argv, ++argn, "Bidispersity ratio: (1 for monodisperse)");
  cout << dBidispersity << endl;
  double dDR = 0.1;

  double dR = 0.5;
  double dA = dAspect*dR;
  double dArea = 0.5*nSpherocyls*(1+dBidispersity*dBidispersity)*(4*dA + D_PI * dR)*dR;
  double dL = sqrt(dArea / dPacking);
  cout << "Box length L: " << dL << endl;
  double dRMax = dBidispersity*dR;
  double dAMax = dBidispersity*dA;
  double dGamma = 0.;
  double dTotalGamma = 0.;
  Config sConfig = {RANDOM, -D_PI/2, D_PI/2, 0};

  int tStart = time(0);

  Spherocyl_Box cSpherocyls(nSpherocyls, dL, dAspect, dBidispersity, sConfig, dDR);
  cout << "Spherocyls initialized" << endl;

  cSpherocyls.set_gamma(dGamma);
  cSpherocyls.set_total_gamma(dTotalGamma);
  cSpherocyls.set_data_dir(szDir);
  cout << "Configuration set" << endl;
  


  cSpherocyls.get_0e_configs(nConfigs, dBidispersity);

  int tStop = time(0);
  cout << "\nRun Time: " << tStop - tStart << endl;

  return 0;
}
