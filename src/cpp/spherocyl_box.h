#include <vector>
#include <string>
#include <fstream>

#ifndef SPHEROCYL_BOX_H
#define SPHEROCYL_BOX_H

enum initialConfig {RANDOM, RANDOM_ALIGNED, ZERO_E, ZERO_E_ALIGNED}; 

struct Spherocyl {
  double m_dX;  //the x-coordinate
  double m_dY;  //the y-coordinate
  double m_dPhi;  //the orientation
  double m_dR;  //the radius
  double m_dA;  //the half-shaft length
  int m_nCell;  //the cell the particle is in
  double m_dFx;  //The total force on the particle's center of mass in the x direction
  double m_dFy;  //The total force on the particle's center of mass in the y direction
  double m_dTau;  //The total torque on the particle
  double m_dXMoved;
  double m_dYMoved;
  double m_dI;  // I (moment of inertia for massive particle)
  double m_dRC;  // Isolated rotation coefficient [ f(phi) = (1-m_dRC*cos(2*phi))/2 ]
};

class SpherocylBox
{
private:
  double m_dLx;
  double m_dLy;
  double m_dGamma;
  double m_dGammaTotal;
  double m_dStrainRate;
  double m_dResizeRate;
  double m_dStep;
  double m_dPosSaveStep;
  double m_dSESaveStep;
  double m_dPacking;
  double m_dAmax;
  double m_dRmax;

  double m_dPadding;
  double m_dMaxMoved;

  double m_dEnergy;
  double m_dPxx;
  double m_dPyy;
  double m_dPxy;
  double m_dPyx;  // Will probably not be the same as Pxy because there may be a net torque while shearing

  Spherocyl *m_psParticles;
  Spherocyl *m_psTempParticles;
  int m_nParticles;
  std::vector<int> *m_pvCells;
  int m_nCellCols;
  int m_nCellRows;
  int m_nCells;
  double m_dCellWidth;
  double m_dCellHeight;
  int **m_pnCellNeighbors;
  std::vector<int> *m_pvNeighbors;

  std::string m_strOutputDir;
  std::string m_strSEOutput;
  
  void set_defaults();
  bool check_neighbors(int p, int q);
  double calculate_packing_fraction();
  void set_back_gamma();
  void calc_se_force_pair(int p, int q);
  void calc_force_pair(int p, int q);
  void calc_temp_force_pair(int p, int q);
  void calc_temp_forces();
  void strain_step();
  void calc_resized_force_pair(int p, int q, bool bShrink);
  void calc_resized_forces(bool bShrink);
  void resize_step(bool bShrink);
  void gradient_relax_step();
  void reconfigure_cells();
  bool check_particle_contact(int p, int q);
  bool check_particle_contact(int p);
  bool check_particle_cross(int p, int q);
  bool check_particle_cross(int p);
  
public:
  SpherocylBox();
  SpherocylBox(double dL, int nParticles);
  SpherocylBox(double dL, int nParticles, double dPadding);
  SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double dA, double dR);
  SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double dA, double dR, double dPadding);
  SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double *pdA, double *pdR);
  SpherocylBox(double dL, int nParticles, double *pdX, double *pdY, double *pdPhi, double *pdA, double *pdR, double dPadding);
  ~SpherocylBox();

  void set_particle(int nIndex, double dX, double dY, double dPhi, double dA, double dR);
  void configure_cells();
  void find_cells();
  void find_neighbors();
  void display(bool bParticles = 0, bool bCells = 0);
  void save_positions(std::string strFile);
  void calc_se();
  void calc_forces();
  void random_configuration(double dA, double dR, initialConfig config = RANDOM);
  long unsigned int run_strain(double dRunLength);
  long unsigned int run_strain(double dRunLength, long unsigned int nTime);
  long unsigned int resize_box(double dFinalPacking);
  long unsigned int resize_box(double dFinalPacking, long unsigned int nTime);
  void gradient_descent_minimize(int nMaxSteps = 1000000, double dMinE = 0);
  
  void set_output_directory(std::string strOutputDir) { m_strOutputDir = strOutputDir; }
  void set_se_file(std::string strSEOutput) { m_strSEOutput = strSEOutput; }
  void set_strain_rate(double dStrainRate) { m_dStrainRate = dStrainRate; }
  void set_resize_rate(double dResizeRate) { m_dResizeRate = dResizeRate; }
  void set_step(double dStep) { m_dStep = dStep; }
  void set_pos_save_step(double dPosSaveStep) { m_dPosSaveStep = dPosSaveStep; }
  void set_se_save_step(double dSESaveStep) { m_dSESaveStep = dSESaveStep; }
  void set_cell_padding(double dPadding) { m_dPadding = dPadding; }
  void set_gamma(double dGamma) { m_dGamma = dGamma; }
  void set_total_gamma(double dTotalGamma) { m_dGammaTotal = dTotalGamma; }
  
  
};

#endif
