/*  spherocyl_box.h
 *
 *  Arrays with the prefix "h_" are allocated on the host (CPU), 
 *   those with the prefix "d_" are allocated on the device (GPU).
 *  Anything starting with "g_" is solely used in "gold" functions 
 *   on the cpu (for error checking) and will only be compiled if 
 *   the environmental variable GOLD_FUNCS=1
 *  
 */

#ifndef GOLD_FUNCS
#define GOLD_FUNCS 0
#endif

#ifndef SHEAR_INIT
#define SHEAR_INIT 0
#endif

#ifndef SPHEROCYL_BOX_H
#define SPHEROCYL_BOX_H

#include <stdio.h>
#include <string.h>

enum Potential {HARMONIC = 2, HERTZIAN = 5};
enum initialConfig {RANDOM_UNALIGNED, RANDOM_UNIFORM, RANDOM_ALIGNED, ZERO_E_UNALIGNED, ZERO_E_UNIFORM, ZERO_E_ALIGNED, GRID_UNALIGNED, GRID_UNIFORM, GRID_ALIGNED, OTHER};
enum configType {RANDOM, GRID};
struct Config {
  configType type;
  configType angleType;
  double minAngle;
  double maxAngle;
  double overlap;
};

class Spherocyl_Box
{
 private:
  int m_nSpherocyls;
  double m_dLx;  // Box side length
  double m_dLy;
  double m_dPacking;  // Packing fraction
  double m_dGamma;   // Strain parameter
  double m_dTotalGamma;
  Potential m_ePotential;  // Soft core interaction (harmonic or hertzian)
  double m_dStrainRate;
  double m_dStep;
  double m_dMove;
  double m_dKd; // Dissipative constant

  // file output data
  FILE *m_pOutfSE;  // output steam
  const char *m_szDataDir;  // directory to save data (particle configurations)
  const char *m_szFileSE;  // filename for stress energy data

  // Coordinates etc. for spherocyls
  double m_dRMax;  // Largest radius
  double m_dAMax;  // Largest spherocylinder half-shaft
  double m_dASig;  // Relative sigma in width of aspect distribution
  // Position and orientation arrays to be allocated on the CPU (host)
  double *h_pdX;
  double *h_pdY;
  double *h_pdPhi;
  double *h_pdR;  // Radii
  double *h_pdA;
  int *h_pnMemID;  // IDs of particles reordered in memory
  int *h_bNewNbrs;
  // Position and orientation arrays to be allocated on the GPU (device)
  double *d_pdX;
  double *d_pdY;
  double *d_pdPhi;
  double *d_pdR;
  double *d_pdA;
  double *d_pdArea;
  double *d_pdMOI;  // Moment of Inertia
  double *d_pdIsoC;  // Coefficient for isolated rotation function
  bool m_bMOI;
  double *d_pdTempX;  // Temporary positions used in several routines
  double *d_pdTempY;  //  only needed on the device
  double *d_pdTempPhi;
  double *d_pdXMoved;  // Amount each spherocylinder moved since last finding neighbors
  double *d_pdYMoved;
  double *d_pdDx;  // Search directions Dx, Dy, Dt used for minimization routines
  double *d_pdDy;
  double *d_pdDt;
  int *d_pnInitID;  // The original ID of the particle
  int *d_pnMemID;  // ID of the current location of the particle in memory
  int *d_bNewNbrs;  // Set to 1 when particle moves more than dEpsilon
  // Arrays for "gold" routines (for testing)
#if GOLD_FUNCS == 1
  double *g_pdX;
  double *g_pdY;
  double *g_pdPhi;
  double *g_pdR;
  double *g_pdA;
  double *g_pdTempX;
  double *g_pdTempY;
  double *g_pdTempPhi;
  int *g_pnInitID;
  int *g_pnMemID;
  double *g_pdXMoved;
  double *g_pdYMoved;
  int *g_bNewNbrs;
#endif

  // Stresses and forces
  float *m_pfEnergy;
  float *m_pfPxx;
  float *m_pfPyy;
  float *m_pfPxy;
  float *m_pfPyx;
  float m_fP;  // Total pressure
  float *h_pfSE;  // Array for transfering the stresses and energy
  double *h_pdBlockSums;
  double *h_pdLineEnergy;
  double *h_pdEnergies;
  double *h_pdFx;
  double *h_pdFy;
  double *h_pdFt;
  // GPU
  float *d_pfSE;
  double *d_pdBlockSums;
  double *d_pdEnergies;
  double *d_pdFx;
  double *d_pdFy;
  double *d_pdFt;
  double *d_pdTempFx;
  double *d_pdTempFy;
  double *d_pdTempFt;
#if GOLD_FUNCS == 1
  float *g_pfSE;
  double *g_pdFx;
  double *g_pdFy;
  double *g_pdFt;
#endif
  
  // These variables are used for spatial subdivision and
  // neighbor lists for faster contact detection etc.
  int m_nCells; 
  int m_nCellRows;
  int m_nCellCols;
  double m_dCellW;    // Cell width
  double m_dCellH;    // Cell height
  int *h_pnCellID;    // Cell ID for each particle
  int *h_pnAdjCells;  // Which cells are next to each other
  int m_nMaxPPC;      // Max particles per cell
  int *h_pnPPC;       // Number of particles in each cell
  int *h_pnCellList;  // List of particles in each cell
  int m_nMaxNbrs;     // Max neighbors a particle can have
  int *h_pnNPP;       // Number of neighbors (possible contacts) of each particle
  int *h_pnNbrList;   // List of each particle's neighbors
  // GPU arrays
  int *d_pnCellID;
  int *d_pnAdjCells;
  int *d_pnPPC;
  int *d_pnCellList;
  int *d_pnNPP;
  int *d_pnNbrList;
#if GOLD_FUNCS == 1
  int *g_pnCellID;
  int *g_pnAdjCells;
  int *g_pnPPC;
  int *g_pnCellList;
  int *g_pnNPP;
  int *g_pnNbrList;
#endif

  // Used to not have to update neighbors every step when things are moving
  double m_dEpsilon;

  // These variables define configurations for kernel cuda kernel launches
  int m_nGridSize;  // Grid size (# of thread blocks) for finding cell IDs
  int m_nBlockSize;  // Block size (# threads per block)
  int m_nSM_CalcF;
  int m_nSM_CalcSE;  // Shared memory per block
  int m_nDeviceMem;
  
  void construct_defaults();
  void configure_cells();
  void configure_cells(double dMaxGamma);
  void reconfigure_cells();
  void reconfigure_cells(double dMaxGamma);
  void set_kernel_configs();
  double calculate_packing();
  void set_shapes(int seed, double dBidispersity);
  
  void strain_step(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0);
  void strain_step_2p(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0);
  void pure_strain_step_2p(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0);
  void strain_step_ngl(long unsigned int nTime, bool bSvStress = 0, bool bSvPos = 0);
  void resize_step(long unsigned int tTime, double dEpsilon, bool bSvStress, bool bSvPos);
  void resize_step_2p(long unsigned int tTime, double dEpsilon, bool bSvStress, bool bSvPos);
  void y_resize_step_2p(long unsigned int tTime, double dEpsilon, bool bSvStress, bool bSvPos);
  void relax_step(unsigned long int nTime, bool bSvStress, bool bSvPos);
  void relax_step_2p(unsigned long int nTime, bool bSvStress, bool bSvPos);
  int cjfr_relax_step(double dMinStep, double dMaxStep);
  int cjpr_relax_step(double dMinStep, double dMaxStep, double dAngleDampling);
  int cjpr_relax_step_2p(double dMinStep, double dMaxStep);
  int new_cjpr_relax_step_2p(double dMinStep, double dMaxStep);
  int gradient_descent_step_2p(double dMinStep, double dMaxStep);
  int force_cjpr_relax_step(double dMinStep, double dMaxStep, bool bAngle);
  int gd_relax_step(double dMinStep, double dMaxStep);
  int force_line_search(bool bFirstStep, bool bSecondStep, double dMinStep, double dMaxStep);
  int line_search(bool bFirstStep, bool bSecondStep, double dMinStep, double dMaxStep);
  int line_search_2p(bool bFirstStep, bool bSecondStep, double dMinStep, double dMaxStep);
  int new_line_search_2p(double dMinStep, double dMaxStep);
  int qs_one_cycle(int nTime, double dMaxE, double dResizeStep, double dMinStep);
  int resize_to_energy_2p(int nTime, double dEnergy, double dResizeStep);
  int y_resize_to_energy_2p(int nTime, double dEnergy, double dResizeStep);
  int resize_relax_step_2p(int nTime, double dStep, int nMaxCJSteps = 2000000);
  int resize_to_energy_1p(int nTime, double dEnergy, double dResizeStep);
  void flip_group();
  double uniformToAngleDist(double dUniform, double dAspect);

 public:
  Spherocyl_Box(int nSpherocyls, double dL, double dAspect, double dBidispersity, 
		Config config, double dEpsilon = 0.1, double dASig = 0, int nMaxPPC = 18, 
		int nMaxNbrs = 45, Potential ePotential = HARMONIC);
  Spherocyl_Box(int nSpherocyls, double dLx, double dLy, double dAspect, 
		double dBidispersity, Config config, double dEpsilon = 0.1, double dASig = 0,
		int nMaxPPC = 18, int nMaxNbrs = 45, Potential ePotential = HARMONIC);
  Spherocyl_Box(int nSpherocyls, double dL, double *pdX, double *pdY, double *pdPhi, 
		double *pdR, double *pdA, double dEpsilon = 0.1,
		int nMaxPPC = 18, int nMaxNbrs = 45, Potential ePotential = HARMONIC);
  Spherocyl_Box(int nSpherocyls, double dLx, double dLy, double *pdX, double *pdY,
  		double *pdPhi, double *pdR, double *pdA, double dEpsilon = 0.1,
  		int nMaxPPC = 18, int nMaxNbrs = 45, Potential ePotential = HARMONIC);
  ~Spherocyl_Box();


  void save_positions(long unsigned int nTime);
  void save_positions_bin(long unsigned int nTime);
  void place_random_spherocyls(int seed = 0, bool bRandAngle = 1, double dBidispersity = 1);
  void place_random_0e_spherocyls(int seed = 0, bool bRandAngle = 1, double dBidispersity = 1);
  void place_spherocyl_grid(int seed = 0, bool bRandAngle = 0);
  void place_spherocyls(Config config, int seed = 0, double dBidispersity = 1);
  void get_0e_configs(int nConfigs, double dBidispersity);
  void find_neighbors();
  void set_back_gamma();
  void flip_shear_direction();
  void rotate_by_gamma();
  void reorder_particles();
  void reset_IDs();
  void calculate_stress_energy();
  void calculate_stress_energy_2p();
  bool check_for_contacts();
  bool check_for_contacts(int nIndex, double dTol = 0);
  bool check_for_crosses(int nIndex, double dEpsilon = 1e-5);
  void run_strain(double dStartGam, double dStopGam, double dSvStressGam, double dSvPosGam);
  void run_strain_2p(double dStartGam, double dStopGam, double dSvStressGam, double dSvPosGam);
  void run_pure_strain_2p(long unsigned int nStart, double dAspectStop, double dSvStressGam, double dSvPosGam);
  void run_strain(long unsigned int nStart, double dRunGamma, double dSvStressGam, double dSvPosGam);
  void run_strain_2p(long unsigned int nStart, double dRunGamma, double dSvStressGam, double dSvPosGam);
  void run_strain(long unsigned int nSteps);
  void run_strain_2p(long unsigned int nSteps);
  void run_strain(double dGammaMax);
  void resize_box(long unsigned int nStart, double dEpsilon, double dFinalPacking, double SvStressRate, double dSvPosRate);
  void resize_box_2p(long unsigned int nStart, double dEpsilon, double dFinalPacking, double SvStressRate, double dSvPosRate);
  void relax_box(unsigned long int nSteps, double dMaxStep, double dMinStep, int nStressSaveInt, int nPosSaveInt);
  void relax_box_2p(unsigned long int nSteps, double dMaxStep, double dMinStep, int nStressSaveInt, int nPosSaveInt);
  void cjfr_relax(double dMinStep, double dMaxStep);
  void cjpr_relax(double dMinStep, double dMaxStep, int nMaxSteps, double dAngleDamping);
  void cjpr_relax_2p(double dMinStep, double dMaxStep, int nMaxSteps);
  void gradient_relax_2p(double dMinStep, double dMaxStep, int nMaxSteps);
  void force_cjpr_relax(double dMinStep, double dMaxStep, int nMaxSteps);
  void gd_relax(double dMinStep, double dMaxStep);
  void quasistatic_compress(double dMaxPack, double dResizeStep, double dMinStep);
  void compress_qs2p(double dMaxPacking, double dResizeStep);
  void expand_qs2p(double dMinPacking, double dResizeStep);
  void quasistatic_cycle(int nCycles, double dMaxE, double dResizeStep, double dMinStep);
  void quasistatic_find_jam(double dMaxE, double dMinE, double dResizeStep, double dMinStep);
  void find_jam_2p(double dJamE, double dResizeStep);
  void y_compress_energy_2p(double dTargetE, double dResizeStep);
  void find_energy_2p(double dResizeStep, double dMinResizeStep, int nMaxSteps);
  void find_jam_1p(double dJamE, double dResizeStep);
  
#if GOLD_FUNCS == 1
  void calculate_stress_energy_gold();
#endif

  void display(bool bParticles = 1, bool bCells = 1, bool bNbrs = 1, bool bStress = 1);

  // Functions for setting parameters after initialization
  void set_gamma(double dGamma) { m_dGamma = dGamma; }
  void set_total_gamma(double dTotalGamma) { m_dTotalGamma = dTotalGamma; }
  void set_step(double dStep) { m_dStep = dStep; }
  void set_strain(double dStrain) { m_dStrainRate = dStrain; }
  void set_Kd(double dKd) { m_dKd = dKd; }  // Set dissipative constant
  void set_data_dir(const char *szDataDir) { m_szDataDir = szDataDir; }
  void set_se_file(const char *szFileSE) { m_szFileSE = szFileSE; }
};

#endif
