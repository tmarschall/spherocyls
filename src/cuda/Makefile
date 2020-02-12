# compilers
CXX=g++
NVCC=nvcc

# paths of cuda libraries
CUDAPATH=/software/cuda/10.0.130/usr/local/cuda-10.0/
# for cudpp:
#CUDASDKPATH=/usr/local/cuda/SDK/NVIDIA_GPU_Computing_SDK/C

# includes for header files
CXXINCLUDE=-I$(CUDAPATH)/include
CUDAINCLUDE=-I$(CUDAPATH)/include

# compiler flags: include all warnings, optimization level 3
CXXFLAGS=-O1 -DGOLD_FUNCS=0 -DSHEAR_INIT=0 -std=c++11
NVCCFLAGS=--ptxas-options=-v -O1 -arch=sm_35 -DGOLD_FUNCS=0
# linker flags: include all warnings, include library files
LDFLAGS=-Wall -lcudart -lcurand 
#target_compile_options(strain PRIVATE -DSHEAR_INIT=1)
#target_compile_options(resize PRIVATE -DSHEAR_INIT=0)

# object files
CXXOBJECTS=spherocyl_box.o cudaErr.o dat_file_input.o
NVCCOBJECTS=find_neighbors.o calculate_stress_energy.o data_primitives_2.o strain_dynamics.o energy_minimize.o 2point_dynamics.o

# final executable
EXE=../../exe/spherocyl_box
RESZ=../../exe/spherocyl_resize
RELX=../../exe/spherocyl_relax
CJ=../../exe/spherocyl_cj_relax
QSC=../../exe/spherocyl_qs_compress
JAM=../../exe/spherocyl_findjam
MKCFG=../../exe/spherocyl_configs
TEST=../../exe/test
SRK2P=../../exe/spherocyl2p_compress
RSZ2P=../../exe/spherocyl2p_resize
QS2P=../../exe/spherocyl2p_quasistatic
FJ2P=../../exe/spherocyl2p_findjam
YJ2P=../../exe/spherocyl2p_ycompress
PURE=../../exe/spherocyl2p_pureshear
#PROF_EXE=../../exe/spherocyl_box_profile

#rules:

all: $(EXE) $(RESZ) $(MKCFG)
strain: $(EXE)
resize: $(RESZ)
relax: $(RELX)
cj: $(CJ)
qs: $(QSC)
jam: $(JAM)
mkconfigs: $(MKCFG)
cutest: $(TEST)
resz2p: $(RSZ2P)
qs2p: $(QS2P)
srk2p: $(SRK2P)
jam2p: $(FJ2P)
yjam2p: $(YJ2P)
pure: $(PURE)
#prof: $(PROF_EXE)

######################
# compile executable #
######################

$(EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main.o
	$(CXX) -o $(EXE) main.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(RESZ): $(CXXOBJECTS) $(NVCCOBJECTS) main_resize.o
	$(CXX) -o $(RESZ) main_resize.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(RELX): $(CXXOBJECTS) $(NVCCOBJECTS) main_relax.o
	$(CXX) -o $(RELX) main_relax.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(CJ): $(CXXOBJECTS) $(NVCCOBJECTS) main_cj_relax.o
	$(CXX) -o $(CJ) main_cj_relax.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(QSC): $(CXXOBJECTS) $(NVCCOBJECTS) main_qs_compress.o
	$(CXX) -o $(QSC) main_qs_compress.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(JAM): $(CXXOBJECTS) $(NVCCOBJECTS) main_1point_findjam.o
	$(CXX) -o $(JAM) main_1point_findjam.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(RSZ2P): $(CXXOBJECTS) $(NVCCOBJECTS) main_2point_resize.o
	$(CXX) -o $(RSZ2P) main_2point_resize.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(SRK2P): $(CXXOBJECTS) $(NVCCOBJECTS) main_2point_qscompress.o
	$(CXX) -o $(SRK2P) main_2point_qscompress.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(FJ2P): $(CXXOBJECTS) $(NVCCOBJECTS) main_2point_findjam.o
	$(CXX) -o $(FJ2P) main_2point_findjam.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(YJ2P): $(CXXOBJECTS) $(NVCCOBJECTS) main_2point_ycompress.o
	$(CXX) -o $(YJ2P) main_2point_ycompress.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(PURE): $(CXXOBJECTS) $(NVCCOBJECTS) main_pure_shear.o
	$(CXX) -o $(PURE) main_pure_shear.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(QS2P): $(CXXOBJECTS) $(NVCCOBJECTS) main_qs2p.o
	$(CXX) -o $(QS2P) main_qs2p.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(MKCFG): $(CXXOBJECTS) $(NVCCOBJECTS) main_configmaker.o
	$(CXX) -o $(MKCFG) main_configmaker.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(TEST): test.o $(CXXOBJECTS) $(NVCCOBJECTS)
	$(CXX) -o $(TEST) test.o cudaErr.o $(LDFLAGS)


#$(PROF_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_profile.o
#	$(CXX) -o $(PROF_EXE) main_profile.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

###################
# compile objects #
###################

#c++ files
main.o: main.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main.cpp

main_resize.o: main_resize.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_resize.cpp

main_relax.o: main_relax.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_relax.cpp

main_cj_relax.o: main_cj_relax.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_cj_relax.cpp

main_qs_compress.o: main_qs_compress.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_qs_compress.cpp

main_1point_findjam.o: main_1point_findjam.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_1point_findjam.cpp

main_configmaker.o: main_configmaker.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_configmaker.cpp

main_qs2p.o: main_qs2p.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_qs2p.cpp

main_2point_resize.o: main_2point_resize.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_2point_resize.cpp

main_2point_qscompress.o: main_2point_qscompress.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_2point_qscompress.cpp

main_2point_findjam.o: main_2point_findjam.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c main_2point_findjam.cpp

main_2point_ycompress.o: main_2point_ycompress.cpp spherocyl_box.h 2point_dynamics.cu
	$(CXX) $(CXXFLAGS) -c main_2point_ycompress.cpp

main_pure_shear.o: main_pure_shear.cpp spherocyl_box.h 2point_dynamics.cu
	$(CXX) $(CXXFLAGS) -c main_pure_shear.cpp

test.o: test.cu cudaErr.h
	$(NVCC) --ptxas-options=-v -arch=sm_35 -DGOLD_FUNCS=0 -c test.cu

#main_profile.o: main_profile.cpp spherocyl_box.h
#	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_profile.cpp

spherocyl_box.o: spherocyl_box.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) -c spherocyl_box.cpp

cudaErr.o: cudaErr.h cudaErr.cpp
	$(CXX) $(CXXFLAGS) -c cudaErr.cpp

dat_file_input.o: file_input.h dat_file_input.cpp
	$(CXX) $(CXXFLAGS) -c dat_file_input.cpp

#cuda files
find_neighbors.o: spherocyl_box.h data_primitives.h cudaErr.h find_neighbors.cu
	$(NVCC) $(NVCCFLAGS) -c find_neighbors.cu

calculate_stress_energy.o: spherocyl_box.h cudaErr.h calculate_stress_energy.cu
	$(NVCC) $(NVCCFLAGS) -c calculate_stress_energy.cu

strain_dynamics.o: spherocyl_box.h cudaErr.h strain_dynamics.cu
	$(NVCC) $(NVCCFLAGS) -c strain_dynamics.cu

energy_minimize.o: spherocyl_box.h cudaErr.h energy_minimize.cu
	$(NVCC) $(NVCCFLAGS) -c energy_minimize.cu

2point_dynamics.o: spherocyl_box.h cudaErr.h 2point_dynamics.cu
	$(NVCC) $(NVCCFLAGS) -c 2point_dynamics.cu

data_primitives_2.o: data_primitives.h
	$(NVCC) $(NVCCFLAGS) -c data_primitives_2.cu

#clean up object files or other assembly files
clean:
	rm -f *.o *.ptx

