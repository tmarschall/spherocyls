# compilers
CXX=g++

# includes for header files
CXXINCLUDE=

# compiler flags: include all warnings, optimization level 3
CXXFLAGS=-O3 -Wall

# object files
CXXOBJECTS=spherocyl_box.o dat_file_input.o strain_dynamics.o resize_dynamics.o

# final executable
STR_EXE=../../exe/cpp_spherocyl_strain
RESZ_EXE=../../exe/cpp_spherocyl_resize

#rules:

all: $(STR_EXE) $(RESZ_EXE)
strain: $(STR_EXE)
resize: $(RESZ_EXE)

######################
# compile executable #
######################

$(STR_EXE): $(CXXOBJECTS) main.o
	$(CXX) -o $(STR_EXE) main.o $(CXXOBJECTS)

$(RESZ_EXE): $(CXXOBJECTS) main_resize.o
	$(CXX) -o $(RESZ_EXE) main_resize.o $(CXXOBJECTS)

###################
# compile objects #
###################

#c++ files
main.o: main.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main.cpp

main_resize.o: main_resize.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_resize.cpp

spherocyl_box.o: spherocyl_box.cpp spherocyl_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c spherocyl_box.cpp

dat_file_input.o: file_input.h dat_file_input.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c dat_file_input.cpp

strain_dynamics.o: spherocyl_box.h strain_dynamics.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c strain_dynamics.cpp

resize_dynamics.o: spherocyl_box.h resize_dynamics.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c resize_dynamics.cpp


#clean up object files or other assembly files
clean:
	rm -f *.o *.ptx

