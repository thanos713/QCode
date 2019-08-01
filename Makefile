SHELL   = /bin/sh

FC      = g++

GSLCFLAGS = -I#absolute path of GSL/include directory
GSLLFLAGS = -L#absolute path of GSL/lib -lgsl -lgslcblas -lm


all : QCode.x


.PHONY : clean
clean :
	rm -f *.o
	rm -rf src/*.o
	rm QCode.x


.SUFFIXES :
.SUFFIXES : .cpp .o
.cpp.o :
	$(FC) -c $< -o $@ $(GSLCFLAGS)


LCHobj = main.o src/tools.o src/diis.o src/integrals.o src/ccsd.o src/ccsdt.o src/read.o src/excited_states.o src/hf.o src/not_used.o

QCode.x: $(LCHobj)
	$(FC) $(LCHobj) $(GSLCFLAGS) $(GSLLFLAGS) -o $@
