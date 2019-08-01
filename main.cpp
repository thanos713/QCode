#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS

#include "headers/common.h"
#include "headers/tools.h"
#include "headers/diis.h"
#include "headers/integrals.h"
#include "headers/ccsd.h"
#include "headers/ccsdt.h"
#include "headers/read.h"
#include "headers/excited_states.h"
#include "headers/hf.h"
#include "headers/not_used.h"

/* Global variables for storing the number of orbitals, the number of occupied orbitals, 2*norb and number of electrons respectively.
The number of electrons is read from the geometry (according to the elements present) and the number of orbitals is the last number of the geometry file.
The rest are calculated from the others.*/
int norb, nocc, dim, Nelec;

int main()
{
	readGeom(); //Reading Nelec and Number of orbitals
  double enuc, ****eri;   gsl_matrix *T, *S, *V, *Hcore; //All the integrals and the nuclear repulsion.
  allocate4(eri, norb+2);
  S = gsl_matrix_alloc(norb, norb); T = gsl_matrix_alloc(norb, norb); V = gsl_matrix_alloc(norb, norb); Hcore = gsl_matrix_alloc(norb, norb);
  gsl_matrix_set_zero(Hcore); gsl_matrix_set_zero(S); gsl_matrix_set_zero(V); gsl_matrix_set_zero(T);

	readData(eri, T, S, V, &enuc); //Reading all the integrals and the nuclear repulsion.

  gsl_matrix_set_zero(Hcore); gsl_matrix_add(Hcore, T);  gsl_matrix_add(Hcore, V); //Construct Hcore

	gsl_matrix *Fock = gsl_matrix_alloc(norb, norb); 	gsl_matrix *D = gsl_matrix_alloc(norb,norb); 	gsl_matrix *C = gsl_matrix_alloc(norb, norb); 	gsl_vector *epsilon = gsl_vector_alloc(norb);

  /*Keeping Ssaved, because it is needed later in DIIS for HF.*/
	gsl_matrix *Ssaved = gsl_matrix_alloc(norb, norb);
	for (int i=0; i<norb; i++)
					for (int j=0;j<norb;j++)
									gsl_matrix_set(Ssaved, i, j, gsl_matrix_get(S,i,j));

  int hfmaxiter = 128;

	double Eelec = HF(S, T, V, eri, Hcore, D, Fock, C, epsilon, Ssaved, hfmaxiter);
	printf("E(nuc) + E(HF) = %.12lf\n", Eelec+enuc);

    transformIntegrals(eri, C); //transform from AO basis to MO spatial

	  //printf("E(MP2) with spatial orbitals = %.12lf\n", mp2(epsilon, eri));

    double  ****soeri;
    allocate4(soeri, dim+1);

    gsl_matrix *soFock = gsl_matrix_alloc(dim, dim);

    spinOrbitals(soeri, eri);
    spinOFock(soFock, epsilon);

    int ccsdmaxiter=128; //1 for MP2

		double **ts, ****td;
		allocate2(ts, dim); allocate4(td, dim);
		set_zero(ts, dim); set_zero(td, dim);

		if (Eelec != 0) {
	    double eccsd = doccsd(soFock, soeri, ccsdmaxiter, ts, td);
			if (ccsdmaxiter != 1) {
	            printf("E(CCSD) = %.12lf\nE(HF) + E(CCSD) = %.12lf\n", eccsd, Eelec+enuc+eccsd);
							if (eccsd != 0) {
	    				double eccsdt = doccsdt(soFock, soeri, ts, td);
	        		printf("E(T) = %.12lf\nE(HF) + E(CCSD) + E(T) = %.12lf\n", eccsdt, Eelec+enuc+eccsd+eccsdt);}
						  }
		  else
							printf("E(HF) + E(MP2) = %.12lf\n", Eelec+enuc+eccsd);

		  free(ts); free(td);

			CIS(soeri, soFock, epsilon, eri);
			//RPA1(soeri, soFock);
			RPA2(soeri, soFock);

	 }
	 		else
							printf("No more calculations available. HF didn't converge.\n");

	return 0;
}
