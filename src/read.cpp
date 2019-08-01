#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"


/*The part of the code that reads the atoms, the geometry and the 1-/2- electron integrals from the files.
The names of the files are hardcoded.*/

/*Function for reading the atoms and the geometry from the geometry file.
It also reads the number of orbitals in the end of the file (the last number). Be cautious.*/
void readGeom()
{
	 FILE *geomfile = fopen("inputs/geom.dat", "r");

	 int N;
	 double **coord, *zval;

	 fscanf(geomfile, "%d", &N);
	 coord = (double**) malloc(N*sizeof(double*)); //x,y,z coordinates for each atom
	 zval = (double *)malloc(sizeof(double)*N); //Atomic number of each atom

	for (int i = 0; i < N; i++)
					coord[i] = (double*) malloc(3*sizeof(double));

	double sum=0;
	for (int i=0; i<N; i++)
	{
					fscanf(geomfile, "%lf %lf %lf %lf\n", &zval[i], &coord[i][0], &coord[i][1], &coord[i][2]);
					//Format of geometry file: atom_number x y z
					sum += zval[i];
					//sum is used to compute the number of electrons
  }

  fscanf(geomfile, "%d", &norb);
  Nelec = (int)sum;
	nocc = Nelec/2 - 1; //-1 because the arrays of the 1 and 2 electron integrals start from 1 and not 0.
	dim = norb*2; //The size for the spin-orbitals

	/*	Example for water with sto-3g:
	    #define norb 7
			#define nocc 4
			#define dim 14
			#define Nelec 10*/

	fclose(geomfile);
}

/*Function for reading the one- and two- electron integrals from the files.
It also reads the number of orbitals in the end of the file (the last number). Be cautious.*/
void readData(double ****eri, gsl_matrix *T, gsl_matrix *S, gsl_matrix *V, double *enuc)
{
			FILE *Vfile = fopen("inputs/v.dat", "r"), *Tfile = fopen("inputs/t.dat", "r"), *enucfile = fopen("inputs/enuc.dat", "r"), *Sfile = fopen("inputs/s.dat", "r"), *erifile = fopen("inputs/eri.dat", "r");
			fscanf(enucfile, "%lf", enuc);

			double val;
			//Reading one electron integrals.
			for (int i=0; i< norb*(norb+1)/2; i++)
			{
				int pos1,pos2;
				fscanf(Sfile, "%d %d %lf\n", &pos1, &pos2, &val);
										gsl_matrix_set(S, pos1-1, pos2-1, val);
				fscanf(Tfile, "%d %d %lf\n", &pos1, &pos2, &val);
										gsl_matrix_set(T, pos1-1, pos2-1, val);
				fscanf(Vfile, "%d %d %lf\n", &pos1, &pos2, &val);
										gsl_matrix_set(V, pos1-1, pos2-1, val);
										gsl_matrix_set(S, pos2-1, pos1-1, gsl_matrix_get (S, pos1-1, pos2-1));
										gsl_matrix_set(V, pos2-1, pos1-1, gsl_matrix_get (V, pos1-1, pos2-1));
										gsl_matrix_set(T, pos2-1, pos1-1, gsl_matrix_get (T, pos1-1, pos2-1));

			}
			set_zero(eri, norb);
		  int mu,nu,l,s;
			//Reading two electron integrals.
			while (fscanf(erifile, "%d %d %d %d %lf", &mu, &nu, &l, &s, &val) != EOF)
			{
										eri[mu][nu][l][s] = val;
										eri[nu][mu][l][s] = val;
										eri[mu][nu][s][l] = val;
										eri[nu][mu][s][l] = val;
										eri[l][s][mu][nu] = val;
										eri[s][l][mu][nu] = val;
										eri[l][s][nu][mu] = val;
										eri[s][l][nu][mu] = val;
			}

			fclose(erifile);
			fclose(Sfile);
			fclose(Tfile);
			fclose(Vfile);
}
