#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS


/* Global variables for storing the number of orbitals, the number of occupied orbitals, 2*norb and number of electrons respectively.
The number of electrons is read from the geometry (according to the elements present) and the number of orbitals is the last number of the geometry file.
The rest are calculated from the others.*/
int norb, nocc, dim, Nelec;


/*Function for allocating *NxNxNxN* 4D C/C++ arrays/tensors automatically.*/
void allocate4(double ****&tensor, int dimension)
{
        tensor = (double****)calloc(dimension, sizeof(double ***));
        for(int ii = 0; ii < dimension; ii++) {
                tensor[ii] = (double***)calloc(dimension, sizeof(double**));
                for(int jj = 0; jj < dimension; jj++) {
                        tensor[ii][jj] = (double**)calloc(dimension, sizeof(double*));
                        for(int kk = 0; kk < dimension; kk++) {
                                tensor[ii][jj][kk] = (double*)calloc(dimension, sizeof(double)); }}}
}

/*Function for allocating square 2D C/C++ arrays/matrices automatically.*/
void allocate2(double **&matrix, int dimension)
{
        matrix = (double**)calloc(dimension, sizeof(double *));
        for(int ii = 0; ii < dimension; ii++) {
                matrix[ii] = (double*)calloc(dimension, sizeof(double));}
}


/*Function for printing GSL matrices with some text.
The 3rd argument takes the size of the matrix (only square matrices supported).*/
void clever_printf(gsl_matrix *matrix, const char *text, int size)
{
	printf("\n%s\n", text);

	for (int i =0; i<size; i++)
		printf("\t%d", i+1);
	printf("\n");

	for (int i=0; i<size; i++)
	{
		    printf("%d  ", i+1);
				for (int j=0; j<size; j++)
				{
					printf("%.5lf ", gsl_matrix_get(matrix, i,j));
				}
				printf("\n");
	}
}


/*Overloaded function for computing the residual (root-mean square) of 2 square GSL matrices.
Used in Hartree-Fock to check for convergence.*/
double rms(gsl_matrix *new_, gsl_matrix *old_, int dimension)
{
        double res = 0;
            for (int mu=0; mu<dimension; mu++)
                for (int nu=0; nu<dimension; nu++)
                    res += (gsl_matrix_get(old_,mu ,nu) - gsl_matrix_get(new_, mu, nu))*(gsl_matrix_get(old_,mu ,nu) - gsl_matrix_get(new_, mu, nu));
            res = sqrt(res);
        return res;

}

/*Overloaded function for computing the residual (root-mean square) of 2 rank-4 tensors.
Used in coupled-cluster to check for convergence.*/
double rms(double ****new_, double ****old_, int dimension)
{
        double res = 0;
            for (int mu=0; mu<dimension; mu++)
                for (int nu=0; nu<dimension; nu++)
							  			for (int i=0; i<dimension; i++)
														for (int j=0; j<dimension; j++)
                    							res += pow(old_[mu][nu][i][j] - new_[mu][nu][i][j], 2);
            res = sqrt(res);
        return res;

}

/*Overloaded function for setting to zero 2D square C/C++ arrays/matrices.*/
void set_zero(double **matrix, int dimension)
{
        for (int i=0; i<dimension; i++)
                for (int j=0; j<dimension; j++)
                        matrix[i][j] = 0;
}

/*Overloaded function for setting to zero 4D *NxNxNxN* C/C++ arrays/tensors.*/
void set_zero(double ****tensor, int dimension)
{
        for (int i=0; i<dimension; i++)
                for (int j=0; j<dimension; j++)
                        for (int k=0; k<dimension; k++)
                                for (int l=0; l<dimension; l++)
                                        tensor[i][j][k][l] = 0;
}


/*Below there are the DIIS functions for HF, singles and doubles of CCSD.
DIIS can be found in here:
		Pulay, P. (1980). Convergence acceleration of iterative sequences. the case of scf iteration. Chemical Physics Letters, 73(2), 393-398
DIIS for CC can be found in here:
		G.E. Scuseria, T.J. Lee, and H.F. Schaefer, “Accelerating the conference of the coupled-cluster approach. The use of the DIIS method”, Chem. Phys. Lett. 130, 236 (1986).
*/


/*Function for extrapolating the new Fock matrix in Hartree-Fock using DIIS.
The last 5 Fock matrices must be provided as well as the 5 respective error matrices.
For now the number of DIIS vectors is hardcoded.
The return value is the extrapolated Fock matrix.*/
gsl_matrix *diisfunc(gsl_matrix *e1, gsl_matrix *e2, gsl_matrix *e3, gsl_matrix *e4, gsl_matrix *e5, gsl_matrix *F1, gsl_matrix *F2, gsl_matrix *F3, gsl_matrix *F4, gsl_matrix *F5)
{
/*The extrapolated Fock matrix is F' = sum_i (ciFi)
To find ci's, we construct the matrix B with Bij = Tr(ei*ej), because ei*ej is a matrix.
The system of linear equations that we solve to find ci's is:

		(B11 B12 ... B1m -1)     (c1)            ( 0 )
	  (B21 B22 ... B2m -1)     (c2)            ( 0 )
    (... ... ... ... -1)     (...)      =    (...)
		(Bm1 Bm2 ... Bmm -1)     (cm)            ( 0 )
		(-1  -1  ... -1   0)   (lambda)          (-1 )
*/

    double trace = 0;
    gsl_matrix *tmp = gsl_matrix_alloc(norb,norb); //ei*ej matrix
    gsl_vector *x = gsl_vector_alloc(6); //The ci's plus the lambda. 6 (six) because I am storing 5 error vectors and Fock matrices.

    gsl_matrix *B = gsl_matrix_alloc(6,6);
    gsl_vector *right = gsl_vector_alloc(6); //rhs of the system of the linear equations
    gsl_vector_set_zero(right);

//Computing B.
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e1, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e2, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e3, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 2, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e1, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e2, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e3, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 2, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e1, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e2, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e3, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e4, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e5, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e4, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e5, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e4, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e5, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e5, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e4, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e3, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e2, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 1, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e1, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 0, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e1, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 0, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e2, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 1, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e3, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e4, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e5, 0.0, tmp);
    for (int i=0; i<norb; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 4, trace);
//Finished computing tr(ei*ej).
//Starting to store the far-right and far-down elements of B.
    for (int i=0; i<6; i++)
    {
            gsl_matrix_set(B, 5, i, -1);
            gsl_matrix_set(B, i, 5, -1);
    }
    gsl_matrix_set(B, 5, 5, 0);
//B is ready.
    gsl_vector_set(right, 5, -1); //The vector on the rhs of the system of linear equations is initialized

//Solving the system
    gsl_permutation *p; int ss;
    x = gsl_vector_alloc(6);
    p = gsl_permutation_alloc(6);
    gsl_linalg_LU_decomp(B, p, &ss);
    gsl_linalg_LU_solve(B, p, right, x);
//Use of GSL functions that already exist.

//Storing again the Fock matrices, so that we do not change them accidentally.
    gsl_matrix *FF1 = gsl_matrix_alloc(norb, norb);
    gsl_matrix *FF2 = gsl_matrix_alloc(norb, norb);
    gsl_matrix *FF3 = gsl_matrix_alloc(norb, norb);
    gsl_matrix *FF4 = gsl_matrix_alloc(norb, norb);
    gsl_matrix *FF5 = gsl_matrix_alloc(norb, norb);

    for (int i=0; i<norb; i++) {
        for (int j=0; j<norb; j++) {
            gsl_matrix_set(FF1, i, j, gsl_matrix_get(F1, i, j));
            gsl_matrix_set(FF2, i, j, gsl_matrix_get(F2, i, j));
            gsl_matrix_set(FF3, i, j, gsl_matrix_get(F3, i, j));
            gsl_matrix_set(FF4, i, j, gsl_matrix_get(F4, i, j));
            gsl_matrix_set(FF5, i, j, gsl_matrix_get(F5, i, j));}}

//Extrapolation
    gsl_matrix_scale (FF1, gsl_vector_get(x,0)); //Multiply by c1
    gsl_matrix_scale (FF2, gsl_vector_get(x,1)); //etc
    gsl_matrix_scale (FF3, gsl_vector_get(x,2));
    gsl_matrix_scale (FF4, gsl_vector_get(x,3));
    gsl_matrix_scale (FF5, gsl_vector_get(x,4));
    gsl_matrix_add(FF1, FF2); //Add evrerything to FF1 and return it.
    gsl_matrix_add(FF1, FF3);
    gsl_matrix_add(FF1, FF4);
    gsl_matrix_add(FF1, FF5);
//Done. Result is on FF1.
		free(tmp); free(x); free(B);
    return FF1;

}

/*Function for extrapolating the new single-amplitudes in Coupled-cluster singles-doubles (CCSD) using DIIS.
There are no comments because the same apply as in the HF case.
The last 5 singles matrices must be provided as well as the 5 respective error matrices.
For now the number of DIIS vectors is hardcoded.
The return value is the extrapolated singles matrix.*/
double **singlesdiisfunc(gsl_matrix *e1, gsl_matrix *e2, gsl_matrix *e3, gsl_matrix *e4, gsl_matrix *e5, gsl_matrix *F1, gsl_matrix *F2, gsl_matrix *F3, gsl_matrix *F4, gsl_matrix *F5)
{
    double trace = 0;
    gsl_matrix *tmp = gsl_matrix_alloc(dim,dim);
    gsl_vector *x = gsl_vector_alloc(6);

    gsl_matrix *B = gsl_matrix_alloc(6,6); //6 giati sto DIIS mou xrisimopoio 5 errors
    gsl_vector *right = gsl_vector_alloc(6);
    gsl_vector_set_zero(right);

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e1, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e2, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e3, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 2, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e1, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e2, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e3, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 2, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e1, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e2, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e3, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e4, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e5, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e4, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e5, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e4, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e5, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e5, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e4, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e3, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e2, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 1, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e1, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 0, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e1, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 0, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e2, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 1, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e3, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e4, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e5, 0.0, tmp);
    for (int i=0; i<dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 4, trace);

    for (int i=0; i<6; i++)
    {
            gsl_matrix_set(B, 5, i, -1);
            gsl_matrix_set(B, i, 5, -1);
    }
    gsl_matrix_set(B, 5, 5, 0);
    gsl_vector_set(right, 5, -1);
    gsl_permutation *p; int ss;
    x = gsl_vector_alloc(6);
    p = gsl_permutation_alloc(6);
    gsl_linalg_LU_decomp(B, p, &ss);
    gsl_linalg_LU_solve(B, p, right, x);

    gsl_matrix *FF1 = gsl_matrix_alloc(dim, dim);
    gsl_matrix *FF2 = gsl_matrix_alloc(dim, dim);
    gsl_matrix *FF3 = gsl_matrix_alloc(dim, dim);
    gsl_matrix *FF4 = gsl_matrix_alloc(dim, dim);
    gsl_matrix *FF5 = gsl_matrix_alloc(dim, dim);

    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            gsl_matrix_set(FF1, i, j, gsl_matrix_get(F1, i, j));
            gsl_matrix_set(FF2, i, j, gsl_matrix_get(F2, i, j));
            gsl_matrix_set(FF3, i, j, gsl_matrix_get(F3, i, j));
            gsl_matrix_set(FF4, i, j, gsl_matrix_get(F4, i, j));
            gsl_matrix_set(FF5, i, j, gsl_matrix_get(F5, i, j));}}

    gsl_matrix_scale (FF1, gsl_vector_get(x,0));
    gsl_matrix_scale (FF2, gsl_vector_get(x,1));
    gsl_matrix_scale (FF3, gsl_vector_get(x,2));
    gsl_matrix_scale (FF4, gsl_vector_get(x,3));
    gsl_matrix_scale (FF5, gsl_vector_get(x,4));
    gsl_matrix_add(FF1, FF2);
    gsl_matrix_add(FF1, FF3);
    gsl_matrix_add(FF1, FF4);
    gsl_matrix_add(FF1, FF5);

		double **ts;
		allocate2(ts, dim);

		for (int i=0; i<dim; i++)
			for (int j=0; j<dim; j++)
					ts[i][j] = gsl_matrix_get(FF1, i, j);

		free(tmp); free(x); free(B);
    return ts;

}


/*Function for extrapolating the new double-amplitudes in Coupled-cluster singles-doubles (CCSD) using DIIS.
There are no comments because the same apply as in the HF case.
The last 5 doubles matrices must be provided as well as the 5 respective error matrices.
For now the number of DIIS vectors is hardcoded.
The return value is the extrapolated doubles tensor.
Careful: The doubles are a rank-4 tensor. The way it turns into a matrix has a comment inside the function below.*/
double ****doublesdiisfunc(gsl_matrix *e1, gsl_matrix *e2, gsl_matrix *e3, gsl_matrix *e4, gsl_matrix *e5, gsl_matrix *F1, gsl_matrix *F2, gsl_matrix *F3, gsl_matrix *F4, gsl_matrix *F5)
{
    double trace = 0;
    gsl_matrix *tmp = gsl_matrix_alloc(dim*dim,dim*dim);
    gsl_vector *x = gsl_vector_alloc(6);

    gsl_matrix *B = gsl_matrix_alloc(6,6); //6 giati sto DIIS mou xrisimopoio 5 errors
    gsl_vector *right = gsl_vector_alloc(6);
    gsl_vector_set_zero(right);

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e1, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e2, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e3, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 2, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e1, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e2, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e3, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 2, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e1, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 0, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e2, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 1, trace);
    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e3, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e4, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e1, e5, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 0, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e4, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e2, e5, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 1, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e4, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e3, e5, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 2, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e5, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 4, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e4, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e3, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e2, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 1, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e1, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 0, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e1, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 0, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e2, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 1, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e3, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 2, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e4, e4, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 3, 3, trace);

    trace = 0;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, e5, e5, 0.0, tmp);
    for (int i=0; i<dim*dim; i++)
            trace += gsl_matrix_get(tmp, i, i);
    gsl_matrix_set(B, 4, 4, trace);

    for (int i=0; i<6; i++)
    {
            gsl_matrix_set(B, 5, i, -1);
            gsl_matrix_set(B, i, 5, -1);
    }
    gsl_matrix_set(B, 5, 5, 0);
    gsl_vector_set(right, 5, -1);
    gsl_permutation *p; int ss;
    x = gsl_vector_alloc(6);
    p = gsl_permutation_alloc(6);
    gsl_linalg_LU_decomp(B, p, &ss);
    gsl_linalg_LU_solve(B, p, right, x);

    gsl_matrix *FF1 = gsl_matrix_alloc(dim*dim, dim*dim);
    gsl_matrix *FF2 = gsl_matrix_alloc(dim*dim, dim*dim);
    gsl_matrix *FF3 = gsl_matrix_alloc(dim*dim, dim*dim);
    gsl_matrix *FF4 = gsl_matrix_alloc(dim*dim, dim*dim);
    gsl_matrix *FF5 = gsl_matrix_alloc(dim*dim, dim*dim);

    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
					for (int k=0; k<dim; k++) {
							for (int l=0; l<dim; l++) {
            gsl_matrix_set(FF1, i+j*dim, k+l*dim, gsl_matrix_get(F1, i+j*dim, k+l*dim));
            gsl_matrix_set(FF2, i+j*dim, k+l*dim, gsl_matrix_get(F2, i+j*dim, k+l*dim));
            gsl_matrix_set(FF3, i+j*dim, k+l*dim, gsl_matrix_get(F3, i+j*dim, k+l*dim));
            gsl_matrix_set(FF4, i+j*dim, k+l*dim, gsl_matrix_get(F4, i+j*dim, k+l*dim));
            gsl_matrix_set(FF5, i+j*dim, k+l*dim, gsl_matrix_get(F5, i+j*dim, k+l*dim));}}}}

    gsl_matrix_scale (FF1, gsl_vector_get(x,0));
    gsl_matrix_scale (FF2, gsl_vector_get(x,1));
    gsl_matrix_scale (FF3, gsl_vector_get(x,2));
    gsl_matrix_scale (FF4, gsl_vector_get(x,3));
    gsl_matrix_scale (FF5, gsl_vector_get(x,4));
    gsl_matrix_add(FF1, FF2);
    gsl_matrix_add(FF1, FF3);
    gsl_matrix_add(FF1, FF4);
    gsl_matrix_add(FF1, FF5);

		double ****td;
		allocate4(td, dim);

		for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++)
					for (int k=0; k<dim; k++)
							for (int l=0; l<dim; l++)
									td[i][j][k][l] = gsl_matrix_get(FF1, i+j*dim, k+l*dim); }
                  /* i+j*dim, k+l*dim is used to turn a tensor of dimxdimxdimxdim to a matrix of dim**2xdim**2.
									The order of the loops *must* be i,j,k,l and also the way everything must be the same when you call doublesdiisfunc() from the doccsd().*/

		free(tmp); free(x); free(B);
    return td;

}

/*Function for computing the Trace(DS) (density x overlap) which gives the number of occupied orbitals.
Used mostly as a sanity check.
Note: For non-molecular hamiltonians we can directly use Trace(D).*/
void findTraceDensity(gsl_matrix *Ssqrt, gsl_matrix *D)
{
        gsl_matrix *SDS = gsl_matrix_alloc(norb, norb);
        gsl_matrix *SD = gsl_matrix_alloc(norb, norb);
        gsl_matrix *invSsqrt = gsl_matrix_alloc(norb, norb);
        gsl_matrix *Ssqrt_for_inv = gsl_matrix_alloc(norb, norb);

        for (int i=0;i<norb;i++)
                for (int j=0; j<norb;j++) //Needed because linalg_lu_decomp destroys the old matrix.
                        gsl_matrix_set(Ssqrt_for_inv,i,j,gsl_matrix_get(Ssqrt,i,j));

        int ss;
        gsl_permutation * p = gsl_permutation_alloc (norb); //Actual inversing of Ssqrt
        gsl_linalg_LU_decomp (Ssqrt_for_inv, p, &ss);
        gsl_linalg_LU_invert (Ssqrt_for_inv, p, invSsqrt);

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invSsqrt, D, 0.0, SD); //S^-1/2 x D
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, SD, invSsqrt, 0.0, SDS); // S^-1/2 x D x S-1/2 = DS^-1
				//The reason we have DS-1 depends on how the overlap integrals are provided.

      //  clever_printf(SDS, "Density matrix.", norb);
        double noccupied=0;
        for (int i=0;i<norb;i++)
                noccupied += gsl_matrix_get(SDS,i,i); //Computing the trace of DS
        printf("trace = %lf\n", noccupied);
				free(SDS); free(SD); free(invSsqrt); free(Ssqrt_for_inv);
}

/*Function for transforming the two-electron integrals from the AO basis to spatial-orbitals.
The formula used is: (p q|r s) = sigma_mu c[p][mu] sigma_nu c[q][nu] sigma_lambda c[r][lambda] sigma_sigma c[s][sigma] (mu nu|lambda sigma)
where the rhs refers to AO basis with C's being the MO coefficients from HF.
MP2 can use these directly in this implementation.
The transformation is needed if we want to use spin-orbitals later.
Be careful: eri change *permanently* to the MO basis.*/
void transformIntegrals(double ****eri, gsl_matrix *C)
{
    double sum=0, ****temperi; //temperi is needed to break the summation in 4 steps, one for each index.

		allocate4(temperi, norb+2);

    for (int j4=0;j4<norb;j4++){ for (int j3=0;j3<norb;j3++) { for (int j2=0; j2<norb; j2++) { for (int j1=0; j1<norb;j1++){
                                        sum = 0;
                                        for (int ii=0; ii<norb; ii++) {
                                                sum += gsl_matrix_get(C,ii,j1)*eri[ii+1][j2+1][j3+1][j4+1];
                                        }
                                        temperi[j1+1][j2+1][j3+1][j4+1] = sum; } } } }

        for (int j4=0;j4<norb;j4++){ for (int j3=0;j3<norb;j3++) { for (int j2=0; j2<norb; j2++) { for (int j1=0; j1<norb;j1++){
                                        sum = 0;
                                        for (int ii=0; ii<norb; ii++) {
                                                sum += gsl_matrix_get(C,ii,j2)*temperi[j1+1][ii+1][j3+1][j4+1];
                                        }
                                        eri[j1+1][j2+1][j3+1][j4+1] = sum; } } } }

        for (int j4=0;j4<norb;j4++){ for (int j3=0;j3<norb;j3++) { for (int j2=0; j2<norb; j2++) { for (int j1=0; j1<norb;j1++){
                                        sum = 0;
                                        for (int ii=0; ii<norb; ii++) {
                                                sum += gsl_matrix_get(C,ii,j3)*eri[j1+1][j2+1][ii+1][j4+1];
                                        }
                                        temperi[j1+1][j2+1][j3+1][j4+1] = sum; } } } }

        for (int j4=0;j4<norb;j4++){ for (int j3=0;j3<norb;j3++) { for (int j2=0; j2<norb; j2++) { for (int j1=0; j1<norb;j1++){
                                        sum = 0;
                                        for (int ii=0; ii<norb; ii++) {
                                                sum += gsl_matrix_get(C,ii,j4)*temperi[j1+1][j2+1][j3+1][ii+1];
                                        }
                                        eri[j1+1][j2+1][j3+1][j4+1] = sum; } } } }

			  free(temperi);
}

/*Function for transforming spatial-orbitals to spin-orbitals.
The spin orbitals are 16 times the spatial orbitals.
The formula is <pq|rs> = (pr|qs) * integral-of-spins.
soeri will contain the final spin-orbital two electron integrals.*/
void spinOrbitals(double ****soeri, double ****eri)
{
    set_zero(soeri, dim+1);

    for(int p=1; p <= dim; p++)
        for(int q=1; q <= dim; q++)
            for(int r=1; r <= dim; r++)
                for(int s=1; s <= dim; s++)
                {
                      double value1, value2;
                      value1 = eri[(p+1)/2][(r+1)/2][(q+1)/2][(s+1)/2] * (p%2 == r%2) * (q%2 == s%2);
                      value2 = eri[(p+1)/2][(s+1)/2][(q+1)/2][(r+1)/2] * (p%2 == s%2) * (q%2 == r%2);
                      soeri[p-1][q-1][r-1][s-1] = value1 - value2;
                }

}

/*Function for transforming AO basis Fock matrix to spin-orbital basis Fock matrix.
This is done by just doubling the size of the matrix and store in the diagonal the energies (each one twice) from HF.*/
void spinOFock(gsl_matrix *soFock , gsl_vector *epsilon)
{

    for (int i=0; i<dim; i++)
    {
        gsl_matrix_set(soFock, i, i, gsl_vector_get(epsilon, i/2));
        gsl_matrix_set(soFock, i+1, i+1, gsl_vector_get(epsilon, i/2));
        ++i;
    }

}

//---------------------------------------
/*Below there are some functions for the implementation of CCSD.
These functions and this implementation is based on:
J.F. Stanton, J. Gauss, J.D. Watts, and R.J. Bartlett, J. Chem. Phys. volume 94, pp. 4334-4345 (1991).
I will refer to it as [1].

Careful: a,b,c,d,... correspond to virtual and i,j,k,l,... to occupied.
         This means that the order *must* be the same and not mess up ai with ia. */
//---------------------------------------

/*Eq. 9 of [1].*/
double taut(int a, int b,int i,int j, double **ts, double ****td)
{
    return td[a][b][i][j]+0.5*(ts[a][i]*ts[b][j]-ts[b][i]*ts[a][j]);
}

/*Eq. 10 of [1].*/
double tau(int a, int b,int i,int j, double **ts, double ****td)
{
    return td[a][b][i][j]+ts[a][i]*ts[b][j]-ts[b][i]*ts[a][j];
}

/*Basic equation of CC for computing the energy, after getting the
singles, doubles, the Fock matrix and the two-electron integrals.
Everything is in terms of spin-orbitals.*/
double ccsdCalc(gsl_matrix *soFock, double ****soeri, double **ts, double ****td)
{
  double eccsd = 0.0;
  for (int i=0; i<Nelec; i++) {
    for (int a=Nelec; a<dim; a++) {
      eccsd += gsl_matrix_get(soFock, i, a)*ts[a][i];
      for (int j=0; j<Nelec; j++) {
        for (int b=Nelec; b<dim; b++) {
          eccsd += 0.25*soeri[i][j][a][b]*td[a][b][i][j] + 0.5*soeri[i][j][a][b]*ts[a][i]*ts[b][j];}}}}

  return eccsd;
}

/*Function for updating all the intermediates (eqs. 3 - 8) of [1].
Nelec to dim means virtual orbitals and 0 to Nelec means occupied virtuals.*/
void updateIntermediates(gsl_matrix *soFock, double ****soeri, double **Fae, double **Fmi, double **Fme, double ****Wmnij, double ****Wabef, double ****Wmbej, double **ts, double ****td)
{
        set_zero(Fae, dim);
        set_zero(Fmi, dim);
        set_zero(Fme, dim);
        set_zero(Wmnij, dim);
        set_zero(Wabef, dim);
        set_zero(Wmbej, dim);

    for (int a=Nelec; a<dim; a++){
        for (int e=Nelec; e<dim; e++){
            Fae[a][e] = (1-(a==e))*gsl_matrix_get(soFock, a,e);
            for (int m=0; m<Nelec; m++){
                Fae[a][e] += -0.5*gsl_matrix_get(soFock, m, e) * ts[a][m];
                for (int f=Nelec; f<dim; f++){
                    Fae[a][e] += ts[f][m]*soeri[m][a][f][e];
                    for (int n=0; n<Nelec; n++){
                        Fae[a][e] += -0.5*taut(a,f,m,n, ts, td)*soeri[m][n][e][f];}}}}} //Eq. 3 of [1].

    for (int m=0; m<Nelec; m++){
        for (int i=0; i<Nelec; i++){
            Fmi[m][i] = (1-(m==i))*gsl_matrix_get(soFock, m,i);
            for (int e=Nelec; e<dim; e++){
                Fmi[m][i] += 0.5*gsl_matrix_get(soFock, m, e) * ts[e][i];
                for (int n=0; n<Nelec; n++){
                    Fmi[m][i] += ts[e][n]*soeri[m][n][i][e];
                    for (int f=Nelec; f<dim; f++){
                        Fmi[m][i] += 0.5*taut(e,f, i, n, ts, td)*soeri[m][n][e][f]; }}}}} //Eq. 4 of [1].


    for (int m=0; m<Nelec; m++){
        for (int e=Nelec; e<dim; e++){
            Fme[m][e] = gsl_matrix_get(soFock, m, e);
            for (int n=0; n<Nelec; n++){
                for (int f=Nelec; f<dim; f++){
                    Fme[m][e] += ts[f][n]*soeri[m][n][e][f]; }}}} //Eq. 5 of [1].


    for (int m=0; m<Nelec; m++){
        for (int n=0; n<Nelec; n++){
            for (int i=0; i<Nelec; i++){
                for (int j=0; j<Nelec; j++){
                    Wmnij[m][n][i][j] = soeri[m][n][i][j];
                    for (int e=Nelec; e<dim; e++){
                        Wmnij[m][n][i][j] += ts[e][j]*soeri[m][n][i][e] - ts[e][i]*soeri[m][n][j][e];
                        for (int f=Nelec; f<dim; f++){
                            Wmnij[m][n][i][j] += 0.25*tau(e,f, i, j, ts, td)*soeri[m][n][e][f]; }}}}}} //Eq. 6 of [1].


    for (int a=Nelec; a<dim; a++){
        for (int b=Nelec; b<dim; b++){
            for (int e=Nelec; e<dim; e++){
                for (int f=Nelec; f<dim; f++){
                    Wabef[a][b][e][f] = soeri[a][b][e][f];
                    for (int m=0; m<Nelec; m++){
                        Wabef[a][b][e][f] += -ts[b][m]*soeri[a][m][e][f] + ts[a][m]*soeri[b][m][e][f];
                        for (int n=0; n<Nelec; n++){
                            Wabef[a][b][e][f] += 0.25*tau(a,b,m,n, ts, td )*soeri[m][n][e][f]; }}}}}} //Eq. 7 of [1].

    for (int m=0; m<Nelec; m++){
        for (int b=Nelec; b<dim; b++){
            for (int e=Nelec; e<dim; e++){
                for (int j=0; j<Nelec; j++){
                    Wmbej[m][b][e][j] = soeri[m][b][e][j];
                    for (int f=Nelec; f<dim; f++){
                        Wmbej[m][b][e][j] += ts[f][j]*soeri[m][b][e][f];}
                        for (int n=0; n<Nelec; n++){
                            Wmbej[m][b][e][j] -= ts[b][n]*soeri[m][n][e][j];
                            for (int f=Nelec; f<dim; f++){
                                Wmbej[m][b][e][j] += -(0.5*td[f][b][j][n]+ts[f][j]*ts[b][n])*soeri[m][n][e][f]; }}}}}} //Eq. 8 of [1].
}


/*Eq. 1 of [1]. Computing new singles.*/
void makeT1( double **ts, double ****td, double **tsnew, double ****soeri, gsl_matrix *soFock, double **Fae, double **Fmi, double **Fme, double **Dai)
{
    set_zero(tsnew, dim);

    for (int a=Nelec; a<dim; a++){
        for (int i=0; i<Nelec; i++){
            tsnew[a][i] = gsl_matrix_get(soFock, i, a);
           for (int e=Nelec; e<dim; e++) {
                tsnew[a][i] += ts[e][i]*Fae[a][e]; }
            for (int m=0; m<Nelec; m++){
                tsnew[a][i] += -ts[a][m]*Fmi[m][i];
                for (int e=Nelec; e<dim; e++){
                    tsnew[a][i] += td[a][e][i][m]*Fme[m][e];
                    for (int f=Nelec; f<dim; f++){
                      tsnew[a][i] += -0.5*td[e][f][i][m]*soeri[m][a][e][f];}
                    for (int n=0; n<Nelec; n++){
                      tsnew[a][i] += -0.5*td[a][e][m][n]*soeri[n][m][e][i];}}}
            for (int n=0; n<Nelec; n++){
                 for (int f=Nelec; f<dim; f++){
                    tsnew[a][i] += -ts[f][n]*soeri[n][a][i][f];}}
                tsnew[a][i] /= Dai[a][i];}}
}

/*Eq. 2 of [1]. Computing new doubles.*/
void makeT2( double **ts, double ****td, double ****tdnew, double ****soeri, gsl_matrix *soFock, double **Fae, double **Fmi, double **Fme, double ****Wmnij, double ****Wabef, double ****Wmbej, double ****Dabij)
{
        set_zero(tdnew, dim);

        for (int a=Nelec; a<dim; a++) {
          for (int b=Nelec; b<dim; b++) {
            for (int i=0; i<Nelec; i++) {
              for (int j=0; j<Nelec; j++) {
                tdnew[a][b][i][j] += soeri[i][j][a][b];
               for (int e=Nelec; e<dim; e++) {
                  tdnew[a][b][i][j] += td[a][e][i][j]*Fae[b][e] - td[b][e][i][j]*Fae[a][e];
                  for (int m=0; m<Nelec; m++) {
                    tdnew[a][b][i][j] += -0.5*td[a][e][i][j]*ts[b][m]*Fme[m][e] + 0.5*td[b][e][i][j]*ts[a][m]*Fme[m][e];
                    continue; }}
                for (int m=0; m<Nelec; m++) {
                  tdnew[a][b][i][j] += -td[a][b][i][m]*Fmi[m][j] + td[a][b][j][m]*Fmi[m][i];
                  for (int e=Nelec; e<dim; e++) {
                    tdnew[a][b][i][j] += -0.5*td[a][b][i][m]*ts[e][j]*Fme[m][e] + 0.5*td[a][b][j][m]*ts[e][i]*Fme[m][e];
                    continue; }}
                for (int e=Nelec; e<dim; e++) {
                  tdnew[a][b][i][j] += ts[e][i]*soeri[a][b][e][j] - ts[e][j]*soeri[a][b][e][i];
                  for (int f=Nelec; f<dim; f++) {
                    tdnew[a][b][i][j] += 0.5*tau(e,f,i,j, ts ,td)*Wabef[a][b][e][f];
                    continue; }}
                for (int m=0; m<Nelec; m++) {
                  tdnew[a][b][i][j] += -ts[a][m]*soeri[m][b][i][j] + ts[b][m]*soeri[m][a][i][j];
                  for (int e=Nelec; e<dim; e++) {
                    tdnew[a][b][i][j] +=  td[a][e][i][m]*Wmbej[m][b][e][j] - ts[e][i]*ts[a][m]*soeri[m][b][e][j];
                    tdnew[a][b][i][j] += -td[a][e][j][m]*Wmbej[m][b][e][i] + ts[e][j]*ts[a][m]*soeri[m][b][e][i];
                    tdnew[a][b][i][j] += -td[b][e][i][m]*Wmbej[m][a][e][j] + ts[e][i]*ts[b][m]*soeri[m][a][e][j];
                    tdnew[a][b][i][j] +=  td[b][e][j][m]*Wmbej[m][a][e][i] - ts[e][j]*ts[b][m]*soeri[m][a][e][i];
                    continue; }
                  for (int n=0; n<Nelec; n++) {
                    tdnew[a][b][i][j] += 0.5*tau(a,b,m,n, ts, td)*Wmnij[m][n][i][j];
                    continue; }}
                tdnew[a][b][i][j] /= Dabij[a][b][i][j];}}}}
}

/*Actual function for performing the cycles of CCSD.
If we just do one iteration, we get MP2.*/
double doccsd(gsl_matrix *soFock, double ****soeri, int maxiter, double **ts, double ****td)
{
  double **Fae, **Fmi, **Fme, **Dai; //Intermediates
	allocate2(Fae, dim);  allocate2(Fmi, dim);  allocate2(Fme, dim);  allocate2(Dai, dim);

	double ****Wmnij, ****Wabef, ****Wmbej, ****Dabij; //Intermediates
	  allocate4(Wmnij, dim);  allocate4(Wabef, dim);  allocate4(Wmbej, dim);  allocate4(Dabij, dim);

	set_zero(Fae, dim); set_zero(Fmi, dim); set_zero(Fme, dim); set_zero(Dai, dim);
	 set_zero(Wmnij, dim); set_zero(Wabef, dim); set_zero(Wmbej, dim); set_zero(Dabij, dim);

   double res = 0;
   int iter = 1;
   for (int a=Nelec; a<dim; a++) //i, j occupied, a,b virtual
        for (int b=Nelec; b<dim; b++)
            for (int i=0; i<Nelec; i++)
                for (int j=0; j<Nelec;j++)
                    td[a][b][i][j] += soeri[i][j][a][b]/(gsl_matrix_get(soFock, i,i) + gsl_matrix_get(soFock, j,j) - gsl_matrix_get(soFock, a,a) - gsl_matrix_get(soFock, b,b));
								  //The guess for the doubles is from MP2. We can also get MP2 amplitudes with one iteration if we start from zero.

    double mmp2=0;
    for (int a=Nelec; a<dim; a++) //i, j occupied, a,b virtual
        for (int b=Nelec; b<dim; b++)
            for (int i=0; i<Nelec; i++)
                for (int j=0; j<Nelec;j++)
                    mmp2 += 0.25*soeri[i][j][a][b]*td[a][b][i][j]; //Computing MP2 energy from the guesses.

    printf("E(MP2) = %.12lf\n", mmp2);
    if (maxiter == 1) //Checking if MP2 is chosen.
        return mmp2;

    printf("Starting CCSD\n");

    for (int a=Nelec; a<dim; a++)
        for (int b=Nelec; b<dim; b++)
            for (int i=0; i<Nelec; i++)
                for (int j=0; j<Nelec; j++)
                    Dabij[a][b][i][j] = gsl_matrix_get(soFock, i,i) + gsl_matrix_get(soFock, j,j) - gsl_matrix_get(soFock, a,a) - gsl_matrix_get(soFock, b,b);
										//Eq. 13 from [1].
    for (int a=Nelec; a<dim; a++)
        for (int i=0; i<Nelec; i++)
            Dai[a][i] = gsl_matrix_get(soFock, i,i) - gsl_matrix_get(soFock, a,a);
						//Eq. 12 from [1].

        double  **tsnew, ****tdnew;
        allocate4(tdnew, dim);
				allocate2(tsnew, dim);

				set_zero(tsnew, dim);
				set_zero(tdnew, dim);

	double eccsd = 0;
	res = 1.0;
  iter=1;
	int diis = 1;

//Error matrices and singles for DIIS.
	gsl_matrix *e1 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *e2 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *e3 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *e4 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *e5 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *T1 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *T2 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *T3 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *T4 = gsl_matrix_alloc(dim,dim);
	gsl_matrix *T5 = gsl_matrix_alloc(dim,dim);

//Error matrices and doubles for DIIS.
	gsl_matrix *ee1 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *ee2 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *ee3 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *ee4 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *ee5 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *TT1 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *TT2 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *TT3 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *TT4 = gsl_matrix_alloc(dim*dim,dim*dim);
	gsl_matrix *TT5 = gsl_matrix_alloc(dim*dim,dim*dim);


    while (res > 1e-11 && iter <= maxiter)
    {
        updateIntermediates(soFock, soeri, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej, ts, td);
        makeT1(ts, td, tsnew, soeri, soFock, Fae, Fmi, Fme, Dai);
        makeT2(ts, td, tdnew, soeri, soFock, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej, Dabij);

				eccsd = ccsdCalc(soFock, soeri, ts, td);
				res = rms(td, tdnew, dim);

				if (res < 1e-5) //If the solution has sufficiently started converging, turn on DIIS.
				{
								if (diis % 5 == 1) //This is just way to see where to store the new error matrix and singles/doubles.
								{									//The same applies to the other 4 cases, because we store 5 DIIS vectors.
												for (int i=0; i<dim; i++)
																for (int j=0;j<dim;j++)
																			{	gsl_matrix_set(e1, i, j, ts[i][j] - tsnew[i][j]); gsl_matrix_set(T1, i, j, tsnew[i][j]); }

												for (int i=0; i<dim; i++)
													for (int j=0; j<dim; j++)
															for (int k=0; k<dim; k++)
																	for (int l=0;l<dim; l++)
																			{ gsl_matrix_set(ee1, i+j*dim, k+l*dim, td[i][j][k][l] - tdnew[i][j][k][l]); gsl_matrix_set(TT1, i+j*dim, k+l*dim, tdnew[i][j][k][l]); }
																			//i+j*dim, k+l*dim to convert from rank-4 tensor to matrix. The same *must* be used in *diisfunc.
								}
								else if (diis % 5 == 2)
								{
												for (int i=0; i<dim; i++)
																for (int j=0;j<dim;j++)
																			{	gsl_matrix_set(e2, i, j, ts[i][j] - tsnew[i][j]); gsl_matrix_set(T2, i, j, tsnew[i][j]); }

												for (int i=0; i<dim; i++)
													for (int j=0; j<dim; j++)
															for (int k=0; k<dim; k++)
																	for (int l=0;l<dim; l++)
																			{ gsl_matrix_set(ee2, i+j*dim, k+l*dim, td[i][j][k][l] - tdnew[i][j][k][l]); gsl_matrix_set(TT2, i+j*dim, k+l*dim, tdnew[i][j][k][l]); }
								}
								else if (diis % 5 == 3)
								{
												for (int i=0; i<dim; i++)
																for (int j=0;j<dim;j++)
																			{	gsl_matrix_set(e3, i, j, ts[i][j] - tsnew[i][j]); gsl_matrix_set(T3, i, j, tsnew[i][j]); }

												for (int i=0; i<dim; i++)
													for (int j=0; j<dim; j++)
															for (int k=0; k<dim; k++)
																	for (int l=0;l<dim; l++)
																			{ gsl_matrix_set(ee3, i+j*dim, k+l*dim, td[i][j][k][l] - tdnew[i][j][k][l]); gsl_matrix_set(TT3, i+j*dim, k+l*dim, tdnew[i][j][k][l]); }
								}
								else if (diis % 5 == 4)
								{
												for (int i=0; i<dim; i++)
																for (int j=0;j<dim;j++)
																			{	gsl_matrix_set(e4, i, j, ts[i][j] - tsnew[i][j]); gsl_matrix_set(T4, i, j, tsnew[i][j]);}

												for (int i=0; i<dim; i++)
													for (int j=0; j<dim; j++)
															for (int k=0; k<dim; k++)
																	for (int l=0;l<dim; l++)
																			{ gsl_matrix_set(ee4, i+j*dim, k+l*dim, td[i][j][k][l] - tdnew[i][j][k][l]); gsl_matrix_set(TT4, i+j*dim, k+l*dim, tdnew[i][j][k][l]); }
								}
								else if (diis % 5 == 0)
								{
													for (int i=0; i<dim; i++)
																	for (int j=0;j<dim;j++)
																				{	gsl_matrix_set(e5, i, j, ts[i][j] - tsnew[i][j]); gsl_matrix_set(T5, i, j, tsnew[i][j]); }

													for (int i=0; i<dim; i++)
														for (int j=0; j<dim; j++)
																for (int k=0; k<dim; k++)
																		for (int l=0;l<dim; l++)
																				{ gsl_matrix_set(ee5, i+j*dim, k+l*dim, td[i][j][k][l] - tdnew[i][j][k][l]); gsl_matrix_set(TT5, i+j*dim, k+l*dim, tdnew[i][j][k][l]); }
								}
								if (diis >= 5) //If we have at least 5 matrices for DIIS.
								{
												ts = singlesdiisfunc(e1, e2, e3, e4, e5, T1, T2, T3, T4, T5);
												td = doublesdiisfunc(ee1, ee2, ee3, ee4, ee5, TT1, TT2, TT3, TT4, TT5);
								}
								++diis;
				}
				if (diis <= 5) { //If we still have not turned on DIIS.
			        for (int i=0; i<dim; i++)
			            for (int j=0; j<dim; j++)
			                ts[i][j] = tsnew[i][j];

			        for (int i=0; i<dim; i++)
			            for (int j=0; j<dim; j++)
			                for (int k=0; k<dim; k++)
			                    for (int l=0; l<dim; l++)
			                        td[i][j][k][l] = tdnew[i][j][k][l]; }

				if (iter<10) //Just some "fancy" printing.
						printf("0%d\t   %.12lf\t %.12lf\n", iter,eccsd, res);
				else
						printf("%2d\t   %.12lf\t %.12lf\n", iter,eccsd, res);
        ++iter;
   }

	 free(Fae); free(Fmi); free(Fme);  free(Dai);
 	 free(Wmnij); free(Wabef); free(Wmbej); free(Dabij);

   if (iter <= maxiter+1 && maxiter != 1)
   {
    printf("CCSD converged.\n");
    return eccsd;
   }
   else
   {
    printf("CCSD NOT converged.\n");
    return 0;
   }

}
//End of CCSD.
//------------------------------------------------------------------------------

/*Starting the implementation of perturbative triples correction to CCSD, aka. CCSD(T). Based on:
 		Raghavachari, Krishnan. “Historical Perspective on: A Fifth-Order Perturbation Comparison of Electron Correlation Theories
 		[Volume 157, Issue 6, 26 May 1989, Pages 479–483].” Chemical Physics Letters, vol. 589, 2013, pp. 35–36.*/

/*The denominator in the energy expression of CCSD(T).*/
double denom(int i, int j, int k, int a, int b, int c, gsl_matrix *soFock)
{
        return (gsl_matrix_get(soFock, i, i) + gsl_matrix_get(soFock, j, j) + gsl_matrix_get(soFock, k, k) - gsl_matrix_get(soFock, a, a) - gsl_matrix_get(soFock, b, b) - gsl_matrix_get(soFock, c, c));
}

/*Actual function for performing CCSD(T).
Keep in mind that CCSD(T) is *not* iterative.
The equations can be found at: http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project6 */
double doccsdt(gsl_matrix *soFock, double ****soeri, double **ts, double ****td)
{
     double et = 0, tttd, tttc;
     for (int i=0; i<Nelec; i++){
        for (int j=0; j<Nelec; j++){
            for (int k=0; k<Nelec; k++){
                for (int a=Nelec; a<dim; a++){
                    for (int b=Nelec; b<dim; b++){
                        for (int c=Nelec; c<dim; c++){
/*Disconnected triples on the fly*/tttd = (ts[a][i]*soeri[j][k][b][c]-ts[a][j]*soeri[i][k][b][c]-ts[a][k]*soeri[j][i][b][c]-ts[b][i]*soeri[j][k][a][c]+ts[b][j]*soeri[i][k][a][c]+ts[b][k]*soeri[j][i][a][c]-ts[c][i]*soeri[j][k][b][a]+ts[c][j]*soeri[i][k][b][a]+ts[c][k]*soeri[j][i][b][a])/denom(i,j,k,a,b,c,soFock);
/*Connected triples on the fly*/ tttc = 0;
                                for (int e=Nelec; e<dim; e++) {
                                tttc += (td[a][e][j][k]*soeri[e][i][b][c]-td[a][e][i][k]*soeri[e][j][b][c]-td[a][e][j][i]*soeri[e][k][b][c]-td[b][e][j][k]*soeri[e][i][a][c]+td[b][e][i][k]*soeri[e][j][a][c]+td[b][e][j][i]*soeri[e][k][a][c]-td[c][e][j][k]*soeri[e][i][b][a]+td[c][e][i][k]*soeri[e][j][b][a]+td[c][e][j][i]*soeri[e][k][b][a])/denom(i,j,k,a,b,c,soFock); }
                                for (int m=0; m<Nelec; m++) {
                                tttc -= (td[b][c][i][m]*soeri[m][a][j][k]-td[b][c][j][m]*soeri[m][a][i][k]-td[b][c][k][m]*soeri[m][a][j][i]-td[a][c][i][m]*soeri[m][b][j][k]+td[a][c][j][m]*soeri[m][b][i][k]+td[a][c][k][m]*soeri[m][b][j][i]-td[b][a][i][m]*soeri[m][c][j][k]+td[b][a][j][m]*soeri[m][c][i][k]+td[b][a][k][m]*soeri[m][c][j][i])/denom(i,j,k,a,b,c,soFock); }
/*Evaluating the energy*/       et += tttc*denom(i,j,k,a,b,c,soFock)*(tttc+tttd)/36;
                        }}}}}}
                return et;
}
//------------------------------------------------------------------------------
/*Now starts the part of the code that reads the atoms, the geometry and the 1-/2- electron integrals from the files.
The names of the files are hardcoded.*/

/*Function for reading the atoms and the geometry from the geometry file.
It also reads the number of orbitals in the end of the file (the last number). Be cautious.*/
void readGeom()
{
	 FILE *geomfile = fopen("geom.dat", "r");
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
All of them are hardcoded.
It also reads the number of orbitals in the end of the file (the last number). Be cautious.*/
void readData(double ****eri, gsl_matrix *T, gsl_matrix *S, gsl_matrix *V, double *enuc)
{
			FILE *Vfile = fopen("v.dat", "r"), *Tfile = fopen("t.dat", "r"), *enucfile = fopen("enuc.dat", "r"), *Sfile = fopen("s.dat", "r"), *erifile = fopen("eri.dat", "r");
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

/*Function for doing MP2 with spatial orbitals.
The equation can be found in any electronic structure/quantum chemistry textbook.*/
double mp2(gsl_vector *epsilon, double ****eri)
{
     double Emp2 = 0;
     for (int i=1;i<=nocc+1;i++){ for (int j=1;j<=nocc+1;j++) { for (int a=nocc+2; a<=norb; a++) { for (int b=nocc+2; b<=norb;b++){
                                        Emp2 += eri[i][a][j][b]*(2*eri[i][a][j][b]-eri[i][b][j][a])/(gsl_vector_get(epsilon,i-1) + gsl_vector_get(epsilon,j-1) - gsl_vector_get(epsilon, a-1) - gsl_vector_get(epsilon, b-1)); } } } }
     return Emp2;
}

/*Function for doing configuration-interaction singles in 2 ways. Keep in mind that the ground state doesn't change in CIS, it is only used for excitation energies.
1) With spin-orbitals. Larger matrix to diagonalize. You get all the values directly, 3 times the triplets and 1 the singlets.
2) With spatial -orbitals. Different case for singlets and different for triplets. You get each eigenvalue only once. Faster to execute because you have smaller matrices.
The spin-orbitals one is commented out.
Relative intensity part is not implemented yet.*/
void CIS(double ****soeri, gsl_matrix *soFock, gsl_vector *epsilon, double ****eri)
{
	int pos=0;

	gsl_vector *eval ;		gsl_matrix *evec ;		gsl_eigen_symmv_workspace *w ;

/*	//CIS with spin-orbitals
	gsl_matrix *Hcis = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); 	//Matrix H in the singly-excited determinant basis

	for (int i=0; i<Nelec; i++) {
		for (int a=Nelec; a<dim; a++) {
					for (int j=0; j<Nelec; j++) {
								for (int b=Nelec; b<dim; b++) {
				gsl_matrix_set(Hcis, pos/(Nelec*nocc), pos%(Nelec*nocc), gsl_matrix_get(soFock, a, b)*(i==j) - gsl_matrix_get(soFock, i, j)*(a==b) + soeri[a][j][i][b]);				++pos;
		}}}}

			 eval = gsl_vector_alloc (Nelec*nocc);
			 evec = gsl_matrix_alloc ( Nelec*nocc,  Nelec*nocc);

			 w = gsl_eigen_symmv_alloc ( Nelec*nocc);
				 gsl_eigen_symmv (Hcis, eval, evec, w);
			 gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

printf("Excitation energies from CIS.\n#\tHartree\n");
for (int i=0; i< Nelec*nocc; i++)
		printf("%d\t%.12lf\n", i, gsl_vector_get(eval, i));
*/

		//CIS with spatial-orbitals (eri) for singlets only
			gsl_matrix *Hciss = gsl_matrix_alloc( (nocc+1)*(norb-nocc-1), (nocc+1)*(norb-nocc-1) );

			pos=0;
			for (int i=1; i<=nocc+1; i++) {
				for (int a=nocc+2; a<=norb; a++) {
					for (int j=1; j<=nocc+1; j++) {
						 for (int b=nocc+2; b<=norb; b++) {
						gsl_matrix_set(Hciss, pos/((nocc+1)*(norb-nocc-1)), pos%((nocc+1)*(norb-nocc-1)), (gsl_vector_get(epsilon, a-1) - gsl_vector_get(epsilon, i-1))*(a==b)*(i==j) + 2*eri[i][a][j][b]- eri[i][j][a][b]);
										++pos;
				}}}}

					 eval = gsl_vector_alloc ( (nocc+1)*(norb-nocc-1) );

					 evec = gsl_matrix_alloc (  (nocc+1)*(norb-nocc-1),  (nocc+1)*(norb-nocc-1));

					 w = gsl_eigen_symmv_alloc ( (nocc+1)*(norb-nocc-1));
						 gsl_eigen_symmv (Hciss, eval, evec, w);
					 gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

		printf("Excitation energies for singlets only with CIS.\n#\tHartree\n");
		for (int i=0; i< (nocc+1)*(norb-nocc-1); i++)
				printf("%d\t%.12lf\n", i, gsl_vector_get(eval, i));

		/*		//Beginning intensity
								gsl_vector_view evec_0 = gsl_matrix_column (evec, 0);
								double dotu;

							for (int i =0; i<=4; i++) //Remember 4
							{
									gsl_vector_view evec_i = gsl_matrix_column (evec, i);
									gsl_blas_ddot(&evec_0.vector, &evec_i.vector, &dotu);
									printf("Relative intensity = %lf\n", dotu );
							}*/

				//CIS with spatial-orbitals (eri) for triplets only
					gsl_matrix *Hcist = gsl_matrix_alloc( (nocc+1)*(norb-nocc-1), (nocc+1)*(norb-nocc-1) );

					pos=0;
					for (int i=1; i<=nocc+1; i++) {
						for (int a=nocc+2; a<=norb; a++) {
							for (int j=1; j<=nocc+1; j++) {
								 for (int b=nocc+2; b<=norb; b++) {
								gsl_matrix_set(Hcist, pos/((nocc+1)*(norb-nocc-1)), pos%((nocc+1)*(norb-nocc-1)), (gsl_vector_get(epsilon, a-1) - gsl_vector_get(epsilon, i-1))*(a==b)*(i==j) - eri[i][j][a][b]);
												++pos;
						}}}}

							 eval = gsl_vector_alloc ( (nocc+1)*(norb-nocc-1) );

							 evec = gsl_matrix_alloc (  (nocc+1)*(norb-nocc-1),  (nocc+1)*(norb-nocc-1));

							 w = gsl_eigen_symmv_alloc ( (nocc+1)*(norb-nocc-1));
								 gsl_eigen_symmv (Hcist, eval, evec, w);
							 gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

				printf("Excitation energies for triplets only with CIS.\n#\tHartree\n");
				for (int i=0; i< (nocc+1)*(norb-nocc-1); i++)
						printf("%d\t%.12lf\n", i, gsl_vector_get(eval, i));

			  free(Hciss); free(Hcist);

}

/*Functions for doing Random Phase Approximation (RPA/TDHF) in 2 ways.
1) RPA1 diagonalizes the
		A   B
	 -A  -B
matrix.
2) RPA2 diagonalizes the
		(A+B)*(A-B)
matrix. It is way faster, but remember that it gives the excitation energies squared. */
void RPA1(double ****soeri, gsl_matrix *soFock)
{
	int pos = 0;
	gsl_matrix *A = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // Matrix A of RPA
	gsl_matrix *B = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // Matrix B of RPA

	for (int i=0; i<Nelec; i++) {
		for (int a=Nelec; a<dim; a++) {
					for (int j=0; j<Nelec; j++) {
								for (int b=Nelec; b<dim; b++) {
				gsl_matrix_set(A, pos/(Nelec*nocc), pos%(Nelec*nocc), gsl_matrix_get(soFock, a, b)*(i==j) - gsl_matrix_get(soFock, i, j)*(a==b) + soeri[a][j][i][b]);
				gsl_matrix_set(B, pos/(Nelec*nocc), pos%(Nelec*nocc), soeri[a][b][i][j]);				++pos;

		}}}}

	gsl_matrix *mRPA = gsl_matrix_alloc( 2*Nelec*nocc,  2*Nelec*nocc); // Total matrix of RPA
int posa = 0, posb=0, pos_a=0, pos_b=0;
		for (pos=0; pos<4*(Nelec*nocc)*(Nelec*nocc); ++pos) {
											if (pos < 2*(Nelec*nocc)*(Nelec*nocc) &&  pos%(2*Nelec*nocc) < Nelec*nocc)
												{ gsl_matrix_set(mRPA, pos/(2*Nelec*nocc), pos%(2*Nelec*nocc), gsl_matrix_get(A,  posa/(Nelec*nocc), posa%(Nelec*nocc)));	++posa;}
										 else if (pos < 2*(Nelec*nocc)*(Nelec*nocc) &&  pos%(2*Nelec*nocc) >= Nelec*nocc)
										 {gsl_matrix_set(mRPA, pos/(2*Nelec*nocc), pos%(2*Nelec*nocc), gsl_matrix_get(B,posb/(Nelec*nocc), posb%(Nelec*nocc)));	++posb;}
										 else if (pos >= 2*(Nelec*nocc)*(Nelec*nocc) &&  pos%(2*Nelec*nocc) < Nelec*nocc)
										 {gsl_matrix_set(mRPA, pos/(2*Nelec*nocc), pos%(2*Nelec*nocc), -gsl_matrix_get(B,pos_b/(Nelec*nocc), pos_b%(Nelec*nocc)));	++pos_b;}
										else
										{ gsl_matrix_set(mRPA, pos/(2*Nelec*nocc), pos%(2*Nelec*nocc), -gsl_matrix_get(A,pos_a/(Nelec*nocc), pos_a%(Nelec*nocc)));	++pos_a;}
							}

			gsl_vector_complex *eval = gsl_vector_complex_alloc (2*Nelec*nocc); //For non-symmetric eigenvalue problems
			gsl_matrix_complex *evec = gsl_matrix_complex_alloc (2*Nelec*nocc, 2*Nelec*nocc);

			gsl_eigen_nonsymmv_workspace *ww = gsl_eigen_nonsymmv_alloc (2*Nelec*nocc);
				 gsl_eigen_nonsymmv (mRPA, eval, evec, ww);
			 gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

printf("Excitation energies from RPA.\n#\tHartree\n");
for (int i=0; i< 2*Nelec*nocc; i++)
		printf("%d\t%.10lf\n", i, gsl_vector_complex_get(eval, i));

  free(A); free(B); free(mRPA);
}

void RPA2(double ****soeri, gsl_matrix *soFock)
{
	int pos = 0;
	gsl_matrix *A = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // Matrix A of RPA
	gsl_matrix *B = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // Matrix B of RPA

	for (int i=0; i<Nelec; i++) {
		for (int a=Nelec; a<dim; a++) {
					for (int j=0; j<Nelec; j++) {
								for (int b=Nelec; b<dim; b++) {
				gsl_matrix_set(A, pos/(Nelec*nocc), pos%(Nelec*nocc), gsl_matrix_get(soFock, a, b)*(i==j) - gsl_matrix_get(soFock, i, j)*(a==b) + soeri[a][j][i][b]);
				gsl_matrix_set(B, pos/(Nelec*nocc), pos%(Nelec*nocc), soeri[a][b][i][j]);				++pos;

		}}}}
		gsl_matrix *C = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // A+B
		gsl_matrix *D = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // A-B

		gsl_matrix_set_zero(C);
		gsl_matrix_set_zero(D);

		gsl_matrix_add(C, A);
		gsl_matrix_add(C, B);
		gsl_matrix_sub(D, B);
		gsl_matrix_add(D, A);

	gsl_matrix *mRPA = gsl_matrix_alloc( Nelec*nocc,  Nelec*nocc); // Total matrix of RPA, (A+B)*(A-B)
	gsl_matrix_set_identity(mRPA);

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, C, D, 0.0, mRPA);

			gsl_vector_complex *eval = gsl_vector_complex_alloc (Nelec*nocc);
			gsl_matrix_complex *evec = gsl_matrix_complex_alloc (Nelec*nocc, Nelec*nocc);

			gsl_eigen_nonsymmv_workspace *ww = gsl_eigen_nonsymmv_alloc (Nelec*nocc);
				 gsl_eigen_nonsymmv (mRPA, eval, evec, ww);
			 gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

printf("Excitation energies from RPA.\n#\tHartree\n");
for (int i=0; i< Nelec*nocc; i++) {
		double value = GSL_REAL(gsl_vector_complex_get(eval, i)); printf("%d\t%.10lf\n", i, sqrt(value)); } //GSL_REAL is used to get the square root of the eigenvalue. Square root because of this RPA formalism

		free(A); free(B); free(C); free(D); free(mRPA);

}


/*Function for performing the actual HF cycles.
DIIS is used automatically with 5 matrices.*/
double HF(gsl_matrix *S, gsl_matrix *T, gsl_matrix *V, double ****eri, gsl_matrix *Hcore, gsl_matrix *D, gsl_matrix *Fock, gsl_matrix *C, gsl_vector *epsilon, const gsl_matrix *Ssaved, int maxiter)
{
	gsl_vector *eval = gsl_vector_alloc (norb); 	gsl_matrix *evec = gsl_matrix_alloc (norb, norb);

	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (norb);
	gsl_eigen_symmv (S, eval, evec, w);
	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

	gsl_matrix *Lambda, *L;

	Lambda = gsl_matrix_alloc(norb, norb); L = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(Lambda); gsl_matrix_set_zero(L);

	for (int i = 0; i < norb; i++)
	{
			gsl_matrix_set(Lambda, i, i, gsl_vector_get (eval, i));
			for (int j=0; j<norb; j++)
			{
					gsl_matrix_set(L, i, j, gsl_matrix_get(evec,i,j));
		  }
 }

	gsl_matrix *Lambdasqrt = gsl_matrix_alloc(norb, norb);


	for (int i=0; i<norb; i++)
								gsl_matrix_set(Lambdasqrt, i, i, 1/sqrt(gsl_matrix_get(Lambda, i, i)));

	gsl_matrix *Ssqrt, *tmp;

	Ssqrt = gsl_matrix_alloc(norb, norb);
	tmp = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(Ssqrt);
	gsl_matrix_set_zero(tmp);

	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Lambdasqrt, L, 0.0, tmp);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, tmp, 0.0, Ssqrt);

	//clever_printf(Ssqrt, "S^-1/2 Matrix:");

	gsl_matrix *F;
	F = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(F);

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Hcore, Ssqrt, 0.0, tmp);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Ssqrt, tmp, 0.0, F);

	//clever_printf(F, "Initial F' matrix:");

	gsl_eigen_symmv (F, eval, evec, w);
	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

	gsl_matrix *Cprime = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(Cprime);
	gsl_vector_set_zero(epsilon);

	for (int i = 0; i < norb; i++)
	{
			gsl_vector_set(epsilon, i, gsl_vector_get (eval, i));
			for (int j=0; j<norb; j++)
			{
				gsl_matrix_set(Cprime, i, j, gsl_matrix_get(evec,i,j));
			}
	}

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Ssqrt, Cprime, 0.0, C);

	//clever_printf(C, "Initial C matrix:", norb);


	double sum;

	for (int i=0; i<norb; i++)
	{
				for (int j=0; j<norb ;j++)
				{
						sum = 0;
						for (int k=0; k<=nocc; k++)
						{
								sum += gsl_matrix_get(C,i,k)*gsl_matrix_get(C,j,k);
						}
						gsl_matrix_set(D,i,j, sum);
				}
	}

//	clever_printf(D, "Initial Density matrix", norb);
				printf("Starting HF. \n");
				printf("\nIter        E(elec)              E(tot)\n");

	double Eelec = 0.0;

	for (int i=0; i<norb; i++)
	{
			for (int j=0; j<norb; j++)
			{
					Eelec += gsl_matrix_get(D,i,j)*(gsl_matrix_get(Hcore,i,j) + gsl_matrix_get(F,i,j));
			}
	}

printf("\n00\t   %.12lf\n", Eelec);

	for (int i=0;i<norb;i++)
	{
		for (int j=0;j<norb;j++)
		{
				gsl_matrix_set(Fock,i,j,gsl_matrix_get(Hcore,i,j));
				for (int k=0; k<norb; k++)
				{
							sum = 0;
							for (int l=0; l<norb;l++)
							{
														sum += (2*eri[i+1][j+1][k+1][l+1] - eri[i+1][k+1][j+1][l+1])*gsl_matrix_get(D,k,l);
							}
						  gsl_matrix_set(Fock,i,j, sum+gsl_matrix_get(Fock,i,j));
				}
		}
	}

	//clever_printf(Fock, "Fock matrix: ", norb);
	Eelec = 0;

	for (int i=0; i<norb; i++)
	{
		for (int j=0; j<norb; j++)
		{
			Eelec += gsl_matrix_get(D,i,j)*(gsl_matrix_get(Hcore,i,j) + gsl_matrix_get(Fock,i,j));
		}
	}

	printf("01\t   %.12lf  \n", Eelec);
	/*error = FDS - SDF, where F is the Fock matrix, D is the Density matrix and S is the overlap.*/
				gsl_matrix *FDS = gsl_matrix_alloc(norb,norb);
				gsl_matrix *SDF = gsl_matrix_alloc(norb,norb);
				gsl_matrix *e1 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *e2 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *e3 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *e4 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *e5 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *F1 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *F2 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *F3 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *F4 = gsl_matrix_alloc(norb,norb);
				gsl_matrix *F5 = gsl_matrix_alloc(norb,norb);

				int iter=2;
				double res = 1.0;
				int diis = 1;
				gsl_matrix_set_zero(e1);
				gsl_matrix_set_zero(e2);
				gsl_matrix_set_zero(e3);
				gsl_matrix_set_zero(e4);
				gsl_matrix_set_zero(e5);

				while (res >= 1e-12 && iter <= maxiter)
				{

								gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Fock, Ssqrt, 0.0, tmp);
								gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Ssqrt, tmp, 0.0, F);

					w = gsl_eigen_symmv_alloc (norb);
						gsl_eigen_symmv (F, eval, evec, w);
					gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

							for (int i = 0; i < norb; i++)
								{
						gsl_vector_set(epsilon, i, gsl_vector_get (eval, i));
						for (int j=0; j<norb; j++)
						{
							gsl_matrix_set(Cprime, i, j, gsl_matrix_get(evec,i,j));
						}
					}

								gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Ssqrt, Cprime, 0.0, C);

								gsl_matrix *D_old = gsl_matrix_alloc(norb,norb);
								for (int i=0; i<norb; i++)
										for (int j=0; j<norb; j++)
												gsl_matrix_set(D_old, i, j, gsl_matrix_get(D, i, j));

					for (int i=0; i<norb; i++)
					{
						for (int j=0; j<norb ;j++)
						{
							sum = 0;
							for (int k=0; k<=nocc; k++)
							{
								sum += gsl_matrix_get(C,i,k)*gsl_matrix_get(C,j,k);
							}
							gsl_matrix_set(D,i,j, sum);
						}
					}
					Eelec = 0;

					for (int i=0;i<norb;i++)
					{
						for (int j=0;j<norb;j++)
						{
							gsl_matrix_set(Fock,i,j,gsl_matrix_get(Hcore,i,j));
							for (int k=0; k<norb; k++)
							{
																				sum = 0;
								for (int l=0; l<norb;l++)
								{
																								sum += (2*eri[i+1][j+1][k+1][l+1] - eri[i+1][k+1][j+1][l+1])*gsl_matrix_get(D,k,l);
								}
																				gsl_matrix_set(Fock,i,j, sum+gsl_matrix_get(Fock,i,j));
							}
						}
					}

					for (int i=0; i<norb; i++)
					{
						for (int j=0; j<norb; j++)
						{
							Eelec += gsl_matrix_get(D,i,j)*(gsl_matrix_get(Hcore,i,j) + gsl_matrix_get(Fock,i,j));
						}
					}
								if (res < 2)
								{
												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Fock, D, 0.0, tmp);
												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, Ssaved, 0.0, FDS);

												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Ssaved, D, 0.0, tmp);
												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, Fock, 0.0, SDF);

												if (diis % 5 == 1)
												{
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(e1, i, j, gsl_matrix_get(FDS,i,j));
																gsl_matrix_sub (e1,  SDF);
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(F1, i, j, gsl_matrix_get(Fock,i,j));
												}
												else if (diis % 5 == 2)
												{
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(e2, i, j, gsl_matrix_get(FDS,i,j));
																gsl_matrix_sub (e2,  SDF);
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(F2, i, j, gsl_matrix_get(Fock,i,j));
												}
												else if (diis % 5 == 3)
												{
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(e3, i, j, gsl_matrix_get(FDS,i,j));
																gsl_matrix_sub (e3,  SDF);
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(F3, i, j, gsl_matrix_get(Fock,i,j));
												}
												else if (diis % 5 == 4)
												{
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(e4, i, j, gsl_matrix_get(FDS,i,j));
																gsl_matrix_sub (e4,  SDF);
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(F4, i, j, gsl_matrix_get(Fock,i,j));
												}
												else if (diis % 5 == 0)
												{
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(e5, i, j, gsl_matrix_get(FDS,i,j));
																gsl_matrix_sub (e5,  SDF);
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(F5, i, j, gsl_matrix_get(Fock,i,j));
												}
												if (diis >= 5)
												{
																Fock = diisfunc(e1, e2, e3, e4, e5, F1, F2, F3, F4, F5);
												}
												++diis;
								}

								res = rms(D, D_old, norb);
								if (iter<10)
										printf("0%d\t   %.12lf  \t%.12lf\n", iter,Eelec, res);
								else
									  printf("%2d\t   %.12lf  \t%.12lf\n", iter,Eelec, res);
					iter++;
				}

				free(Lambda); free(L); free(Lambdasqrt); free(tmp); free(F); free(Cprime); free(FDS); free(SDF); free(e1); free(e2); free(e3); free(e4); free(e5); free(F1);  free(F2); free(F3); free(F4); free(F5); free(w); free(eval); free(evec);

				//findTraceDensity(Ssqrt, D); //Sanity check for the trace of density matrix

				if (iter <= maxiter+1)
					return Eelec;
				else { printf("HF did not converge.\n");
					return 0; }
}

/*Driver program.*/
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
