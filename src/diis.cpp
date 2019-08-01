#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

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
