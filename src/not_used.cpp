#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS

#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

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

//------------------------------------------------------------------------------

/*Function for doing MP2 with spatial orbitals.
Emp2 = Sigma_ij Sigma_ab (ia|jb)[2(ia|jb)-(ib|ja)]/(epsilon_i + epsilon_j - epsilon_a - epsilon_b)
Note again: i,j occupied, a,b virtual*/
double mp2(gsl_vector *epsilon, double ****eri)
{
     double Emp2 = 0;
     for (int i=1;i<=nocc+1;i++){ for (int j=1;j<=nocc+1;j++) { for (int a=nocc+2; a<=norb; a++) { for (int b=nocc+2; b<=norb;b++){
                                        Emp2 += eri[i][a][j][b]*(2*eri[i][a][j][b]-eri[i][b][j][a])/(gsl_vector_get(epsilon,i-1) + gsl_vector_get(epsilon,j-1) - gsl_vector_get(epsilon, a-1) - gsl_vector_get(epsilon, b-1)); } } } }
     return Emp2;
}
