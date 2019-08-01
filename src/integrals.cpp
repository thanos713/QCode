#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

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
