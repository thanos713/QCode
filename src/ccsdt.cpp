#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

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
