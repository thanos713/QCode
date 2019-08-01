#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

/*Function for doing configuration-interaction singles in 2 ways. Keep in mind that the ground state doesn't change in CIS, it is only used for excitation energies.
1) With spin-orbitals. Larger matrix to diagonalize. You get all the values directly, 3 times the triplets and 1 the singlets.
2) With spatial -orbitals. Different case for singlets and different for triplets. You get each eigenvalue only once. Faster to execute because you have smaller matrices.
The spin-orbitals one is commented out.
Relative intensity part is not implemented yet.
Look inside the code for the equations for Hamiltonian Matrix elements.*/
void CIS(double ****soeri, gsl_matrix *soFock, gsl_vector *epsilon, double ****eri)
{
	int pos=0;

	gsl_vector *eval ;		gsl_matrix *evec ;		gsl_eigen_symmv_workspace *w ;



/*	//CIS with spin-orbitals
	//Hia,jb = f_ab delta_ij - f_ij delta_ab + <aj||ib>
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
			/*Hia,jb = f_ab delta_ij - f_ij delta_ab + 2(ia|jb) - (ij|ab)*/
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
				/*Hia,jb = f_ab delta_ij - f_ij delta_ab - (ij|ab)*/
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

/*Functions for doing Random Phase Approximation (RPA/TDHF) in 2 ways. Using spin-orbitals.
1) RPA1 diagonalizes the
		A   B
	 -A  -B
matrix.
2) RPA2 diagonalizes the
		(A+B)*(A-B)
matrix. It is way faster, but remember that it gives the excitation energies squared.
Aia,jb = f_ab delta_ij - f_ij delta_ab + <aj||ib>
Bia,jb = <ab||ij>*/
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

			gsl_vector_complex *eval = gsl_vector_complex_alloc (2*Nelec*nocc); //Complex, because of non-symmetric eigenvalue problem
			gsl_matrix_complex *evec = gsl_matrix_complex_alloc (2*Nelec*nocc, 2*Nelec*nocc);

			gsl_eigen_nonsymmv_workspace *ww = gsl_eigen_nonsymmv_alloc (2*Nelec*nocc); //Non-symmetric diagonalization
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
