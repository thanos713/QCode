#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

/*Function for performing the actual HF cycles.
DIIS is used automatically with 5 matrices.
The procedure described below is based on
Ch. 3 of the text by Szabo and Ostlund.*/
double HF(gsl_matrix *S, gsl_matrix *T, gsl_matrix *V, double ****eri, gsl_matrix *Hcore, gsl_matrix *D, gsl_matrix *Fock, gsl_matrix *C, gsl_vector *epsilon, const gsl_matrix *Ssaved, int maxiter)
{
	gsl_vector *eval = gsl_vector_alloc (norb); 	gsl_matrix *evec = gsl_matrix_alloc (norb, norb);
/*Note: eval will always contain eigenvalues and evec wll contain eigenvectors.

First step is to diagonalize the overlap (S) matrix: SL = L Lambda
The eigenvalues will be stored to the diagonal of Lambda and the eigenvectors to L.
*/
	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (norb);
	gsl_eigen_symmv (S, eval, evec, w); //Diagonalization
	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

	gsl_matrix *Lambda, *L;

	Lambda = gsl_matrix_alloc(norb, norb); L = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(Lambda); gsl_matrix_set_zero(L);

	for (int i = 0; i < norb; i++)
	{
			gsl_matrix_set(Lambda, i, i, gsl_vector_get (eval, i)); //Storing to Lambda
			for (int j=0; j<norb; j++)
			{
					gsl_matrix_set(L, i, j, gsl_matrix_get(evec,i,j)); //Storing to L
		  }
 }
/*Then the orthogonalization matrix is built: S^-1/2 = L Lambda^-1/2 L^T*/
	gsl_matrix *Lambdasqrt = gsl_matrix_alloc(norb, norb);


	for (int i=0; i<norb; i++)
								gsl_matrix_set(Lambdasqrt, i, i, 1/sqrt(gsl_matrix_get(Lambda, i, i))); //Generating Lambda^-1/2

	gsl_matrix *Ssqrt, *tmp; //Ssqrt = S^-1/2

	Ssqrt = gsl_matrix_alloc(norb, norb);
	tmp = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(Ssqrt);
	gsl_matrix_set_zero(tmp);

	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Lambdasqrt, L, 0.0, tmp); //Multiplying RHS
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, tmp, 0.0, Ssqrt);

	//clever_printf(Ssqrt, "S^-1/2 Matrix:");

/*The third step is to build a guess Fock matrix: F = (S^-1/2)^T Hcore S^-1/2
and then diagonalize it FC' = epsilon C', where epsilon contains the
ininitial orbital energies and and C' the initial MO coefficients.*/

	gsl_matrix *F; //Guess Fock matrix
	F = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(F);

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Hcore, Ssqrt, 0.0, tmp);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Ssqrt, tmp, 0.0, F); //Initial Fock matrix

	//clever_printf(F, "Initial F' matrix:");

	gsl_eigen_symmv (F, eval, evec, w);
	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

	gsl_matrix *Cprime = gsl_matrix_alloc(norb, norb);
	gsl_matrix_set_zero(Cprime);
	gsl_vector_set_zero(epsilon);

	for (int i = 0; i < norb; i++)
	{
			gsl_vector_set(epsilon, i, gsl_vector_get (eval, i)); //Epsilon
			for (int j=0; j<norb; j++)
			{
				gsl_matrix_set(Cprime, i, j, gsl_matrix_get(evec,i,j)); //C prime
			}
	}

/*And then we transform the MO coefficients to the AO basis by using the Ssqrt,
C = S^-1/2  C'  . This is needed to construct the density matrix by using:
Dmu nu = Sigma m_occupied Cm mu C m nu*/
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Ssqrt, Cprime, 0.0, C); //AO basis coefficients

	//clever_printf(C, "Initial C matrix:", norb);


	double sum;

	for (int i=0; i<norb; i++)
	{
				for (int j=0; j<norb ;j++)
				{
						sum = 0;
						for (int k=0; k<=nocc; k++) //Up to nocc because we care about the occupied MOs only in the summation
						{
								sum += gsl_matrix_get(C,i,k)*gsl_matrix_get(C,j,k); //Summation of the formula for the Density matrix
						}
						gsl_matrix_set(D,i,j, sum); //Assigning it to Dij
				}
	}

//	clever_printf(D, "Initial Density matrix", norb);
				printf("Starting HF. \n");
				printf("\nIter        E(elec)              E(tot)\n");

	double Eelec = 0.0;
/*Then we compute the initial energy with Sigma_mu nu Dmu nu(Hcore mu nu + Fmu nu)
It is stored in Eelec.
Note: The total energy is the previous one plus the Enuc.*/
	for (int i=0; i<norb; i++)
	{
			for (int j=0; j<norb; j++)
			{
					Eelec += gsl_matrix_get(D,i,j)*(gsl_matrix_get(Hcore,i,j) + gsl_matrix_get(F,i,j)); //SCF energy
			}
	}

printf("\n00\t   %.12lf\n", Eelec); //Zeroth iteration
/*Then it is time to compute the new Fock matrix.
Fmunu = Hcore munu + Sigma_lambda sigma_AO Dlambda sigma [2(mu nu | lambda sigma) - (mu lambda| nu sigma)]
The new Fock matrix is store in Fock, not in F.*/
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
														sum += (2*eri[i+1][j+1][k+1][l+1] - eri[i+1][k+1][j+1][l+1])*gsl_matrix_get(D,k,l); //Sum
							}
						  gsl_matrix_set(Fock,i,j, sum+gsl_matrix_get(Fock,i,j)); //Assignment to Fock
				}
		}
	}

	//clever_printf(Fock, "Fock matrix: ", norb);
	Eelec = 0;

	for (int i=0; i<norb; i++)
	{
		for (int j=0; j<norb; j++)
		{
			Eelec += gsl_matrix_get(D,i,j)*(gsl_matrix_get(Hcore,i,j) + gsl_matrix_get(Fock,i,j)); //New energy
		}
	}

	printf("01\t   %.12lf  \n", Eelec);
	/*error = FDS - SDF, where F is the Fock matrix, D is the Density matrix and S is the overlap.
	Some of the code below is needed for DIIS.*/
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

				int iter=2; //Current iteration
				double res = 1.0; //A value for the residual (doesn't matter as we want to achieve something lower).
				int diis = 1; /*The variable diis says that we will add the first DIIS matrix when we reach the DIIS if statement below.
												6 means again put in the first place, but 7 means put in the second place. So it is basically mod 5,
												because we store 5 matrices.*/
				gsl_matrix_set_zero(e1);
				gsl_matrix_set_zero(e2);
				gsl_matrix_set_zero(e3);
				gsl_matrix_set_zero(e4);
				gsl_matrix_set_zero(e5);

				while (res >= 1e-12 && iter <= maxiter) //Beginning the cycle
				{

								gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Fock, Ssqrt, 0.0, tmp);
								gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Ssqrt, tmp, 0.0, F); //orthogonalization

					w = gsl_eigen_symmv_alloc (norb);
						gsl_eigen_symmv (F, eval, evec, w);
					gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

							for (int i = 0; i < norb; i++)
								{
						gsl_vector_set(epsilon, i, gsl_vector_get (eval, i)); //New orbital energies
						for (int j=0; j<norb; j++)
						{
							gsl_matrix_set(Cprime, i, j, gsl_matrix_get(evec,i,j)); //New MO coefficients
						}
					}

								gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Ssqrt, Cprime, 0.0, C);


								gsl_matrix *D_old = gsl_matrix_alloc(norb,norb);
								for (int i=0; i<norb; i++)
										for (int j=0; j<norb; j++)
												gsl_matrix_set(D_old, i, j, gsl_matrix_get(D, i, j)); //Keeping old Density matrix. Needed to check for convergence.

					for (int i=0; i<norb; i++)
					{
						for (int j=0; j<norb ;j++)
						{
							sum = 0;
							for (int k=0; k<=nocc; k++)
							{
								sum += gsl_matrix_get(C,i,k)*gsl_matrix_get(C,j,k);
							}
							gsl_matrix_set(D,i,j, sum); //Building new Density matrix from MO coefficients
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
																				gsl_matrix_set(Fock,i,j, sum+gsl_matrix_get(Fock,i,j)); //New Fock matrix
							}
						}
					}

					for (int i=0; i<norb; i++)
					{
						for (int j=0; j<norb; j++)
						{
							Eelec += gsl_matrix_get(D,i,j)*(gsl_matrix_get(Hcore,i,j) + gsl_matrix_get(Fock,i,j)); //New energy
						}
					}
								if (res < 2) //Checking if we have sufficiently reduced the gradient. We do not want to start DIIS too early.
								{
												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Fock, D, 0.0, tmp);
												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, Ssaved, 0.0, FDS); //Computing FDS

												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Ssaved, D, 0.0, tmp);
												gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, Fock, 0.0, SDF); //Computing SDF

												if (diis % 5 == 1) //As mentioned before, 1 goes stores at the first place, 6 again to the first, etc.
												{
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(e1, i, j, gsl_matrix_get(FDS,i,j));
																gsl_matrix_sub (e1,  SDF);//error matrix, SDF - FDS
																for (int i=0; i<norb; i++)
																				for (int j=0;j<norb;j++)
																								gsl_matrix_set(F1, i, j, gsl_matrix_get(Fock,i,j)); //Store the current Fock matrix for DIIS, too.
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
												if (diis >= 5)//If we have at least stored 5 DIIS matrices, start extrapolating.
												{
																Fock = diisfunc(e1, e2, e3, e4, e5, F1, F2, F3, F4, F5);
												}
												++diis;
								}

								res = rms(D, D_old, norb); //Checking for convergence using the density matrix.
								if (iter<10)
										printf("0%d\t   %.12lf  \t%.12lf\n", iter,Eelec, res);
								else
									  printf("%2d\t   %.12lf  \t%.12lf\n", iter,Eelec, res);
					iter++;
				}

				free(Lambda); free(L); free(Lambdasqrt); free(tmp); free(F); free(Cprime); free(FDS); free(SDF); free(e1); free(e2); free(e3); free(e4); free(e5); free(F1);  free(F2); free(F3); free(F4); free(F5); free(w); free(eval); free(evec);

				//findTraceDensity(Ssqrt, D); //Sanity check for the trace of density matrix

				if (iter <= maxiter+1) //If we do not surpass the maximum iterations.
					return Eelec;
				else { printf("HF did not converge.\n");
					return 0; }
}
