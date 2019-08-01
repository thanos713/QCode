#include <gsl/gsl_eigen.h> //Used for diagonalizations.
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <gsl/gsl_linalg.h> //Used for solving linear systems of equations, ie. in DIIS
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

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
