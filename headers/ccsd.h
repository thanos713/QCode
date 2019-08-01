double taut(int a, int b,int i,int j, double **ts, double ****td);
double tau(int a, int b,int i,int j, double **ts, double ****td);
double ccsdCalc(gsl_matrix *soFock, double ****soeri, double **ts, double ****td);
void updateIntermediates(gsl_matrix *soFock, double ****soeri, double **Fae, double **Fmi, double **Fme, double ****Wmnij, double ****Wabef, double ****Wmbej, double **ts, double ****td);
void makeT1( double **ts, double ****td, double **tsnew, double ****soeri, gsl_matrix *soFock, double **Fae, double **Fmi, double **Fme, double **Dai);
void makeT2( double **ts, double ****td, double ****tdnew, double ****soeri, gsl_matrix *soFock, double **Fae, double **Fmi, double **Fme, double ****Wmnij, double ****Wabef, double ****Wmbej, double ****Dabij);
double doccsd(gsl_matrix *soFock, double ****soeri, int maxiter, double **ts, double ****td);
