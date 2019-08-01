void allocate4(double ****&tensor, int dimension);
void allocate2(double **&matrix, int dimension);
void set_zero(double **matrix, int dimension);
void set_zero(double ****tensor, int dimension);
void clever_printf(gsl_matrix *matrix, const char *text, int size);
double rms(gsl_matrix *new_, gsl_matrix *old_, int dimension);
double rms(double ****new_, double ****old_, int dimension);
