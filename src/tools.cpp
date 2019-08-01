#include <stdlib.h>
#include <gsl/gsl_blas.h> //Used for muliplying GSL matrices.
#include <math.h>
#include "../headers/common.h"
#include "../headers/tools.h"
#include "../headers/diis.h"

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
