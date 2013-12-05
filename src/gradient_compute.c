
/* Francis algorithm in C; see README */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include "bulge.h"
#include "util.h"

#define N (13)
const double Aorig[] = {
	2.81014,-6.48911,-15.18678,21.77452,5.07491,-33.61697,7.69500,-3.21216,41.49093,16.75451,37.52472,-4.91861,30.00529,
	-79.66993,32.07671,-37.41973,27.72298,3.11131,-59.35638,-17.86218,18.53775,72.09095,3.57814,158.46592,284.03035,333.00127,
	0,17.26496,33.19817,-38.30290,-25.19686,31.03556,-58.71160,35.69149,-70.85779,-69.20413,73.02552,-81.81205,-153.14354,
	0,0,-4.70294,19.82071,17.59862,-9.54619,22.05326,-23.91293,23.02401,50.87808,-87.43072,41.81270,130.30457,
	0,0,0,6.46713,17.41596,-7.30266,-1.34361,0.41457,3.50943,-5.91071,34.33494,15.76940,8.73907,
	0,0,0,0,15.48683,20.92098,26.03202,-17.96809,7.73529,46.79353,-112.27704,-9.34287,71.89896,
	0,0,0,0,0,11.00435,26.79193,4.71123,-4.04375,-14.29687,16.39901,18.17265,-39.43072,
	0,0,0,0,0,0,5.19289,19.11160,3.20148,16.21861,-28.96126,-0.84516,29.94569,
	0,0,0,0,0,0,0,7.72795,10.10828,9.25127,-2.09658,-7.79144,3.98539,
	0,0,0,0,0,0,0,0,3.19594,12.23062,-9.39841,4.20665,6.76808,
	0,0,0,0,0,0,0,0,0,-6.72820,22.71813,-19.60648,-28.18783,
	0,0,0,0,0,0,0,0,0,0,-1.11828,25.13190,9.08088,
	0,0,0,0,0,0,0,0,0,0,0,-20.40095,-4.33512,};



void print_row(FILE *OUTFILE, size_t nshifts_both, double *fshifts, double *bshifts) {
	struct bulge_info forward, backward;
	static double A[N*N];

	int i;
	int fsteps = floor((N - 2*(nshifts_both+1) - 1) / 2.0);
	int bsteps = ceil( (N - 2*(nshifts_both+1) - 1) / 2.0);

	fprintf(OUTFILE, "%4.24f", fshifts[0]);
	for (i = 1; i < nshifts_both; i++) fprintf(OUTFILE, ",%4.24f", fshifts[i]);
	for (i = 0; i < nshifts_both; i++) fprintf(OUTFILE, ",%4.24f", bshifts[i]);

	/* Populate A, chase bulges to center */
	cblas_dcopy(N*N, Aorig, 1, A, 1);
	//ssmd("original matrix", N, A);
	form_bulge(&forward, N, A, 2, fshifts, CHASE_FORWARD);
	//ssmd("with forward shift", N, A);
	form_bulge(&backward, N, A, 2, bshifts, CHASE_BACKWARD);
	//ssmd("with backward shift as well", N, A);
	for (i = 0; i < fsteps; i++) chase_bulge_step(&forward);
	for (i = 0; i < bsteps;  i++) chase_bulge_step(&backward);
	//ssmd("after some chasing and before Schur", N, A);


	/* Schur decomp courtesy of GSL */
	/* http://www.gnu.org/software/gsl/manual/html_node/Real-Nonsymmetric-Matrices.html#Real-Nonsymmetric-Matrices */
	/*************************************************************************/

	/* Big bulge position (first element along diagonal that is within bulge,
	 * top left is 0). */
	size_t bbp = fsteps;

	/* Big bulge size aka num of rows/cols in bulge (i.e. matrix size minus
	 * number of steps taken). */
	size_t bbs = N - fsteps - bsteps;

	/* Rotation matrix from the Schur decomposition */
	double *Q = calloc(bbs*bbs, sizeof(double));

	/* Temporary vector for gemv */
	double *temp = calloc(bbs, sizeof(double));

	/* Wrap matrices in GSL objects */
	gsl_matrix Am, Qm;
	Am.size1 = Am.size2 = Qm.size1 = Qm.size2 = bbs;
	Am.tda = N;
	Qm.tda = bbs;
	Am.data = &A[bbp + bbp*N]; Qm.data = Q;

	/* The Schur decomp populates Q with rotation matrix */
	gsl_eigen_nonsymm_workspace *w = gsl_eigen_nonsymm_alloc(bbs);
	gsl_vector_complex *eval = gsl_vector_complex_alloc(bbs);
	gsl_eigen_nonsymm_params(1, 0, w); // compute full Schur decomp
	gsl_eigen_nonsymm_Z(&Am, eval, &Qm, w);
	set_zero_lt(2, bbs, bbs, &A[bbp + bbp*N], N);
	gsl_eigen_nonsymm_free(w);
	gsl_vector_complex_free(eval);
	//ssmd("A before spikes", N, A);

	/* create the spikes */
	double *V;

	/* vertical spike (a col) */
	V = &A[bbp-1 + (bbp)*N];
	cblas_dgemv(CblasRowMajor, CblasTrans, bbs, bbs, 1.0, Q, bbs, V, N, 0.0, temp, 1);
	//cblas_dcopy(bbs, temp, 1, V, N);
	for (i = 0; i < bbs; i++) fprintf(OUTFILE, ",%4.24f", temp[i]);

	/* horizontal spike (a row) */
	V = &A[bbp + (bbp+bbs)*N];
	cblas_dgemv(CblasRowMajor, CblasTrans, bbs, bbs, 1.0, Q, bbs, V, 1, 0.0, temp, 1);
	//cblas_dcopy(bbs, temp, 1, V, 1);
	for (i = 0; i < bbs; i++) fprintf(OUTFILE, ",%4.24f", temp[i]);

	//ssmd("A after spikes, before propagation", N, A);

	/* propagate the Schur transform throughout the entire matrix (preserve
	 * eigenvalues) */

/* don't need to propagate to get the values we care about */
#if 0
	/* cols */
	for (i = bbs+bbp; i < N; i++) {
		V = &A[i + bbp*N];
		cblas_dgemv(CblasRowMajor, CblasTrans, bbs, bbs, 1.0, Q, bbs, V, N, 0, temp, 1);
		cblas_dcopy(bbs, temp, 1, V, N);
	}

	/* rows */
	for (i = 0; i < bbp; i++) {
		V = &A[bbp + i*N];
		cblas_dgemv(CblasRowMajor, CblasTrans, bbs, bbs, 1.0, Q, bbs, V, 1, 0, temp, 1);
		cblas_dcopy(bbs, temp, 1, V, 1);
	}
#endif

	free(temp);
	fprintf(OUTFILE, "\n");
}



void create_csv(const char *filename, double s1, double s2, double s3, double s4) {
	double fshifts[2], bshifts[2];

	const double step = 0.1;
	const double radius = step*10;

	FILE *f = fopen(filename, "w");

	for (fshifts[0] = s1 - radius; fshifts[0] < s1 + radius; fshifts[0] += step) {
		for (fshifts[1] = s2 - radius; fshifts[1] < s2 + radius; fshifts[1] += step) {
			for (bshifts[0] = s3 - radius; bshifts[0] < s3 + radius; bshifts[0] += step) {
				for (bshifts[1] = s4 - radius; bshifts[1] < s4 + radius; bshifts[1] += step) {
					print_row(f, 2, fshifts, bshifts);
				}
			}
printf("%s completed: %f%%\n", filename, (fshifts[0] - (s1 - radius)) / 2.0 / radius * 100.0);
		}
	}

	fclose(f);
}



int main(int argc, char *argv[]) {

	/* create files with the following shifts:
	 *
	 *   - shifts near 2, 3,   3,   2
	 *   - shifts near 2, 3, 1/3, 1/2
	 */

	create_csv("out2332.csv", 2.0, 3.0, 3.0, 2.0);
	create_csv("out23r3r2.csv", 2.0, 3.0, 1.0/3.0, 1.0/2.0);

	return EXIT_SUCCESS;
}

