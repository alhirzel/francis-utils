
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

/* NOTE: BLAS routine conventions: target is last, size comes before operand,
 * incX comes after */



const size_t Anorm_N = 7, Ahess_N = 7;

/** A is a random test matrix with eigenvalues 2,3,5,7,11,13,17 */
double Anorm[] = {
1.2205,-0.9778,-3.5875,6.3239,-18.6134,-11.5920,-27.5207,
15.4760,14.0774,9.7466,-9.1991,32.6386,10.2165,44.3215,
-5.3925,-6.9141,-0.3162,-5.2980,-8.8870,-4.6269,-7.1133,
-3.2413,1.2535,-0.0849,11.0866,-6.0826,-1.4200,-12.6913,
-6.0873,-0.6122,-5.6896,9.1950,-7.6241,-4.1090,-21.6177,
-7.5542,-8.1024,-4.4474,-2.2788,-5.1283,6.1540,-2.6075,
12.7377,9.4829,10.2203,-1.8157,21.6712,10.9930,33.4019,
};



/** Ahess is the hessenberg form of A */
double Ahess[] = {
1.2205,7.1720,-4.6229,-10.5124,21.2274,-11.3153,-23.0235,
-23.1400,27.3588,-5.6390,-17.8708,38.2222,-38.6213,-42.2665,
0,5.4755,-2.2132,-7.4969,5.9252,0.8973,-20.5604,
0,0,1.3863,5.5655,-2.7729,1.4370,-2.3967,
0,0,0,-1.7075,13.9845,-4.1728,-4.6832,
0,0,0,0,2.1182,4.8584,-4.7575,
0,0,0,0,0,-3.3022,7.2255,
};

double Simp[] = {
	2,2,2,2,
	1,2,2,2,
	0,1,2,2,
	0,0,1,2,
};



void test_two_way(void) {
	double *M = Ahess;
	size_t N = Ahess_N;

	struct bulge_info forward, backward;

	double forward_shifts[] = {3};
	double backward_shifts[] = {1.0/3.0};

	ssmd("original matrix", N, M);

	form_bulge(&forward, N, M, 1, forward_shifts, CHASE_FORWARD);
	ssmd("with forward shift", N, M);

	form_bulge(&backward, N, M, 1, backward_shifts, CHASE_BACKWARD);
	ssmd("with backward shift as well", N, M);

	chase_bulge_step(&forward);
	ssmd("chased forward", N, M);

	chase_bulge_step(&backward);
	ssmd("chased backward", N, M);

	chase_bulge_step(&forward);
	ssmd("chased forward", N, M);

	chase_bulge_step(&backward);
	ssmd("chased backward (nonsensical)", N, M);
}

void test_qr_algorithm(void) {
	double *M = Ahess;
	size_t N = 7;
	double error, tol = 0.000001;

	double shifts[2];

	do {

		/* use last two sub-diagonal entries as shifts */
		shifts[0] = M[N-2 + (N-1)*N];
		shifts[1] = M[N-3 + (N-2)*N];

		/* use last two diagonal entries as shifts */
		//shifts[0] = M[N-1 + (N-1)*N];
		//shifts[1] = M[N-2 + (N-2)*N];

		build_and_chase_bulge(N, M, 2, shifts, CHASE_FORWARD);

		error = cblas_dnrm2(N-1, &M[N], N+1);

		printf("iteration complete with shifts %f %f, error = %6.18f\n", shifts[0], shifts[1], error);
		ssm(N, M);
		printf("press <enter> to do another iteration, or q<enter> to quit\n");
		if (getchar() == 'q') return;

	} while (error > tol);

	ssmd("output of QR algorithm", N, M);
}

void test_development(void) {
	double shifts[] = {0, 0, 0, 1,1,2,24,24,2,42,4};
	double *M = Ahess;
	size_t N = 7;
	int i;

	struct bulge_info b;

	ssmd("%original matrix", N, M);
#if 0
	puts("\n\n\n\n\n\nRUNNING FORWARD...");
	i = form_bulge(&b, N, M, 2, shifts, CHASE_FORWARD);
#else
	puts("\n\n\n\n\n\nRUNNING BACKWARD...");
	i = form_bulge(&b, N, M, 2, shifts, CHASE_BACKWARD);
#endif
	ssmd("%new shiny bulge", b.order, b.M);
	printf("%%need to chase it for %u steps; start the chase!\n\n", i);
	do {
		i = chase_bulge_step(&b);
		printf("%%have %u steps to go, just completed number %lu, here's the result:\n\n", i, b.steps_chased);
		ssm(b.order, b.M);
		printf("eig(M)\n");
	} while (i > 0);
}



void test_bulge_inflate(int argc, char *argv[]) {
	size_t N = 13;
	double A[] = {
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

	struct bulge_info forward, backward;

	//double forward_shifts[] = {1.99, 5.0};
	//double backward_shifts[] = {2.01, 3.0};
	double forward_shifts[] = {2.0, 3.0};
	double backward_shifts[] = {1.0/2.0, 1.0/3.0};

	if (argc - 1 == 1) {
		double t;
		sscanf(argv[1], "%lf", &t);
		forward_shifts[0] += t;
		forward_shifts[1] += t;
		backward_shifts[0] += t;
		backward_shifts[1] += t;
	} else if (argc - 1 == 4) {
		printf("Using supplied shifts...");
		sscanf(argv[1], "%lf", &forward_shifts[0]);
		sscanf(argv[2], "%lf", &forward_shifts[1]);
		sscanf(argv[3], "%lf", &backward_shifts[0]);
		sscanf(argv[4], "%lf", &backward_shifts[1]);
	}

	//ssmd("original matrix", N, A);

	form_bulge(&forward, N, A, 2, forward_shifts, CHASE_FORWARD);
	//ssmd("with forward shift", N, A);

	form_bulge(&backward, N, A, 2, backward_shifts, CHASE_BACKWARD);
	//ssmd("with backward shift as well", N, A);

	int i;
	size_t forward_steps = 3;
	size_t backward_steps = 3;

	for (i = 0; i < forward_steps; i++) chase_bulge_step(&forward);
	for (i = 0; i < backward_steps; i++) chase_bulge_step(&backward);
	//ssmd("after some chasing and before Schur", N, A);


	/* Schur decomp courtesy of GSL */
	/* http://www.gnu.org/software/gsl/manual/html_node/Real-Nonsymmetric-Matrices.html#Real-Nonsymmetric-Matrices */
	/*************************************************************************/

	/* Big bulge position (first element along diagonal that is within bulge,
	 * top left is 0). */
	size_t bbp = forward_steps;

	/* Big bulge size aka num of rows/cols in bulge (i.e. matrix size minus
	 * number of steps taken). */
	size_t bbs = N - forward_steps - backward_steps;

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

	ssmd("before any work done", N, A);

	/* The Schur decomp populates Q with rotation matrix */
	gsl_eigen_nonsymm_workspace *w = gsl_eigen_nonsymm_alloc(bbs);
	gsl_vector_complex *eval = gsl_vector_complex_alloc(bbs);
	gsl_eigen_nonsymm_params(1, 0, w); // compute full Schur decomp
	gsl_eigen_nonsymm_Z(&Am, eval, &Qm, w);
	set_zero_lt(2, bbs, bbs, &A[bbp + bbp*N], N);
	gsl_eigen_nonsymm_free(w);
	gsl_vector_complex_free(eval);

	ssmd("A before spikes", N, A);

	/* create the spikes */
	double *V;

	/* vertical spike (a col) */
	V = &A[bbp-1 + (bbp)*N];
	cblas_dgemv(CblasRowMajor, CblasTrans, bbs, bbs, 1.0, Q, bbs, V, N, 0.0, temp, 1);
	cblas_dcopy(bbs, temp, 1, V, N);

	/* horizontal spike (a row) */
	V = &A[bbp + (bbp+bbs)*N];
	cblas_dgemv(CblasRowMajor, CblasTrans, bbs, bbs, 1.0, Q, bbs, V, 1, 0.0, temp, 1);
	cblas_dcopy(bbs, temp, 1, V, 1);

	ssmd("A after spikes, before propagation", N, A);

	/* propagate the Schur transform throughout the entire matrix (preserve
	 * eigenvalues) */

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

	ssmd("A after spikes and propagation", N, A);

	free(temp);
}



int main(int argc, char *argv[]) {
	//test_two_way();
	//test_qr_algorithm();
	//sscanf(argv[1]
	test_bulge_inflate(argc, argv);

	return EXIT_SUCCESS;
}

