
/* Francis algorithm in C; see README */

#include "util.h"
#include <stdio.h>
#include <math.h>
//#define UNCLEAN_MATRICES 1



/** set matrix to zero, strict lower triangular */
void set_zero_slt(size_t nrows, size_t ncols, double *M, size_t stride) {
	size_t r, c;

	for (r = 1; r < nrows; r++)
		for (c = 0; c < r; c++)
			M[c + stride*r] = 0.0;
}



/** Display a matrix as follows:
 * M=[   +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *            ...           ...                   ...
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn];
 */
void showmat(size_t nrows, size_t ncols, double *M) {
	int r, c;
#ifdef UNCLEAN_MATRICES
	int insignificant = 0;
#endif

	printf("M=[");
	for (r = 0; r < nrows; r++) {
		for (c = 0; c < ncols; c++) {
#ifdef UNCLEAN_MATRICES
			insignificant = (fabs(M[c + r*ncols]) < 0.001);
			if (insignificant) {
				printf("\t          ");
			} else {
#endif
				printf("\t%+3.6f", M[c + r*ncols]);
#ifdef UNCLEAN_MATRICES
			}
#endif

			/* formatting and frill */
			if (c == ncols - 1) {
				if (r == nrows - 1) {
					printf("];\n");
				} else {
					printf(";\n");
				}
			} else {
#ifdef UNCLEAN_MATRICES
				if (!insignificant) {
#endif
					putchar(',');
#ifdef UNCLEAN_MATRICES
				}
#endif
			}
		}
	}
}



/** Display a matrix and a description as follows:
 *
 * <description>:
 *
 * M=[   +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *            ...           ...                   ...
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn];
 */
void smd(const char *description, size_t nrows, size_t ncols, double *M) {
	printf("\n%s:\n\n", description);
	showmat(nrows, ncols, M);
}



/** Display a square matrix as follows:
 * M=[   +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *            ...           ...                   ...
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn];
 */
void ssm(size_t order, double *M) {
	showmat(order, order, M);
}



/** Display a square matrix and a description as follows:
 *
 * <description>:
 *
 * M=[   +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn,
 *            ...           ...                   ...
 *       +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn];
 */
void ssmd(const char *description, size_t order, double *M) {
	smd(description, order, order, M);
}



/** Display a row vector as follows:
 * M=[   +nnn.nnnnnn,   +nnn.nnnnnn,   ...,   +nnn.nnnnnn];
 */
void srv(size_t length, double *v) {
	showmat(1, length, v);
}



/** Display a column vector and a description as follows:
 *
 * <description>:
 *
 * M=[   +nnn.nnnnnn,
 *       +nnn.nnnnnn,
 *            ...
 *       +nnn.nnnnnn];
 */
void scvd(const char *description, size_t length, double *v) {
	smd(description, length, 1, v);
}

