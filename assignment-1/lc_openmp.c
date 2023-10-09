#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "utils.h"

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
} 

/**
 * @brief Output the seconds elapsed while execution.
 *
 * @param seconds Seconds spent on execution, excluding IO.
 */
static void print_time(double const seconds)
{
  printf("Execution time: %0.04fs\n", seconds);
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: %s <points> <labels> <iters> <threads>\n", argv[0]);
        return -1;
    }

    int iterations = atoi(argv[3]);
    if (iterations == 0) {
        perror("atoi");
        return -1;
    }

    int threads = atoi(argv[4]);
    if (threads == 0) {
        perror("atoi");
        return -1;
    }

    int result;

    int num_points; // n
    int dim_points; // m
    double* points; // X
    double* labels; // y

    result = load_data(argv[1], argv[2],
                           &num_points, &dim_points,
                           &points, &labels);
    if (result) {
        return -1;
    }

    double* denom_vals; // Denominator values
    double* XT; // X^T
    double* Xw; // X*w
    double* w;

    result = allocate_space(num_points, dim_points,
                            &denom_vals, &XT, &Xw, &w);
    if (result) {
        free(points);
        free(labels);
        return -1;
    }

    double start_time = monotonic_seconds();

    // Preprocessing step: Compute X^T and denominator values
    // Initialize w to all 0s
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < dim_points; i++) {
        double sum = 0;
        double* ptr = points + i;
        double* row = XT + i*num_points;
        for (int j = 0; j < num_points; j++) {
            row[j] = *ptr;
            sum += (*ptr) * (*ptr);
            ptr += dim_points;
        }
        denom_vals[i] = sum;
        w[i] = 0;
    }

    // Initialize Xw to all 0s
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < num_points; i++) {
        Xw[i] = 0;
    }

    // Apply the algorithm
    for (int iter = 0; iter < iterations; iter++) {

        // Compute the new w vector
        for (int i = 0; i < dim_points; i++) {

            // Compute new w_i value if denominator is not 0
            // Otherwise, leave w_i as 0
            if (denom_vals[i] == 0) {
                continue;
            }

            double* row = XT + i*num_points;

            // Compute new w_i value
            double numerator = 0;
            #pragma omp parallel for num_threads(threads) reduction(+:numerator)
            for (int j = 0; j < num_points; j++) {
                double Xw_minusi = Xw[j] - row[j]*w[i]; // The jth number in X_{-i}*w_{-i}
                numerator += row[j] * (labels[j] - Xw_minusi);
            }

            double wi_new = numerator / denom_vals[i];

            // Update X*w vector
            #pragma omp parallel for num_threads(threads)
            for (int j = 0; j < num_points; j++) {
                Xw[j] += row[j] * (wi_new-w[i]);
            }

            w[i] = wi_new;
        }

        // Compute and print out the loss
        double loss = 0;
        #pragma omp parallel for num_threads(threads) reduction(+:loss)
        for (int i = 0; i < num_points; i++) {
            double diff = Xw[i] - labels[i];
            loss += diff * diff;
        }
        printf("Iteration %d loss: %lf\n", iter+1, loss);
    }

    double end_time = monotonic_seconds();

    printf("\n");
    print_time(end_time - start_time);

    printf("\nFinal weights:\n");
    for (int i = 0; i < dim_points; i++) {
        printf("%d: %lf\n", i, w[i]);
    }

    free(points);
    free(labels);
    free(denom_vals);
    free(XT);
    free(Xw);
    free(w);

    return 0;
}

