#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

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

// Global variables used for thread control
int count;
int threads;
pthread_mutex_t lock;

// Global variables that are part of the algorithm
int num_points; // n
int dim_points; // m
double* points; // X
double* labels; // y
double* denom_vals; // Denominator values
double* XT; // X^T
double* Xw; // X*w
double* w;

/**
 * @brief Clean up global variables
 *
 * @param step Designates which variables need to be freed
 *             0 - points and labels
 *             1 - points, labels, denom_vals, XT, Xw, and w
 *             2 - Previous vars and lock
 */
void cleanup(int step)
{
    free(points);
    free(labels);

    if (step < 1) {
        return;
    }

    free(denom_vals);
    free(XT);
    free(Xw);
    free(w);

    if (step < 2) {
        return;
    }

    pthread_mutex_destroy(&lock);
}

void* init_func(void* ptr)
{
    // Get id for thread
    pthread_mutex_lock(&lock);
    int id = count++;
    pthread_mutex_unlock(&lock);

    // Determine range to iterate over
    int a = dim_points / threads;
    int b = dim_points % threads;
    int start = a*id + (id<b ? id : b);;
    int end = start + a + (id<b);

    // Compute X^T and denominator values
    // Initialize w to all 0s
    for (int i = start; i < end; i++) {
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

    // Determine range to iterate over
    a = num_points / threads;
    b = num_points % threads;
    start = a*id + (id<b ? id : b);;
    end = start + a + (id<b);

    // Initialize Xw to all 0s
    for (int i = start; i < end; i++) {
        Xw[i] = 0;
    }

    return NULL;
}

typedef struct {
    int i;
    double* row;
    //double wi_new;
} args_t;

void* compute_numerator(void* ptr)
{
    args_t* args = (args_t*) ptr;

    // Get id for thread
    pthread_mutex_lock(&lock);
    int id = count++;
    pthread_mutex_unlock(&lock);

    // Determine range to iterate over
    int a = num_points / threads;
    int b = num_points % threads;
    int start = a*id + (id<b ? id : b);;
    int end = start + a + (id<b);

    // Compute new w_i value
    double sum = 0;
    for (int j = start; j < end; j++) {
        double Xw_minusi = Xw[j] - args->row[j]*w[args->i]; // The jth number in X_{-i}*w_{-i}
        sum += args->row[j] * (labels[j] - Xw_minusi);
    }

    return (void*) sum;
}

void* update_Xw(void* ptr)
{
    // Get id for thread
    pthread_mutex_lock(&lock);
    int id = count++;
    pthread_mutex_unlock(&lock);

    // Determine range to iterate over
    int a = num_points / threads;
    int b = num_points % threads;
    int start = a*id + (id<b ? id : b);;
    int end = start + a + (id<b);

    // Update X*w vector
    //for (int j = start; j < end; j++) {
    //    Xw[j] += row[j] * (wi_new-w[i]);
    //}

    //w[i] = wi_new;
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

    threads = atoi(argv[4]);
    if (threads == 0) {
        perror("atoi");
        return -1;
    }

    int result;

    result = load_data(argv[1], argv[2],
                       &num_points, &dim_points,
                       &points, &labels);
    if (result) {
        return -1;
    }

    result = allocate_space(num_points, dim_points,
                            &denom_vals, &XT, &Xw, &w);
    if (result) {
        cleanup(0);
        return -1;
    }

    result = pthread_mutex_init(&lock, NULL);
    if (result) {
        perror("pthread_mutex_init");
        cleanup(1);
        return -1;
    }

    pthread_t* thread_handles = malloc((threads-1) * sizeof(thread_handles));
    if (thread_handles == NULL) {
        perror("malloc");
        cleanup(2);
        return -1;
    }

    // Begin timing
    double start_time = monotonic_seconds();

    //// Initialization step
    // Create threads
    count = 0;
    for (int i = 0; i < threads-1; i++) {
        pthread_create(&thread_handles[i], NULL, init_func, NULL);
    }
    init_func(NULL); // Perform work in main thread

    // Clean up
    for (int i = 0; i < threads-1; i++) {
        pthread_join(thread_handles[i], NULL);
    }

    // Apply the algorithm
    for (int iter = 0; iter < iterations; iter++) {

        // Compute the new w vector
        for (int i = 0; i < dim_points; i++) {
            args_t args = {
                .i = i,
                .row = XT + i*num_points,
            };

            // Compute new w_i value if denominator is not 0
            // Otherwise, leave w_i as 0
            if (denom_vals[i] == 0) {
                continue;
            }

            //// Compute numerator
            // Create threads
            count = 0;
            for (int i = 0; i < threads-1; i++) {
                pthread_create(&thread_handles[i], NULL, compute_numerator, &i);
            }
            double numerator = (double) compute_numerator(&i); // Perform work in main thread

            // Clean up
            for (int i = 0; i < threads-1; i++) {
                void* retval;
                pthread_join(thread_handles[i], &retval);
                numerator += (double) retval;
            }

            double wi_new = numerator / denom_vals[i];

            // Update X*w vector
            for (int j = 0; j < num_points; j++) {
                Xw[j] += args->row[j] * (wi_new-w[i]);
            }

            w[i] = wi_new;
        }

        // Compute and print out the loss
        double loss = 0;
        for (int i = 0; i < num_points; i++) {
            double val = Xw[i] - labels[i];
            loss += val * val;
        }
        printf("Iteration %d loss: %lf\n", iter+1, loss);
    }

    // Finish timing
    double end_time = monotonic_seconds();
    printf("\n");
    print_time(end_time - start_time);

    printf("\nFinal weights:\n");
    for (int i = 0; i < dim_points; i++) {
        printf("%lf\n", w[i]);
    }

    cleanup(3);
    free(thread_handles);

    return 0;
}

