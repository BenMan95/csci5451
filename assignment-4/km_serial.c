#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define MAX_ITERS 20

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/* OSX timer includes */
#ifdef __MACH__
  #include <mach/mach.h>
  #include <mach/mach_time.h>
#endif

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
#ifdef __MACH__
  /* OSX */
  static mach_timebase_info_data_t info;
  static double seconds_per_unit;
  if(seconds_per_unit == 0) {
    mach_timebase_info(&info);
    seconds_per_unit = (info.numer / info.denom) / 1e9;
  }
  return seconds_per_unit * mach_absolute_time();
#else
  /* Linux systems */
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/**
 * @brief Output the seconds elapsed while clustering.
 *
 * @param seconds Seconds spent on k-means clustering, excluding IO.
 */
static void print_time(double const seconds)
{
  printf("k-means clustering time: %0.04fs\n", seconds);
}

typedef struct {
    int num; // The number of points
    int dim; // The number of dimensions
    double* coords; // The jth coord of the ith points is coords[i*dim+j]
} points_t;

/**
 * @brief Loads point data from a file
 * 
 * @param filename The file to load point data from
 * @param points The points_t struct to load data to
 */
void load_points(
    char* filename,
    points_t* points)
{
    FILE *fh = fopen(filename, "r");
    fscanf(fh, "%d %d", &points->num, &points->dim);

    int size = points->num * points->dim;
    points->coords = (double*) malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        fscanf(fh, "%lf", points->coords + i);
    }

    fclose(fh);
}

/**
 * @brief Writes cluster assignments to clusters.txt
 * 
 * @param clusters The array to read cluster assignments from
 * @param n The number of points
 */
void output_clusters(int* clusters, int n)
{
    FILE *fh = fopen("clusters.txt", "w");

    for (int i = 0; i < n; i++) {
        fprintf(fh, "%d\n", clusters[i]);
    }

    fclose(fh);
}

/**
 * @brief Output centroids to centroids.txt
 * 
 * @param centroids The points_t struct to read centroids from
 */
void output_centroids(points_t centroids)
{
    FILE *fh = fopen("centroids.txt", "w");

    fprintf(fh, "%d %d\n", centroids.num, centroids.dim);
    for (int i = 0; i < centroids.num; i++) {
        for (int j = 0; j < centroids.dim; j++) {
            if (j != 0) {
                fprintf(fh, " ");
            }
            fprintf(fh, "%lf", centroids.coords[i*centroids.dim + j]);
        }
        fprintf(fh, "\n");
    }

    fclose(fh);
}

/**
 * @brief Determine the nearest centroid to a point
 * 
 * @param centroids The list of centroids to use
 * @param point The point to determine the nearest centroid for
 *              The dimension should match the dimension of the centroids list
 * 
 * @returns The index of the nearest centroid
 */
int nearest_centroid(points_t centroids, double* point)
{
    double best_dist = DBL_MAX; // Uses squared distance
    int best_centroid;

    for (int i = 0; i < centroids.num; i++) {
        // Compute distance to the ith centroid
        double dist = 0;
        for (int j = 0; j < centroids.dim; j++) {
            double diff = point[j] - centroids.coords[i*centroids.dim + j];
            dist += diff * diff;
        }

        // Update the best distance and centroid
        if (dist < best_dist) {
            best_dist = dist;
            best_centroid = i;
        }
    }

    return best_centroid;
}

/**
 * @brief Computes new centroids based on current clusters
 * 
 * @param centroids Where to write the new centroids to
 * @param points Where to read the points from
 * @param clusters The current cluster each point is assigned to
 * @param counts An array to be used for temporary storage
 */
void compute_centroids(points_t centroids, points_t points, int* clusters, int* counts)
{
    // Initialize counts/centroids to 0
    for (int i = 0; i < centroids.num; i++) {
        counts[i] = 0;
        for (int j = 0; j < centroids.dim; j++) {
            int idx = i*centroids.dim + j;
            centroids.coords[idx] = 0;
        }
    }

    // Compute coordinate sums and counts for each cluster
    for (int i = 0; i < points.num; i++) {
        int cluster = clusters[i];
        counts[cluster]++;
        for (int j = 0; j < points.dim; j++) {
            int c_idx = cluster*centroids.dim + j;
            int p_idx = i*points.dim + j;
            centroids.coords[c_idx] += points.coords[p_idx];
        }
    }

    // Divide to get averages
    for (int i = 0; i < centroids.num; i++) {
        int count = counts[i];
        for (int j = 0; j < centroids.dim; j++) {
            int idx = i*centroids.dim + j;
            centroids.coords[idx] = centroids.coords[idx] / count;
        }
    }
}

/**
 * @brief Assigns clusters for a set of points and centroids
 * 
 * @param points The points to assign clusters for
 * @param centroids The centroids to assigne clusters to
 * @param clusters Where to write cluster assignments to
 * 
 * @returns If cluster assignments have converged
 */
int assign_clusters(points_t points, points_t centroids, int* clusters)
{
    int converged = 1;
    for (int i = 0; i < points.num; i++) {
        // Determine new cluster
        int new = nearest_centroid(centroids, points.coords + i*points.dim);

        // Update cluster if necessary
        if (new != clusters[i]) {
            converged = 0;
            clusters[i] = new;
        }
    }
    return converged;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <input> <clusters>\n", argv[0]);
        return -1;
    }

    // LOAD / INITIALIZE DATA --------------------------------------------------

    // Load point data
    points_t points;
    load_points(argv[1], &points);

    // Initialize centroids struct
    points_t centroids;
    centroids.num = atoi(argv[2]);
    centroids.dim = points.dim;
    centroids.coords = malloc(centroids.num * centroids.dim * sizeof(double));

    int* clusters = malloc(points.num * sizeof(int)); // The cluster each point is assigned to
    int* counts = malloc(centroids.num * sizeof(int)); // The number of points assigned to each cluster

    // PERFORM ALGORITHM -------------------------------------------------------
    double t0 = monotonic_seconds();

    // Select initial K centroids
    for (int i = 0; i < centroids.num * centroids.dim; i++) {
        centroids.coords[i] = points.coords[i];
    }

    // Assign initial clusters
    for (int i = 0; i < points.num; i++) {
        clusters[i] = nearest_centroid(centroids, points.coords + i*points.dim);
        printf("%d\n", clusters[i]);
    }

    // Perform algorithm until convergence or iteration limit
    int iters = 0;
    int converged = 0;
    while (!converged && iters < MAX_ITERS) {
        compute_centroids(centroids, points, clusters, counts);
        converged = assign_clusters(points, centroids, clusters);
        iters++;
        printf("%d %d\n", iters, converged);
    }

    // OUTPUT RESULTS / CLEAN UP -----------------------------------------------
    double t1 = monotonic_seconds();
    print_time(t1 - t0);

    output_clusters(clusters, points.num);
    output_centroids(centroids);

    free(points.coords);
    free(centroids.coords);
    free(clusters);
    free(counts);

    return 0;
}
