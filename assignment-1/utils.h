#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Load point data from a file
 *
 * @param file_name The name of the file to load data from
 * @param num_points Where to write the number of points to
 * @param dim_points Where to write the dimensionality of the data to
 * @param points The array to write the points to
 *               points[m*dim_points + n] is the nth coord of the mth point
 *
 * @return Returns 0 on success, -1 on error
 */
int load_points(char* file_name, int* num_points,
                int* dim_points, double** points);

/**
 * @brief Load label data from a file
 *
 * @param file_name The name of the file to load data from
 * @param num_points Where to write the number of labels to
 * @param labels The array to write the labels to
 *
 * @return Returns 0 on success, -1 on error
 */
int load_labels(char* file_name, int* num_labels, double** labels);

/**
 * @brief Combines the function of load_points() and load_labels()
 *
 * @param points_file The name of the file to load points from
 * @param labels_file The name of the file to load labels from
 * @param num_points Where to write the number of points to
 * @param dim_points Where to write the dimensionality of the data to
 * @param points The array to write the points to
 * @param labels The array to write the labels to
 *
 * @return Returns 0 on success, -1 on error
 */
int load_data(char* points_file, char* labels_file, int* num_points,
              int* dim_points, double** points, double** labels);

/**
 * @brief Allocate remaining space needed for the algorithm
 *
 * @param num_points The number of points
 * @param num_points The dimension of the points
 * @param denom_vals Where to save pointer for denominator values
 * @param XT Where to save pointer for XT matrix
 * @param Xw Where to save pointer for Xw vector
 * @param x Where to save pointer for w vector
 *
 * @return Returns 0 on success, -1 on error
 */
int allocate_space(int num_points, int dim_points, double** denom_vals,
                   double** XT, double** Xw, double** w);

#endif

