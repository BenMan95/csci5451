#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

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
                int* dim_points, double** points)
{
    int result;

    FILE* fh = fopen(file_name, "r");
    if (fh == NULL) {
        perror("fopen");
        return -1;
    }

    result = fscanf(fh, "%d %d", num_points, dim_points);
    if (result == EOF) {
        perror("fscanf");
        fclose(fh);
        return -1;
    }

    int eles = (*num_points) * (*dim_points);
    *points = malloc(eles * sizeof(double));
    if (*points == NULL) {
        perror("malloc");
        fclose(fh);
        return -1;
    }

    for (int i = 0; i < eles; i++) {
        result = fscanf(fh, "%lf", *points + i);
        if (result == EOF) {
            perror("fscanf");
            free(*points);
            fclose(fh);
            return -1;
        }
    }

    result = fclose(fh);
    if (result) {
        perror("fclose");
        free(*points);
        return -1;
    }

    return 0;
}

/**
 * @brief Load label data from a file
 *
 * @param file_name The name of the file to load data from
 * @param num_points Where to write the number of labels to
 * @param labels The array to write the labels to
 *
 * @return Returns 0 on success, -1 on error
 */
int load_labels(char* file_name, int* num_labels, double** labels)
{
    int result;

    FILE* fh = fopen(file_name, "r");
    if (fh == NULL) {
        perror("fopen");
        return -1;
    }

    result = fscanf(fh, "%d", num_labels);
    if (result == EOF) {
        perror("fscanf");
        fclose(fh);
        return -1;
    }

    *labels = malloc(*num_labels * sizeof(double));
    if (*labels == NULL) {
        perror("malloc");
        fclose(fh);
        return -1;
    }

    for (int i = 0; i < *num_labels; i++) {
        result = fscanf(fh, "%lf", *labels + i);
        if (result == EOF) {
            perror("fscanf");
            free(*labels);
            fclose(fh);
            return -1;
        }
    }

    result = fclose(fh);
    if (result) {
        perror("fclose");
        free(*labels);
        return -1;
    }

    return 0;
}

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
              int* dim_points, double** points, double** labels)
{
    int result;

    int num1;
    result = load_points(points_file, &num1, dim_points, points);
    if (result) {
        return -1;
    }

    int num2;
    result = load_labels(labels_file, &num2, labels);
    if (result) {
        free(*points);
        return -1;
    }

    if (num1 != num2) {
        fprintf(stderr, "Point counts do not match\n");
        free(*points);
        free(*labels);
        return -1;
    }

    *num_points = num1;

    return 0;
}

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
                   double** XT, double** Xw, double** w) {
    *denom_vals = malloc(dim_points * sizeof(double));
    if (*denom_vals == NULL) {
        perror("malloc");
        return -1;
    }

    *XT = malloc(num_points * dim_points * sizeof(double));
    if (*XT == NULL) {
        perror("malloc");
        free(*denom_vals);
        return -1;
    }

    *Xw = malloc(num_points * sizeof(double));
    if (*Xw == NULL) {
        perror("malloc");
        free(*denom_vals);
        free(*XT);
        return -1;
    }

    *w = malloc(dim_points * sizeof(double));
    if (*w == NULL) {
        perror("malloc");
        free(*denom_vals);
        free(*XT);
        free(*Xw);
        return -1;
    }

    return 0;
}
