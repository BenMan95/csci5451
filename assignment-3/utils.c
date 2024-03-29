#include "utils.h"

#include <stdlib.h>
#include <stdio.h>

void print_labels(
        char const * const filename,
        unsigned const * const labels,
        size_t const nlabels)
{
    size_t i;
    FILE * fout;

    /* open file */
    if((fout = fopen(filename, "w")) == NULL) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    /* write labels to fout */
    for(i = 0; i < nlabels; ++i) {
        fprintf(fout, "%u\n", labels[i]);
    }

    fclose(fout);
}

void print_time25( double const seconds)
{
    printf("2-5 Time: %0.04fs\n", seconds);
}

void print_time5( double const seconds)
{
    printf("5 Time: %0.04fs\n", seconds);
}

void load_graph(
        char *file_name,
        graph_t *graph)
{
    // Open file
    FILE *fh = fopen(file_name, "r");

    // Read node and edge counts
    fscanf(fh, "%d %d", &graph->num_nodes, &graph->num_edges);

    // Allocate graph arrays
    graph->counts = (int*) malloc(graph->num_nodes * sizeof(int));
    graph->offsets = (int*) malloc(graph->num_nodes * sizeof(int));
    graph->edges = (int*) malloc(graph->num_edges * sizeof(int));

    // Initialize edges counts
    for (int i = 0; i < graph->num_nodes; i++) {
        graph->counts[i] = 0;
    }

    // Load data from file
    int *data = (int*) malloc(2 * graph->num_edges * sizeof(int));
    for (int i = 0; i < graph->num_edges; i++) {
        // Read edge
        int a, b;
        fscanf(fh, "%d %d", &a, &b);

        // Increment edge count
        graph->counts[a]++;

        // Add data to temporary array
        data[i*2] = a;
        data[i*2+1] = b;
    }

    // Set edge array offsets
    int offset = 0;
    for (int i = 0; i < graph->num_nodes; i++) {
        graph->offsets[i] = offset;
        offset += graph->counts[i];
    }

    // Create a temporary array for tracking current index in each subarray
    int *indices = (int*) malloc(graph->num_nodes * sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) {
        indices[i] = 0;
    }

    // Load data from earlier array to graph
    for (int i = 0; i < graph->num_edges; i++) {
        int a = data[i*2];
        int b = data[i*2+1];
        graph->edges[graph->offsets[a] + indices[a]++] = b;
    }

    // Clean up
    free(indices);
    free(data);
    fclose(fh);
}

void free_graph_arrays(graph_t *graph)
{
    free(graph->counts);
    free(graph->offsets);
    free(graph->edges);
}
