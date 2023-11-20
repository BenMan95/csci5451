#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Check inputs
    if (argc != 3) {
        printf("Usage: %s <graph> <labels>\n", argv[0]);
        return 1;
    }

    // Open graphs and labels
    FILE *fh_graph = fopen(argv[1], "r");
    FILE *fh_labels = fopen(argv[2], "r");

    // Read node and edge counts
    int num_nodes, num_edges;
    fscanf(fh_graph, "%d %d", &num_nodes, &num_edges);

    // Load labels
    int *labels = malloc(num_nodes * sizeof(int));
    for (int i = 0; i < num_nodes; i++) {
        fscanf(fh_labels, "%d", &labels[i]);
    }

    // Check edges
    int diffs = 0;
    for (int i = 0; i < num_edges; i++) {
        // Read edge
        int a, b;
        fscanf(fh_graph, "%d %d", &a, &b);

        // Check edge
        if (labels[a] != labels[b]) {
            printf("%d:%d, %d:%d\n", a, labels[a], b, labels[b]);
            diffs++;
        }
    }

    printf("%d different edges\n", diffs);

    fclose(fh_graph);
    fclose(fh_labels);
    free(labels);
    return diffs != 0;
}