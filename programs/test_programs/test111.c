#include <stdio.h>
#include <string.h>

#define GRID_SIZE 10
#define SIMULATION_STEPS 15

// Struct Definitions
typedef struct {
    int id;
    int isBurning;  // 1 if the tree is on fire, 0 otherwise
    int isBurned;   // 1 if the tree has burned down, 0 otherwise
} Tree;

typedef struct {
    int x, y;       // Coordinates in the grid
} GridCell;

// Function Prototypes
void initializeSimulation(Tree trees[], GridCell grid[]);
void spreadFire(Tree trees[], GridCell grid[]);
void copyTreeStates(Tree dest[], Tree src[]);
void printSimulationState(Tree trees[]);

int main() {
    Tree trees[GRID_SIZE * GRID_SIZE];
    GridCell grid[GRID_SIZE * GRID_SIZE];

    initializeSimulation(trees, grid);

    for (int step = 0; step < SIMULATION_STEPS; step++) {
        printf("Step %d:\n", step + 1);
        spreadFire(trees, grid);
        printSimulationState(trees);
        printf("\n");
    }

    return 0;
}

// Function Definitions

void initializeSimulation(Tree trees[], GridCell grid[]) {
    // Initialize the forest grid
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int index = i * GRID_SIZE + j;
            grid[index].x = i;
            grid[index].y = j;

            // Assign each tree a unique ID
            trees[index].id = index + 1;
            trees[index].isBurning = 0;
            trees[index].isBurned = 0;
        }
    }

    // Set initial fire at the center
    int center = GRID_SIZE / 2;
    int centerIndex = center * GRID_SIZE + center;
    trees[centerIndex].isBurning = 1;
}

void spreadFire(Tree trees[], GridCell grid[]) {
    Tree nextTrees[GRID_SIZE * GRID_SIZE];

    // Copy current tree states into the temporary array
    copyTreeStates(nextTrees, trees);

    // Spread fire
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int index = i * GRID_SIZE + j;
            Tree *currentTree = &trees[index];
            if (currentTree->isBurning) {
                currentTree->isBurned = 1;
                currentTree->isBurning = 0;

                // Spread fire to neighboring cells
                if (i > 0 && !trees[(i - 1) * GRID_SIZE + j].isBurned)
                    nextTrees[(i - 1) * GRID_SIZE + j].isBurning = 1;
                if (i < GRID_SIZE - 1 && !trees[(i + 1) * GRID_SIZE + j].isBurned)
                    nextTrees[(i + 1) * GRID_SIZE + j].isBurning = 1;
                if (j > 0 && !trees[i * GRID_SIZE + (j - 1)].isBurned)
                    nextTrees[i * GRID_SIZE + (j - 1)].isBurning = 1;
                if (j < GRID_SIZE - 1 && !trees[i * GRID_SIZE + (j + 1)].isBurned)
                    nextTrees[i * GRID_SIZE + (j + 1)].isBurning = 1;
            }
        }
    }

    // Update the original array with the next state
    copyTreeStates(trees, nextTrees);
}

void copyTreeStates(Tree dest[], Tree src[]) {
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        dest[i].id = src[i].id;
        dest[i].isBurning = src[i].isBurning;
        dest[i].isBurned = src[i].isBurned;
    }
}

void printSimulationState(Tree trees[]) {
    printf("Grid State:\n");
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int index = i * GRID_SIZE + j;
            Tree *tree = &trees[index];
            if (tree->isBurned) {
                printf(" X ");
            } else if (tree->isBurning) {
                printf(" * ");
            } else {
                printf(" O ");
            }
        }
        printf("\n");
    }
}
