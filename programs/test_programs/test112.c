#include <stdio.h>

#define WAREHOUSE_SIZE 5
#define NUM_WORKERS 3
#define NUM_STEPS 10

// Struct Definitions
typedef struct {
    int id;            // Unique crate ID
    int weight;        // Weight of the crate
    int currentX;      // Current X position in the warehouse
    int currentY;      // Current Y position in the warehouse
    int destinationX;  // Destination X position
    int destinationY;  // Destination Y position
} Crate;

typedef struct {
    int id;         // Worker ID
    int x, y;       // Current worker position
    int carryingCrate; // ID of the crate being carried, -1 if none
} Worker;

// Function Prototypes
void initializeWarehouse(Crate crates[], Worker workers[]);
void assignTasks(Crate crates[], Worker workers[]);
void moveWorkers(Crate crates[], Worker workers[]);
void printWarehouseState(Crate crates[], Worker workers[]);

int main() {
    Crate crates[WAREHOUSE_SIZE * WAREHOUSE_SIZE];
    Worker workers[NUM_WORKERS];

    initializeWarehouse(crates, workers);

    for (int step = 0; step < NUM_STEPS; step++) {
        printf("Step %d:\n", step + 1);
        assignTasks(crates, workers);
        moveWorkers(crates, workers);
        printWarehouseState(crates, workers);
        printf("\n");
    }

    return 0;
}

// Function Definitions

void initializeWarehouse(Crate crates[], Worker workers[]) {
    // Initialize crates
    int crateCounter = 0;
    for (int i = 0; i < WAREHOUSE_SIZE; i++) {
        for (int j = 0; j < WAREHOUSE_SIZE; j++) {
            int index = i * WAREHOUSE_SIZE + j;
            crates[index].id = ++crateCounter;
            crates[index].weight = 10 + (index % 5) * 5;  // Variable weights
            crates[index].currentX = i;
            crates[index].currentY = j;
            crates[index].destinationX = (i + 1) % WAREHOUSE_SIZE;
            crates[index].destinationY = (j + 1) % WAREHOUSE_SIZE;
        }
    }

    // Initialize workers
    for (int i = 0; i < NUM_WORKERS; i++) {
        workers[i].id = i + 1;
        workers[i].x = i % WAREHOUSE_SIZE;    // Spread workers across the warehouse
        workers[i].y = i % WAREHOUSE_SIZE;
        workers[i].carryingCrate = -1;  // Not carrying anything initially
    }
}

void assignTasks(Crate crates[], Worker workers[]) {
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (workers[w].carryingCrate == -1) {  // If not carrying a crate
            for (int c = 0; c < WAREHOUSE_SIZE * WAREHOUSE_SIZE; c++) {
                if (crates[c].currentX == workers[w].x &&
                    crates[c].currentY == workers[w].y &&
                    (crates[c].currentX != crates[c].destinationX ||
                     crates[c].currentY != crates[c].destinationY)) {
                    workers[w].carryingCrate = crates[c].id;
                    break;
                }
            }
        }
    }
}

void moveWorkers(Crate crates[], Worker workers[]) {
    for (int w = 0; w < NUM_WORKERS; w++) {
        Worker *worker = &workers[w];
        if (worker->carryingCrate != -1) {  // If carrying a crate
            for (int c = 0; c < WAREHOUSE_SIZE * WAREHOUSE_SIZE; c++) {
                if (crates[c].id == worker->carryingCrate) {
                    Crate *crate = &crates[c];

                    // Move worker and crate towards the destination
                    if (worker->x < crate->destinationX)
                        worker->x++;
                    else if (worker->x > crate->destinationX)
                        worker->x--;

                    if (worker->y < crate->destinationY)
                        worker->y++;
                    else if (worker->y > crate->destinationY)
                        worker->y--;

                    // Update crate's position to match the worker's
                    crate->currentX = worker->x;
                    crate->currentY = worker->y;

                    // Check if the crate reached its destination
                    if (crate->currentX == crate->destinationX &&
                        crate->currentY == crate->destinationY) {
                        worker->carryingCrate = -1;  // Drop the crate
                    }

                    break;
                }
            }
        } else {
            // Move worker randomly if not carrying a crate
            worker->x = (worker->x + 1) % WAREHOUSE_SIZE;
            worker->y = (worker->y + 1) % WAREHOUSE_SIZE;
        }
    }
}

void printWarehouseState(Crate crates[], Worker workers[]) {
    printf("Crates:\n");
    for (int i = 0; i < WAREHOUSE_SIZE * WAREHOUSE_SIZE; i++) {
        printf("Crate %d: (%d, %d) -> (%d, %d), Weight: %d\n",
               crates[i].id, crates[i].currentX, crates[i].currentY,
               crates[i].destinationX, crates[i].destinationY, crates[i].weight);
    }

    printf("Workers:\n");
    for (int i = 0; i < NUM_WORKERS; i++) {
        printf("Worker %d: (%d, %d), Carrying: %d\n",
               workers[i].id, workers[i].x, workers[i].y,
               workers[i].carryingCrate);
    }
}
