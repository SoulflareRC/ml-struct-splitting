#include <stdio.h>
#include <string.h>

#define MAX_CARS 50
#define MAX_ROADS 4
#define SIMULATION_STEPS 30

// Struct Definitions
typedef struct {
    int carID;
    int roadID;       // Road on which the car is
    double position;  // Position on the road (0-100 scale)
    double speed;     // Speed in units per second
    int hasCrossed;   // 1 if car has crossed the intersection
} Car;

typedef struct {
    int lightID;
    int roadID;       // Road controlled by the light
    char state[10];   // "Green" or "Red"
    int timeRemaining; // Time until the state changes
} TrafficLight;

typedef struct {
    int roadID;
    char name[20];
    int maxCars;      // Maximum cars on the road
    double length;    // Length of the road
} Road;

typedef struct {
    Car cars[MAX_CARS];
    TrafficLight lights[MAX_ROADS];
    Road roads[MAX_ROADS];
    int carCount;
    int step; // Current simulation step
} Simulation;

// Function Prototypes
void initializeSimulation(Simulation *sim);
void updateTrafficLights(Simulation *sim);
void moveCars(Simulation *sim);
void addCars(Simulation *sim, int newCars);
void printSimulationState(Simulation *sim);

int main() {
    Simulation sim;
    initializeSimulation(&sim);

    for (int step = 0; step < SIMULATION_STEPS; step++) {
        printf("Step %d:\n", step + 1);
        updateTrafficLights(&sim);
        moveCars(&sim);
        if (step % 5 == 0) {
            addCars(&sim, 5); // Add new cars every 5 steps
        }
        printSimulationState(&sim);
        printf("\n");
    }

    return 0;
}

// Function Definitions

void initializeSimulation(Simulation *sim) {
    sim->carCount = 0;
    sim->step = 0;

    // Initialize Roads
    for (int i = 0; i < MAX_ROADS; i++) {
        sim->roads[i].roadID = i + 1;
        sprintf(sim->roads[i].name, "Road %d", i + 1);
        sim->roads[i].maxCars = 15;
        sim->roads[i].length = 100.0;
    }

    // Initialize Traffic Lights
    for (int i = 0; i < MAX_ROADS; i++) {
        sim->lights[i].lightID = i + 1;
        sim->lights[i].roadID = i + 1;
        strcpy(sim->lights[i].state, (i % 2 == 0) ? "Green" : "Red");
        sim->lights[i].timeRemaining = 5 + (i * 3);
    }

    // Add Initial Cars
    addCars(sim, 10);
}

void updateTrafficLights(Simulation *sim) {
    for (int i = 0; i < MAX_ROADS; i++) {
        TrafficLight *light = &sim->lights[i];
        light->timeRemaining--;
        if (light->timeRemaining <= 0) {
            if (strcmp(light->state, "Green") == 0) {
                strcpy(light->state, "Red");
                light->timeRemaining = 7;
            } else {
                strcpy(light->state, "Green");
                light->timeRemaining = 5;
            }
        }
    }
}

void moveCars(Simulation *sim) {
    for (int i = 0; i < sim->carCount; i++) {
        Car *car = &sim->cars[i];
        if (car->hasCrossed) {
            continue; // Skip cars that have crossed
        }

        // Get traffic light state for the car's road
        TrafficLight *light = NULL;
        for (int j = 0; j < MAX_ROADS; j++) {
            if (sim->lights[j].roadID == car->roadID) {
                light = &sim->lights[j];
                break;
            }
        }

        if (light == NULL) continue;

        if (strcmp(light->state, "Green") == 0 || car->position < 90) {
            car->position += car->speed;
        }

        if (car->position >= 100.0) {
            car->position = 100.0;
            car->hasCrossed = 1;
        }
    }
}

void addCars(Simulation *sim, int newCars) {
    for (int i = 0; i < newCars; i++) {
        if (sim->carCount >= MAX_CARS) break;
        int roadID = (sim->carCount % MAX_ROADS) + 1;

        sim->cars[sim->carCount].carID = sim->carCount + 1;
        sim->cars[sim->carCount].roadID = roadID;
        sim->cars[sim->carCount].position = 0.0;
        sim->cars[sim->carCount].speed = 5.0 + (sim->carCount % 3); // Speed varies between 5 and 7
        sim->cars[sim->carCount].hasCrossed = 0;

        sim->carCount++;
    }
}

void printSimulationState(Simulation *sim) {
    printf("Traffic Lights:\n");
    for (int i = 0; i < MAX_ROADS; i++) {
        printf("  Road %d: %s (%d sec remaining)\n", sim->lights[i].roadID, sim->lights[i].state, sim->lights[i].timeRemaining);
    }

    printf("Cars:\n");
    for (int i = 0; i < sim->carCount; i++) {
        Car *car = &sim->cars[i];
        printf("  Car %d: Road %d, Position %.2f, Speed %.2f, %s\n",
               car->carID, car->roadID, car->position, car->speed,
               car->hasCrossed ? "Crossed" : "On Road");
    }
}
