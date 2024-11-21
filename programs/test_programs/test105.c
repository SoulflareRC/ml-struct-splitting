#include <stdio.h>
#include <string.h>

#define MAX_ANIMALS 4
#define MAX_PLANTS 3

// Struct for a Plant
typedef struct {
    char name[30];
    int health; // Health of the plant (0-100)
    int growthRate; // How much it grows each cycle
} Plant;

// Struct for an Animal
typedef struct {
    char name[30];
    int energy; // Energy level of the animal (0-100)
    int consumptionRate; // Energy consumed per cycle
    int grazingEffect; // How much it reduces plant health
} Animal;

// Struct for the Environment
typedef struct {
    char name[30];
    int temperature; // Environment temperature
    int rainfall; // Rainfall level
} Environment;

// Function prototypes
void simulateEcosystem(Plant plants[], int plantCount, Animal animals[], int animalCount, Environment env, int cycles);
void displayEcosystem(Plant plants[], int plantCount, Animal animals[], int animalCount, Environment env);

int main() {
    // Define plants
    Plant plants[MAX_PLANTS];
    strcpy(plants[0].name, "Grass");
    plants[0].health = 100;
    plants[0].growthRate = 10;

    strcpy(plants[1].name, "Bush");
    plants[1].health = 80;
    plants[1].growthRate = 8;

    strcpy(plants[2].name, "Tree");
    plants[2].health = 90;
    plants[2].growthRate = 5;

    // Define animals
    Animal animals[MAX_ANIMALS];
    strcpy(animals[0].name, "Deer");
    animals[0].energy = 60;
    animals[0].consumptionRate = 15;
    animals[0].grazingEffect = 20;

    strcpy(animals[1].name, "Rabbit");
    animals[1].energy = 50;
    animals[1].consumptionRate = 10;
    animals[1].grazingEffect = 10;

    strcpy(animals[2].name, "Cow");
    animals[2].energy = 80;
    animals[2].consumptionRate = 20;
    animals[2].grazingEffect = 25;

    strcpy(animals[3].name, "Goat");
    animals[3].energy = 70;
    animals[3].consumptionRate = 12;
    animals[3].grazingEffect = 15;

    // Define environment
    Environment env;
    strcpy(env.name, "Savannah");
    env.temperature = 30;
    env.rainfall = 50;

    // Simulate the ecosystem for 5 cycles
    simulateEcosystem(plants, MAX_PLANTS, animals, MAX_ANIMALS, env, 5);

    return 0;
}

// Function to simulate the ecosystem
void simulateEcosystem(Plant plants[], int plantCount, Animal animals[], int animalCount, Environment env, int cycles) {
    for (int cycle = 1; cycle <= cycles; cycle++) {
        printf("\nCycle %d:\n", cycle);

        // Simulate plant growth
        for (int i = 0; i < plantCount; i++) {
            plants[i].health += plants[i].growthRate;
            if (plants[i].health > 100) plants[i].health = 100; // Cap health at 100
        }

        // Simulate animal grazing
        for (int i = 0; i < animalCount; i++) {
            for (int j = 0; j < plantCount; j++) {
                plants[j].health -= animals[i].grazingEffect;
                if (plants[j].health < 0) plants[j].health = 0; // Ensure health doesn't go negative
            }
            animals[i].energy -= animals[i].consumptionRate;
            if (animals[i].energy < 0) animals[i].energy = 0; // Ensure energy doesn't go negative
        }

        // Adjust environment based on conditions
        if (env.temperature > 35) {
            for (int i = 0; i < plantCount; i++) {
                plants[i].health -= 5; // Heat stress
            }
        }

        if (env.rainfall < 30) {
            for (int i = 0; i < plantCount; i++) {
                plants[i].health -= 10; // Drought stress
            }
        }

        displayEcosystem(plants, plantCount, animals, animalCount, env);
    }
}

// Function to display the state of the ecosystem
void displayEcosystem(Plant plants[], int plantCount, Animal animals[], int animalCount, Environment env) {
    printf("\nEnvironment: %s | Temperature: %d°C | Rainfall: %dmm\n", env.name, env.temperature, env.rainfall);

    printf("\nPlants:\n");
    for (int i = 0; i < plantCount; i++) {
        printf("  %s - Health: %d\n", plants[i].name, plants[i].health);
    }

    printf("\nAnimals:\n");
    for (int i = 0; i < animalCount; i++) {
        printf("  %s - Energy: %d\n", animals[i].name, animals[i].energy);
    }
}
