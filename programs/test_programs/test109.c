#include <stdio.h>
#include <string.h>

#define MAX_VEHICLES 20
#define MAX_DRIVERS 10
#define MAX_TRIPS 50

// Struct Definitions
typedef struct {
    int driverID;
    char name[50];
    char licenseNumber[20];
    int assignedVehicleID; // Links to a vehicle
} Driver;

typedef struct {
    int vehicleID;
    char type[20]; // e.g., "Truck", "Car"
    char licensePlate[10];
    double mileage; // Kilometers per liter
    double fuelLevel; // Current fuel level in liters
} Vehicle;

typedef struct {
    int tripID;
    int vehicleID;  // Linked to a vehicle
    int driverID;   // Linked to a driver
    double distance; // Distance of the trip in kilometers
    char date[11];  // YYYY-MM-DD
} Trip;

// Function Prototypes
void addDriver(Driver drivers[], int *driverCount, int id, const char *name, const char *licenseNumber, int vehicleID);
void addVehicle(Vehicle vehicles[], int *vehicleCount, int id, const char *type, const char *licensePlate, double mileage, double fuelLevel);
void recordTrip(Trip trips[], int *tripCount, Vehicle vehicles[], int vehicleCount, Driver drivers[], int driverCount, int tripID, int driverID, int vehicleID, double distance, const char *date);
void refillFuel(Vehicle vehicles[], int vehicleCount, int vehicleID, double fuelAdded);
void displayVehicles(Vehicle vehicles[], int vehicleCount);
void displayDrivers(Driver drivers[], int driverCount);
void displayTrips(Trip trips[], int tripCount);

int main() {
    Vehicle vehicles[MAX_VEHICLES];
    Driver drivers[MAX_DRIVERS];
    Trip trips[MAX_TRIPS];

    int vehicleCount = 0, driverCount = 0, tripCount = 0;

    // Initialize Vehicles
    for (int i = 0; i < 5; i++) {
        char type[20], licensePlate[10];
        sprintf(type, i % 2 == 0 ? "Truck" : "Car");
        sprintf(licensePlate, "ABC-%02d", i + 1);
        addVehicle(vehicles, &vehicleCount, i + 1, type, licensePlate, (i + 10) * 1.5, 50.0);
    }

    // Initialize Drivers
    for (int i = 0; i < 3; i++) {
        char name[50], licenseNumber[20];
        sprintf(name, "Driver %c", 'A' + i);
        sprintf(licenseNumber, "LIC%04d", 1000 + i);
        addDriver(drivers, &driverCount, i + 1, name, licenseNumber, (i % vehicleCount) + 1);
    }

    // Record Trips
    for (int i = 0; i < 5; i++) {
        char date[11];
        sprintf(date, "2024-11-%02d", 22 + i);
        recordTrip(trips, &tripCount, vehicles, vehicleCount, drivers, driverCount, i + 1, (i % driverCount) + 1, (i % vehicleCount) + 1, 150.0 + (i * 10), date);
    }

    // Refill Fuel for Selected Vehicles
    for (int i = 0; i < 3; i++) {
        refillFuel(vehicles, vehicleCount, i + 1, 20.0 + (i * 5));
    }

    // Display All Data
    displayVehicles(vehicles, vehicleCount);
    displayDrivers(drivers, driverCount);
    displayTrips(trips, tripCount);

    return 0;
}

// Function Definitions

void addDriver(Driver drivers[], int *driverCount, int id, const char *name, const char *licenseNumber, int vehicleID) {
    if (*driverCount < MAX_DRIVERS) {
        drivers[*driverCount].driverID = id;
        strcpy(drivers[*driverCount].name, name);
        strcpy(drivers[*driverCount].licenseNumber, licenseNumber);
        drivers[*driverCount].assignedVehicleID = vehicleID;
        (*driverCount)++;
    } else {
        printf("Maximum number of drivers reached.\n");
    }
}

void addVehicle(Vehicle vehicles[], int *vehicleCount, int id, const char *type, const char *licensePlate, double mileage, double fuelLevel) {
    if (*vehicleCount < MAX_VEHICLES) {
        vehicles[*vehicleCount].vehicleID = id;
        strcpy(vehicles[*vehicleCount].type, type);
        strcpy(vehicles[*vehicleCount].licensePlate, licensePlate);
        vehicles[*vehicleCount].mileage = mileage;
        vehicles[*vehicleCount].fuelLevel = fuelLevel;
        (*vehicleCount)++;
    } else {
        printf("Maximum number of vehicles reached.\n");
    }
}

void recordTrip(Trip trips[], int *tripCount, Vehicle vehicles[], int vehicleCount, Driver drivers[], int driverCount, int tripID, int driverID, int vehicleID, double distance, const char *date) {
    if (*tripCount < MAX_TRIPS) {
        int vehicleIndex = -1, driverIndex = -1;
        for (int i = 0; i < vehicleCount; i++) {
            if (vehicles[i].vehicleID == vehicleID) {
                vehicleIndex = i;
                break;
            }
        }
        for (int i = 0; i < driverCount; i++) {
            if (drivers[i].driverID == driverID) {
                driverIndex = i;
                break;
            }
        }
        if (vehicleIndex == -1 || driverIndex == -1) {
            printf("Invalid driver or vehicle ID.\n");
            return;
        }

        double fuelNeeded = distance / vehicles[vehicleIndex].mileage;
        if (vehicles[vehicleIndex].fuelLevel >= fuelNeeded) {
            vehicles[vehicleIndex].fuelLevel -= fuelNeeded;
            trips[*tripCount].tripID = tripID;
            trips[*tripCount].vehicleID = vehicleID;
            trips[*tripCount].driverID = driverID;
            trips[*tripCount].distance = distance;
            strcpy(trips[*tripCount].date, date);
            (*tripCount)++;
        } else {
            printf("Not enough fuel for vehicle ID %d.\n", vehicleID);
        }
    } else {
        printf("Maximum number of trips reached.\n");
    }
}

void refillFuel(Vehicle vehicles[], int vehicleCount, int vehicleID, double fuelAdded) {
    for (int i = 0; i < vehicleCount; i++) {
        if (vehicles[i].vehicleID == vehicleID) {
            vehicles[i].fuelLevel += fuelAdded;
            printf("Refilled %.2f liters of fuel for vehicle ID %d.\n", fuelAdded, vehicleID);
            return;
        }
    }
    printf("Vehicle ID %d not found for refueling.\n", vehicleID);
}

void displayVehicles(Vehicle vehicles[], int vehicleCount) {
    printf("\nVehicles:\n");
    for (int i = 0; i < vehicleCount; i++) {
        printf("ID: %d, Type: %s, License Plate: %s, Mileage: %.2f, Fuel Level: %.2f\n",
               vehicles[i].vehicleID, vehicles[i].type, vehicles[i].licensePlate, vehicles[i].mileage, vehicles[i].fuelLevel);
    }
}

void displayDrivers(Driver drivers[], int driverCount) {
    printf("\nDrivers:\n");
    for (int i = 0; i < driverCount; i++) {
        printf("ID: %d, Name: %s, License: %s, Assigned Vehicle ID: %d\n",
               drivers[i].driverID, drivers[i].name, drivers[i].licenseNumber, drivers[i].assignedVehicleID);
    }
}

void displayTrips(Trip trips[], int tripCount) {
    printf("\nTrips:\n");
    for (int i = 0; i < tripCount; i++) {
        printf("Trip ID: %d, Driver ID: %d, Vehicle ID: %d, Distance: %.2f, Date: %s\n",
               trips[i].tripID, trips[i].driverID, trips[i].vehicleID, trips[i].distance, trips[i].date);
    }
}
