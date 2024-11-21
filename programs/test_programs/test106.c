#include <stdio.h>
#include <string.h>

#define MAX_MEMORY_SIZE 10
#define MAX_REGISTERS 4

// Struct for CPU
typedef struct {
    char name[50];
    int clockSpeed;  // in GHz
    int cores;
} CPU;

// Struct for Memory
typedef struct {
    int memory[MAX_MEMORY_SIZE];  // Simulated memory
} Memory;

// Struct for Registers
typedef struct {
    int registers[MAX_REGISTERS];  // Simulated CPU registers
} Registers;

// Function prototypes
void loadDataIntoMemory(Memory *mem, int data[], int size);
void performArithmeticOperation(Registers *regs, int regIndex, int operation, int value);
void displaySystemState(CPU cpu, Memory mem, Registers regs);

int main() {
    // Initialize CPU
    CPU cpu;
    strcpy(cpu.name, "Intel Core i7");
    cpu.clockSpeed = 3;  // 3 GHz
    cpu.cores = 8;

    // Initialize Memory
    Memory mem;
    int data[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    loadDataIntoMemory(&mem, data, MAX_MEMORY_SIZE);

    // Initialize Registers
    Registers regs;
    for (int i = 0; i < MAX_REGISTERS; i++) {
        regs.registers[i] = 0;  // Start with all registers set to 0
    }

    // Perform operations
    performArithmeticOperation(&regs, 0, 1, 5);  // Add 5 to register 0
    performArithmeticOperation(&regs, 1, 2, 3);  // Subtract 3 from register 1
    performArithmeticOperation(&regs, 2, 3, 2);  // Multiply register 2 by 2

    // Display system state
    displaySystemState(cpu, mem, regs);

    return 0;
}

// Function to load data into memory
void loadDataIntoMemory(Memory *mem, int data[], int size) {
    for (int i = 0; i < size; i++) {
        mem->memory[i] = data[i];
    }
}

// Function to perform arithmetic operations on registers
// operation: 1 for addition, 2 for subtraction, 3 for multiplication
void performArithmeticOperation(Registers *regs, int regIndex, int operation, int value) {
    if (regIndex < 0 || regIndex >= MAX_REGISTERS) {
        printf("Invalid register index!\n");
        return;
    }

    switch (operation) {
        case 1:  // Addition
            regs->registers[regIndex] += value;
            break;
        case 2:  // Subtraction
            regs->registers[regIndex] -= value;
            break;
        case 3:  // Multiplication
            regs->registers[regIndex] *= value;
            break;
        default:
            printf("Invalid operation!\n");
            break;
    }
}

// Function to display the current state of the system
void displaySystemState(CPU cpu, Memory mem, Registers regs) {
    printf("CPU: %s, Clock Speed: %d GHz, Cores: %d\n", cpu.name, cpu.clockSpeed, cpu.cores);

    printf("Memory (10 cells): ");
    for (int i = 0; i < MAX_MEMORY_SIZE; i++) {
        printf("%d ", mem.memory[i]);
    }
    printf("\n");

    printf("Registers: ");
    for (int i = 0; i < MAX_REGISTERS; i++) {
        printf("R%d = %d ", i, regs.registers[i]);
    }
    printf("\n");
}
