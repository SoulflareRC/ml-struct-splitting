#include <stdio.h>

#define SIZE1 112  // Define size of the array of the first struct
#define SIZE2 512   // Define size of the array of the second struct
#define ITERATIONS 151233  // Total number of iterations in the later loops
#define INTERVAL 10000

// Define the first struct with 8 fields
struct StructA {
    int field1;
    int field2;
    int field3;
    int field4;
    int field5;
    int field6;
    int field7;
    int field8;
    double filler[1000]; 
};

// Define the second struct with 5 fields
struct StructB {
    int fieldA;
    int fieldB;
    int fieldC;
    int fieldD;
    int fieldE;
    double filler[1000]; 
};

int main() {
    struct StructA arrA[SIZE1];  // Array of StructA
    struct StructB arrB[SIZE2];  // Array of StructB

    // Initialize the array of StructA
    for (int i = 0; i < SIZE1; i++) {
        arrA[i].field1 = i + 1;
        arrA[i].field2 = (i + 1) * 2;
        arrA[i].field3 = (i + 1) * 3;
        arrA[i].field4 = (i + 1) * 4;
        arrA[i].field5 = (i + 1) * 5;
        arrA[i].field6 = (i + 1) * 6;
        arrA[i].field7 = (i + 1) * 7;
        arrA[i].field8 = (i + 1) * 8;
    }

    // Initialize the array of StructB
    for (int i = 0; i < SIZE2; i++) {
        arrB[i].fieldA = i + 10;
        arrB[i].fieldB = (i + 10) * 2;
        arrB[i].fieldC = (i + 10) * 3;
        arrB[i].fieldD = (i + 10) * 4;
        arrB[i].fieldE = (i + 10) * 5;
    }

    // Loop 1: Access and print fields from StructA
    printf("Loop 1 - Accessing StructA (fields 1, 2, 3):\n");
    for (int i = 0; i < SIZE1; i++) {
        int f = arrA[i].field1; 
        f = arrA[i].field2; 
        f = arrA[i].field3; 
        if(((i+1)%INTERVAL)==0)
        printf("StructA %d: field1 = %d, field2 = %d, field3 = %d\n",
               i, arrA[i].field1, arrA[i].field2, arrA[i].field3);
    }

    // Loop 2: Access and print fields from StructB
    printf("\nLoop 2 - Accessing StructB (fields A, B, C) with more iterations:\n");
    for (int i = 0; i < ITERATIONS; i++) {
        // Use modulo to handle out-of-bounds access by wrapping the index
        int idx = i % SIZE2;
        int f = 0; 
        f = arrB[idx].fieldA; 
        f = arrB[idx].fieldB; 
        f = arrB[idx].fieldC;  
        if(((i+1)%INTERVAL)==0)
        printf("Iteration %d - StructB %d: fieldA = %d, fieldB = %d, fieldC = %d\n",
               i, idx, arrB[idx].fieldA, arrB[idx].fieldB, arrB[idx].fieldC);
    }

    // Loop 3: Mix access between StructA and StructB and calculate fields in StructB
    printf("\nLoop 3 - Accessing both StructA and StructB with calculated fields:\n");
    for (int i = 0; i < ITERATIONS; i++) {
        int idxA = i % SIZE1;
        int idxB = i % SIZE2;

        // Calculate some fields in StructB based on StructA values
        arrB[idxB].fieldD = arrA[idxA].field4 + arrA[idxA].field5;
        arrB[idxB].fieldE = arrA[idxA].field6 * 2;
        int f = 0; 
        f = arrB[idxB].fieldD; 
        f = arrB[idxB].fieldE;  
        f = arrA[idxA].field4 + arrA[idxA].field5; 
        f = arrA[idxA].field6 + arrA[idxA].field5; 
        if(((i+1)%INTERVAL)==0)
        printf("Iteration %d - StructA %d: field4 = %d, field5 = %d | StructB %d: fieldD (calculated) = %d, fieldE (calculated) = %d\n",
               i, idxA, arrA[idxA].field4, arrA[idxA].field5, idxB, arrB[idxB].fieldD, arrB[idxB].fieldE);
    }

    return 0;
}
