#include <stdio.h>
#include <math.h>

// Define a struct for matrix operations
struct Matrix
{
    double determinant;
    double trace;
};

int main()
{
    const int matrixCount = 2;
    // Define arrays of struct instances to store data
    struct Matrix matrices[matrixCount];
    matrices[0].determinant = 1; 
    matrices[0].trace = 2; 
    
    printf("Matrix 1: Determinant=%f Trace=%f\n", matrices[0].determinant,matrices[0].trace); // this caused segfault 

    return 0;
}
