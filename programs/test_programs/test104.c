#include <stdio.h>

#define MAX_SIZE 3

// Define a struct for a matrix
typedef struct {
    int rows;
    int cols;
    int data[MAX_SIZE][MAX_SIZE];
} Matrix;

// Function prototypes
Matrix createMatrix(int rows, int cols, int values[MAX_SIZE][MAX_SIZE]);
void displayMatrix(const Matrix *m);
Matrix addMatrices(const Matrix *a, const Matrix *b);
Matrix subtractMatrices(const Matrix *a, const Matrix *b);
Matrix multiplyMatrices(const Matrix *a, const Matrix *b);
Matrix transposeMatrix(const Matrix *m);

int main() {
    // Predefined input for Matrix A
    int valuesA[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Predefined input for Matrix B
    int valuesB[3][3] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    // Create matrices with the predefined values
    Matrix matrixA = createMatrix(3, 3, valuesA);
    Matrix matrixB = createMatrix(3, 3, valuesB);

    printf("Matrix A:\n");
    displayMatrix(&matrixA);

    printf("\nMatrix B:\n");
    displayMatrix(&matrixB);

    printf("\nAdding matrices:\n");
    Matrix resultAdd = addMatrices(&matrixA, &matrixB);
    displayMatrix(&resultAdd);

    printf("\nSubtracting matrices:\n");
    Matrix resultSubtract = subtractMatrices(&matrixA, &matrixB);
    displayMatrix(&resultSubtract);

    printf("\nMultiplying matrices:\n");
    Matrix resultMultiply = multiplyMatrices(&matrixA, &matrixB);
    displayMatrix(&resultMultiply);

    printf("\nTranspose of Matrix A:\n");
    Matrix transposeA = transposeMatrix(&matrixA);
    displayMatrix(&transposeA);

    return 0;
}

// Create a matrix with given dimensions and values
Matrix createMatrix(int rows, int cols, int values[MAX_SIZE][MAX_SIZE]) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = values[i][j];
        }
    }
    return m;
}

// Display a matrix
void displayMatrix(const Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%d ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Add two matrices
Matrix addMatrices(const Matrix *a, const Matrix *b) {
    Matrix result = createMatrix(a->rows, a->cols, (int[MAX_SIZE][MAX_SIZE]){0});
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result.data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return result;
}

// Subtract two matrices
Matrix subtractMatrices(const Matrix *a, const Matrix *b) {
    Matrix result = createMatrix(a->rows, a->cols, (int[MAX_SIZE][MAX_SIZE]){0});
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result.data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return result;
}

// Multiply two matrices
Matrix multiplyMatrices(const Matrix *a, const Matrix *b) {
    Matrix result = createMatrix(a->rows, b->cols, (int[MAX_SIZE][MAX_SIZE]){0});
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                result.data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    return result;
}

// Transpose a matrix
Matrix transposeMatrix(const Matrix *m) {
    Matrix result = createMatrix(m->cols, m->rows, (int[MAX_SIZE][MAX_SIZE]){0});
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result.data[j][i] = m->data[i][j];
        }
    }
    return result;
}
