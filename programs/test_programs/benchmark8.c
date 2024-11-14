#include <stdio.h>
#include <math.h>

// Define a struct for vector operations
struct Vector
{
    double x;
    double y;
    double z;
    double magnitude;
    double direction;
    char description[200]; // Increase the description size
    // Exclude some members from usage
    // int unusedField1;
    // double unusedField2;
    // Add more fields as needed
};

// Define a struct for matrix operations
struct Matrix
{
    double elements[6][6]; // Increase the matrix size
    double determinant;
    double trace;
    // Exclude some members from usage
    // double unusedField3;
    // Add more fields as needed
};

// Define a struct for complex number operations
struct ComplexNumber
{
    double real;
    double imaginary;
    double magnitude;
    double phase;
    // Exclude some members from usage
    // char unusedField4[100];
    // Add more fields as needed
};

// Define a struct for representing particles
struct Particle
{
    int id;
    double mass;
    struct Vector position;
    struct Vector velocity;
    // Add more fields as needed
};

// Define a struct for sensor data
struct SensorData
{
    int sensorId;
    double value;
    char timestamp[30];
    // Add more fields as needed
};

int main()
{
    // Define arrays of struct instances to store data
    struct Vector vectors[3];
    struct Matrix matrices[2];
    struct ComplexNumber complexNumbers[4];
    struct Particle particles[3];
    struct SensorData sensorData[5];

    // Manual member assignment for vectors
    vectors[0].x = 1.0;
    vectors[0].y = 2.0;
    vectors[0].z = 3.0;
    snprintf(vectors[0].description, sizeof(vectors[0].description), "Vector 1 description");

    vectors[1].x = 4.0;
    vectors[1].y = 5.0;
    vectors[1].z = 6.0;
    snprintf(vectors[1].description, sizeof(vectors[1].description), "Vector 2 description");

    vectors[2].x = 7.0;
    vectors[2].y = 8.0;
    vectors[2].z = 9.0;
    snprintf(vectors[2].description, sizeof(vectors[2].description), "Vector 3 description");

    // Manual member assignment for matrices
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            for (int k = 0; k < 6; ++k)
            {
                matrices[i].elements[j][k] = i * 10 + j + k;
            }
        }
    }

    // Manual member assignment for complex numbers
    for (int i = 0; i < 4; ++i)
    {
        complexNumbers[i].real = i + 1;
        complexNumbers[i].imaginary = i - 1;
    }

    // Manual member assignment for particles
    particles[0].id = 1;
    particles[0].mass = 10.0;
    particles[0].position.x = 1.0;
    particles[0].position.y = 2.0;
    particles[0].position.z = 3.0;
    particles[0].velocity.x = 0.5;
    particles[0].velocity.y = 0.0;
    particles[0].velocity.z = -0.5;

    particles[1].id = 2;
    particles[1].mass = 15.0;
    particles[1].position.x = -2.0;
    particles[1].position.y = 3.0;
    particles[1].position.z = 1.0;
    particles[1].velocity.x = 1.0;
    particles[1].velocity.y = -0.5;
    particles[1].velocity.z = 0.0;

    particles[2].id = 3;
    particles[2].mass = 20.0;
    particles[2].position.x = 0.0;
    particles[2].position.y = 0.0;
    particles[2].position.z = 5.0;
    particles[2].velocity.x = 0.0;
    particles[2].velocity.y = 1.0;
    particles[2].velocity.z = 0.0;

    // Manual member assignment for sensor data
    for (int i = 0; i < 5; ++i)
    {
        sensorData[i].sensorId = i + 1;
        sensorData[i].value = i * 2.5;
        snprintf(sensorData[i].timestamp, sizeof(sensorData[i].timestamp), "2023-12-01 12:00:00");
    }

    // Perform mathematical operations on vectors
    for (int i = 0; i < 3; ++i)
    {
        vectors[i].magnitude = sqrt(vectors[i].x * vectors[i].x +
                                    vectors[i].y * vectors[i].y +
                                    vectors[i].z * vectors[i].z);

        vectors[i].direction = atan2(vectors[i].y, vectors[i].x);
    }

    // Perform mathematical operations on matrices
    for (int i = 0; i < 2; ++i)
    {
        // Calculate determinant
        matrices[i].determinant =
            matrices[i].elements[0][0] * (matrices[i].elements[1][1] * matrices[i].elements[2][2] -
                                          matrices[i].elements[2][1] * matrices[i].elements[1][2]) -
            matrices[i].elements[0][1] * (matrices[i].elements[1][0] * matrices[i].elements[2][2] -
                                          matrices[i].elements[2][0] * matrices[i].elements[1][2]) +
            matrices[i].elements[0][2] * (matrices[i].elements[1][0] * matrices[i].elements[2][1] -
                                          matrices[i].elements[2][0] * matrices[i].elements[1][1]);

        // Calculate trace
        matrices[i].trace = matrices[i].elements[0][0] + matrices[i].elements[1][1] + matrices[i].elements[2][2];
    }

    // Perform mathematical operations on complex numbers
    for (int i = 0; i < 4; ++i)
    {
        complexNumbers[i].magnitude = sqrt(complexNumbers[i].real * complexNumbers[i].real +
                                           complexNumbers[i].imaginary * complexNumbers[i].imaginary);

        complexNumbers[i].phase = atan2(complexNumbers[i].imaginary, complexNumbers[i].real);
    }

    // Display results
    printf("Vectors:\n");
    for (int i = 0; i < 3; ++i)
    {
        printf("Vector %d: Magnitude=%.2f, Direction=%.2f radians, Description=%s\n", i + 1, vectors[i].magnitude,
               vectors[i].direction, vectors[i].description);
    }

    printf("\nMatrices:\n");
    for (int i = 0; i < 2; ++i)
    {
        printf("Matrix %d: Determinant=%.2f, Trace=%.2f\n", i + 1, matrices[i].determinant, matrices[i].trace);
    }

    printf("\nComplex Numbers:\n");
    for (int i = 0; i < 4; ++i)
    {
        printf("Complex Number %d: Magnitude=%.2f, Phase=%.2f radians\n", i + 1, complexNumbers[i].magnitude,
               complexNumbers[i].phase);
    }

    printf("\nParticles:\n");
    for (int i = 0; i < 3; ++i)
    {
        printf("Particle %d: ID=%d, Mass=%.2f, Position=%.2f, %.2f, %.2f, Velocity=%.2f, %.2f, %.2f\n", i + 1,
               particles[i].id, particles[i].mass, particles[i].position.x, particles[i].position.y,
               particles[i].position.z, particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
    }

    printf("\nSensor Data:\n");
    for (int i = 0; i < 5; ++i)
    {
        printf("Sensor %d: ID=%d, Value=%.2f, Timestamp=%s\n", i + 1, sensorData[i].sensorId, sensorData[i].value,
               sensorData[i].timestamp);
    }

    return 0;
}
