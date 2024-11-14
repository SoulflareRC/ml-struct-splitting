#include <stdio.h>
#include <math.h>

// Define a struct for a circle with more than 5 fields
struct Circle
{
    double radius;
    double diameter;
    double area;
    double circumference;
    double x_center;
    double y_center;
    // Add more fields as needed
};

int main()
{
    // Define the number of circles in the array
    const int numCircles = 20;

    // Create an array of struct instances
    struct Circle circles[20];

    // Initialize each circle with different radii
    circles[0].radius = 5.0;
    circles[1].radius = 8.0;
    circles[2].radius = 10.0;

    // Loop through the array and calculate properties for each circle
    for (int i = 0; i < numCircles; ++i)
    {
        circles[i].x_center = 0.0; // Assuming the center is at (0, 0) for simplicity
        circles[i].y_center = 0.0;
        circles[i].radius = i; 
    }

    for (int i = 0; i < numCircles; ++i)
    {
        circles[i].diameter = 2.0 * circles[i].radius;
        circles[i].area = 3.14 * circles[i].radius * circles[i].radius;
        circles[i].circumference = 2 * 3.14 * circles[i].radius;
    }

    for (int i = 0; i < numCircles; ++i)
    {
        // Print the properties of each circle
        printf("Circle %d Properties:\n", i + 1);
        printf("Radius: %.2f\n", circles[i].radius);
        printf("Diameter: %.2f\n", circles[i].diameter);
        printf("Area: %.2f\n", circles[i].area);
        printf("Circumference: %.2f\n", circles[i].circumference);
        printf("\n");
    }

    // Add more print statements or calculations as needed

    return 0;
}
