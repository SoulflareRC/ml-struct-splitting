#include <stdio.h>
#include <string.h>

#define MAX_STUDENTS 3
#define MAX_NAME_LEN 50
#define MAX_COURSES 2
#define SUBJECT_COUNT 3

// Define a struct for Student
typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    float marks[SUBJECT_COUNT]; // Marks in 3 subjects
    float average;
    char grade;
} Student;

// Define a struct for Course
typedef struct {
    char courseName[MAX_NAME_LEN];
    Student students[MAX_STUDENTS];
} Course;

// Function prototypes
void calculateAveragesAndGrades(Student students[], int count);
void printCourseDetails(const Course courses[], int courseCount);
void assignGrades(Student* student);

int main() {
    // Predefined courses and students
    Course courses[MAX_COURSES];

    // Assign data for Course 1
    strcpy(courses[0].courseName, "Math 101");
    courses[0].students[0] = (Student){1, "Alice", {85.0, 90.0, 78.0}, 0.0, '\0'};
    courses[0].students[1] = (Student){2, "Bob", {72.0, 88.0, 91.0}, 0.0, '\0'};
    courses[0].students[2] = (Student){3, "Charlie", {80.0, 70.0, 75.0}, 0.0, '\0'};

    // Assign data for Course 2
    strcpy(courses[1].courseName, "Science 102");
    courses[1].students[0] = (Student){4, "Diana", {92.0, 85.0, 88.0}, 0.0, '\0'};
    courses[1].students[1] = (Student){5, "Eve", {78.0, 80.0, 82.0}, 0.0, '\0'};
    courses[1].students[2] = (Student){6, "Frank", {65.0, 70.0, 68.0}, 0.0, '\0'};

    // Process each course
    for (int i = 0; i < MAX_COURSES; i++) {
        calculateAveragesAndGrades(courses[i].students, MAX_STUDENTS);
    }

    // Print details of all courses
    printCourseDetails(courses, MAX_COURSES);

    return 0;
}

// Function to calculate averages and assign grades
void calculateAveragesAndGrades(Student students[], int count) {
    for (int i = 0; i < count; i++) {
        float sum = 0;
        for (int j = 0; j < SUBJECT_COUNT; j++) {
            sum += students[i].marks[j];
        }
        students[i].average = sum / SUBJECT_COUNT;
        assignGrades(&students[i]);
    }
}

// Function to assign grades based on average
void assignGrades(Student* student) {
    if (student->average >= 85.0) {
        student->grade = 'A';
    } else if (student->average >= 70.0) {
        student->grade = 'B';
    } else if (student->average >= 50.0) {
        student->grade = 'C';
    } else {
        student->grade = 'F';
    }
}

// Function to print course details
void printCourseDetails(const Course courses[], int courseCount) {
    for (int i = 0; i < courseCount; i++) {
        printf("Course: %s\n", courses[i].courseName);
        for (int j = 0; j < MAX_STUDENTS; j++) {
            const Student* student = &courses[i].students[j];
            printf("  Student ID: %d\n", student->id);
            printf("    Name: %s\n", student->name);
            printf("    Marks: %.2f, %.2f, %.2f\n",
                   student->marks[0], student->marks[1], student->marks[2]);
            printf("    Average: %.2f\n", student->average);
            printf("    Grade: %c\n\n", student->grade);
        }
    }
}
