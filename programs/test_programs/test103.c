#include <stdio.h>
#include <string.h>

#define MAX_NAME_LEN 50
#define MAX_COURSES 3
#define MAX_STUDENTS 3
#define MAX_ENROLLMENTS 5

// Struct for Course
typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    int capacity;
    int enrolledCount;
    int studentIds[MAX_STUDENTS]; // Fixed-size array for enrolled student IDs
} Course;

// Struct for Student
typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    int enrolledCoursesCount;
    int enrolledCourses[MAX_ENROLLMENTS]; // Maximum courses a student can enroll in
} Student;

// Function prototypes
void initializeCourses(Course courses[], int *courseCount);
void initializeStudents(Student students[], int *studentCount);
void enrollStudentInCourse(Course courses[], int courseCount, Student students[], int studentCount, int studentId, int courseId);
void displayEnrollmentStatus(Course courses[], int courseCount, Student students[], int studentCount);

int main() {
    int courseCount, studentCount;

    // Declare arrays for courses and students
    Course courses[MAX_COURSES];
    Student students[MAX_STUDENTS];

    // Initialize courses and students
    initializeCourses(courses, &courseCount);
    initializeStudents(students, &studentCount);

    // Simulate enrollments
    enrollStudentInCourse(courses, courseCount, students, studentCount, 1, 101); // Alice -> Math
    enrollStudentInCourse(courses, courseCount, students, studentCount, 2, 102); // Bob -> Physics
    enrollStudentInCourse(courses, courseCount, students, studentCount, 1, 102); // Alice -> Physics
    enrollStudentInCourse(courses, courseCount, students, studentCount, 3, 103); // Charlie -> Chemistry
    enrollStudentInCourse(courses, courseCount, students, studentCount, 1, 101); // Alice tries Math again (should fail)

    // Display enrollment status
    displayEnrollmentStatus(courses, courseCount, students, studentCount);

    return 0;
}

// Function to initialize courses
void initializeCourses(Course courses[], int *courseCount) {
    *courseCount = MAX_COURSES;

    courses[0].id = 101;
    strcpy(courses[0].name, "Mathematics");
    courses[0].capacity = 2;
    courses[0].enrolledCount = 0;

    courses[1].id = 102;
    strcpy(courses[1].name, "Physics");
    courses[1].capacity = 2;
    courses[1].enrolledCount = 0;

    courses[2].id = 103;
    strcpy(courses[2].name, "Chemistry");
    courses[2].capacity = 1;
    courses[2].enrolledCount = 0;
}

// Function to initialize students
void initializeStudents(Student students[], int *studentCount) {
    *studentCount = MAX_STUDENTS;

    students[0].id = 1;
    strcpy(students[0].name, "Alice");
    students[0].enrolledCoursesCount = 0;

    students[1].id = 2;
    strcpy(students[1].name, "Bob");
    students[1].enrolledCoursesCount = 0;

    students[2].id = 3;
    strcpy(students[2].name, "Charlie");
    students[2].enrolledCoursesCount = 0;
}

// Function to enroll a student in a course
void enrollStudentInCourse(Course courses[], int courseCount, Student students[], int studentCount, int studentId, int courseId) {
    Course *course = NULL;
    Student *student = NULL;

    // Find the course
    for (int i = 0; i < courseCount; i++) {
        if (courses[i].id == courseId) {
            course = &courses[i];
            break;
        }
    }

    // Find the student
    for (int i = 0; i < studentCount; i++) {
        if (students[i].id == studentId) {
            student = &students[i];
            break;
        }
    }

    if (!course || !student) {
        printf("Invalid student or course ID.\n");
        return;
    }

    // Check if the course is full
    if (course->enrolledCount >= course->capacity) {
        printf("%s could not enroll in %s: Course is full.\n", student->name, course->name);
        return;
    }

    // Check if the student is already enrolled
    for (int i = 0; i < student->enrolledCoursesCount; i++) {
        if (student->enrolledCourses[i] == courseId) {
            printf("%s is already enrolled in %s.\n", student->name, course->name);
            return;
        }
    }

    // Enroll the student
    course->studentIds[course->enrolledCount++] = studentId;
    student->enrolledCourses[student->enrolledCoursesCount++] = courseId;
    printf("%s successfully enrolled in %s.\n", student->name, course->name);
}

// Function to display enrollment status
void displayEnrollmentStatus(Course courses[], int courseCount, Student students[], int studentCount) {
    printf("\nEnrollment Status:\n");

    for (int i = 0; i < courseCount; i++) {
        printf("Course: %s (ID: %d)\n", courses[i].name, courses[i].id);
        printf("  Enrolled Students (%d/%d): ", courses[i].enrolledCount, courses[i].capacity);
        for (int j = 0; j < courses[i].enrolledCount; j++) {
            printf("%d ", courses[i].studentIds[j]);
        }
        printf("\n");
    }

    printf("\nStudent Enrollment:\n");
    for (int i = 0; i < studentCount; i++) {
        printf("Student: %s (ID: %d)\n", students[i].name, students[i].id);
        printf("  Enrolled Courses: ");
        for (int j = 0; j < students[i].enrolledCoursesCount; j++) {
            printf("%d ", students[i].enrolledCourses[j]);
        }
        printf("\n");
    }
}
