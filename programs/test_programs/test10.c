#include <stdio.h>
#include <string.h>

#define MAX_EMPLOYEES 4
#define MAX_DEPARTMENTS 3
#define MAX_NAME_LEN 50

// Define a struct for Employee
typedef struct {
    int id;
    char name[MAX_NAME_LEN];
    float salary;       // Base salary
    int hoursWorked;    // Hours worked in the month
    float totalSalary;  // Calculated total salary with bonuses
} Employee;

// Define a struct for Department
typedef struct {
    char name[MAX_NAME_LEN];
    Employee employees[MAX_EMPLOYEES];
    float totalDepartmentSalary;
} Department;

// Define a struct for Company
typedef struct {
    char name[MAX_NAME_LEN];
    Department departments[MAX_DEPARTMENTS];
} Company;

// Function prototypes
void calculateSalariesAndBonuses(Department* department);
void printCompanyDetails(const Company* company);
float calculateBonus(int hoursWorked);

int main() {
    // Initialize a company
    Company company;
    strcpy(company.name, "Tech Corp");

    // Initialize departments
    strcpy(company.departments[0].name, "Engineering");
    company.departments[0].employees[0] = (Employee){1, "Alice", 5000.0, 160, 0.0};
    company.departments[0].employees[1] = (Employee){2, "Bob", 4500.0, 180, 0.0};
    company.departments[0].employees[2] = (Employee){3, "Charlie", 4800.0, 150, 0.0};
    company.departments[0].employees[3] = (Employee){4, "David", 4700.0, 200, 0.0};

    strcpy(company.departments[1].name, "Marketing");
    company.departments[1].employees[0] = (Employee){5, "Eve", 4000.0, 170, 0.0};
    company.departments[1].employees[1] = (Employee){6, "Frank", 3800.0, 160, 0.0};
    company.departments[1].employees[2] = (Employee){7, "Grace", 4100.0, 175, 0.0};
    company.departments[1].employees[3] = (Employee){8, "Hank", 3900.0, 150, 0.0};

    strcpy(company.departments[2].name, "HR");
    company.departments[2].employees[0] = (Employee){9, "Ivy", 4200.0, 180, 0.0};
    company.departments[2].employees[1] = (Employee){10, "Jack", 3900.0, 160, 0.0};
    company.departments[2].employees[2] = (Employee){11, "Kate", 4000.0, 190, 0.0};
    company.departments[2].employees[3] = (Employee){12, "Leo", 4300.0, 170, 0.0};

    // Calculate salaries and bonuses for each department
    for (int i = 0; i < MAX_DEPARTMENTS; i++) {
        calculateSalariesAndBonuses(&company.departments[i]);
    }

    // Print company details
    printCompanyDetails(&company);

    return 0;
}

// Function to calculate salaries and bonuses for all employees in a department
void calculateSalariesAndBonuses(Department* department) {
    department->totalDepartmentSalary = 0.0;
    for (int i = 0; i < MAX_EMPLOYEES; i++) {
        Employee* emp = &department->employees[i];
        float bonus = calculateBonus(emp->hoursWorked);
        emp->totalSalary = emp->salary + bonus;
        department->totalDepartmentSalary += emp->totalSalary;
    }
}

// Function to calculate bonus based on hours worked
float calculateBonus(int hoursWorked) {
    if (hoursWorked > 180) {
        return 1000.0;
    } else if (hoursWorked >= 160) {
        return 500.0;
    } else {
        return 0.0;
    }
}

// Function to print company details
void printCompanyDetails(const Company* company) {
    printf("Company: %s\n\n", company->name);
    for (int i = 0; i < MAX_DEPARTMENTS; i++) {
        const Department* dept = &company->departments[i];
        printf("Department: %s\n", dept->name);
        printf("Total Department Salary: %.2f\n", dept->totalDepartmentSalary);

        for (int j = 0; j < MAX_EMPLOYEES; j++) {
            const Employee* emp = &dept->employees[j];
            printf("  Employee ID: %d\n", emp->id);
            printf("    Name: %s\n", emp->name);
            printf("    Base Salary: %.2f\n", emp->salary);
            printf("    Hours Worked: %d\n", emp->hoursWorked);
            printf("    Total Salary (with bonus): %.2f\n\n", emp->totalSalary);
        }
    }
}
