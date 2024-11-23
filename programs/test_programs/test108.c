#include <stdio.h>
#include <string.h>

#define MAX_ITEMS 10
#define MAX_SUPPLIERS 5
#define MAX_ORDERS 15
#define MAX_CUSTOMERS 5

// Struct Definitions
typedef struct {
    int supplierID;
    char name[50];
    char phone[15];
} Supplier;

typedef struct {
    int itemID;
    char name[30];
    int quantity;
    double price;
    int supplierID; // Linked to a supplier
} Item;

typedef struct {
    int customerID;
    char name[50];
    char address[100];
    char phone[15];
} Customer;

typedef struct {
    int orderID;
    int customerID;  // Linked to a customer
    int itemID;      // Linked to an item
    int quantity;
    char date[11];   // YYYY-MM-DD
} Order;

// Function Prototypes
void addSupplier(Supplier suppliers[], int *supplierCount, int id, const char *name, const char *phone);
void addItem(Item items[], int *itemCount, int id, const char *name, int quantity, double price, int supplierID);
void addCustomer(Customer customers[], int *customerCount, int id, const char *name, const char *address, const char *phone);
void createOrder(Order orders[], int *orderCount, Item items[], int itemCount, int id, int customerID, int itemID, int quantity, const char *date);
void displaySuppliers(Supplier suppliers[], int supplierCount);
void displayItems(Item items[], int itemCount);
void displayCustomers(Customer customers[], int customerCount);
void displayOrders(Order orders[], int orderCount);

int main() {
    Supplier suppliers[MAX_SUPPLIERS];
    Item items[MAX_ITEMS];
    Customer customers[MAX_CUSTOMERS];
    Order orders[MAX_ORDERS];

    int supplierCount = 0, itemCount = 0, customerCount = 0, orderCount = 0;

    // Initialize Suppliers
    for (int i = 0; i < 2; i++) {
        char name[50];
        char phone[15];
        sprintf(name, "Supplier %c", 'A' + i);
        sprintf(phone, "123-456-78%02d", i);
        addSupplier(suppliers, &supplierCount, i + 1, name, phone);
    }

    // Initialize Items
    for (int i = 0; i < 2; i++) {
        char name[30];
        sprintf(name, "Item %c", 'X' + i);
        addItem(items, &itemCount, 101 + i, name, (i + 1) * 100, (i + 1) * 20.0, i + 1);
    }

    // Initialize Customers
    for (int i = 0; i < 2; i++) {
        char name[50];
        char address[100];
        char phone[15];
        sprintf(name, "Customer %c", 'A' + i);
        sprintf(address, "%d Main Street", i + 1);
        sprintf(phone, "555-12%02d", i + 34);
        addCustomer(customers, &customerCount, i + 1, name, address, phone);
    }

    // Create Orders
    for (int i = 0; i < 2; i++) {
        char date[11];
        sprintf(date, "2024-11-%02d", 22 + i);
        createOrder(orders, &orderCount, items, itemCount, i + 1, i + 1, 101 + i, (i + 1) * 10, date);
    }

    // Display All Entities
    displaySuppliers(suppliers, supplierCount);
    displayItems(items, itemCount);
    displayCustomers(customers, customerCount);
    displayOrders(orders, orderCount);

    return 0;
}

// Function Definitions

void addSupplier(Supplier suppliers[], int *supplierCount, int id, const char *name, const char *phone) {
    if (*supplierCount < MAX_SUPPLIERS) {
        suppliers[*supplierCount].supplierID = id;
        strcpy(suppliers[*supplierCount].name, name);
        strcpy(suppliers[*supplierCount].phone, phone);
        (*supplierCount)++;
    } else {
        printf("Maximum number of suppliers reached.\n");
    }
}

void addItem(Item items[], int *itemCount, int id, const char *name, int quantity, double price, int supplierID) {
    if (*itemCount < MAX_ITEMS) {
        items[*itemCount].itemID = id;
        strcpy(items[*itemCount].name, name);
        items[*itemCount].quantity = quantity;
        items[*itemCount].price = price;
        items[*itemCount].supplierID = supplierID;
        (*itemCount)++;
    } else {
        printf("Maximum number of items reached.\n");
    }
}

void addCustomer(Customer customers[], int *customerCount, int id, const char *name, const char *address, const char *phone) {
    if (*customerCount < MAX_CUSTOMERS) {
        customers[*customerCount].customerID = id;
        strcpy(customers[*customerCount].name, name);
        strcpy(customers[*customerCount].address, address);
        strcpy(customers[*customerCount].phone, phone);
        (*customerCount)++;
    } else {
        printf("Maximum number of customers reached.\n");
    }
}

void createOrder(Order orders[], int *orderCount, Item items[], int itemCount, int id, int customerID, int itemID, int quantity, const char *date) {
    if (*orderCount < MAX_ORDERS) {
        int itemIndex = -1;
        for (int i = 0; i < itemCount; i++) {
            if (items[i].itemID == itemID) {
                itemIndex = i;
                break;
            }
        }
        if (itemIndex == -1) {
            printf("Item ID %d not found.\n", itemID);
            return;
        }

        if (items[itemIndex].quantity >= quantity) {
            items[itemIndex].quantity -= quantity;
            orders[*orderCount].orderID = id;
            orders[*orderCount].customerID = customerID;
            orders[*orderCount].itemID = itemID;
            orders[*orderCount].quantity = quantity;
            strcpy(orders[*orderCount].date, date);
            (*orderCount)++;
        } else {
            printf("Insufficient stock for item ID %d\n", itemID);
        }
    } else {
        printf("Maximum number of orders reached.\n");
    }
}

void displaySuppliers(Supplier suppliers[], int supplierCount) {
    printf("\nSuppliers:\n");
    for (int i = 0; i < supplierCount; i++) {
        printf("ID: %d, Name: %s, Phone: %s\n", suppliers[i].supplierID, suppliers[i].name, suppliers[i].phone);
    }
}

void displayItems(Item items[], int itemCount) {
    printf("\nItems:\n");
    for (int i = 0; i < itemCount; i++) {
        printf("ID: %d, Name: %s, Quantity: %d, Price: %.2f, SupplierID: %d\n", 
            items[i].itemID, items[i].name, items[i].quantity, items[i].price, items[i].supplierID);
    }
}

void displayCustomers(Customer customers[], int customerCount) {
    printf("\nCustomers:\n");
    for (int i = 0; i < customerCount; i++) {
        printf("ID: %d, Name: %s, Address: %s, Phone: %s\n", 
            customers[i].customerID, customers[i].name, customers[i].address, customers[i].phone);
    }
}

void displayOrders(Order orders[], int orderCount) {
    printf("\nOrders:\n");
    for (int i = 0; i < orderCount; i++) {
        printf("OrderID: %d, CustomerID: %d, ItemID: %d, Quantity: %d, Date: %s\n", 
            orders[i].orderID, orders[i].customerID, orders[i].itemID, orders[i].quantity, orders[i].date);
    }
}
