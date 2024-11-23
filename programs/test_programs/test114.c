#include <stdio.h>
#include <string.h>

#define MAX_ITEMS 10
#define MAX_NAME_LENGTH 50

// Struct for storing information about an item
typedef struct {
    char name[MAX_NAME_LENGTH];
    int quantity;
    float price;
} Item;

// Function Prototypes
void displayInventory(Item inventory[], int itemCount);
void addItem(Item inventory[], int *itemCount, const char *name, int quantity, float price);
void removeItem(Item inventory[], int *itemCount, const char *name);
float calculateTotalValue(Item inventory[], int itemCount);

int main() {
    Item inventory[MAX_ITEMS]; // Array to store items
    int itemCount = 0;         // Number of items in the inventory

    // Adding some items to the inventory
    addItem(inventory, &itemCount, "Apple", 10, 0.5);
    addItem(inventory, &itemCount, "Banana", 20, 0.3);
    addItem(inventory, &itemCount, "Orange", 15, 0.6);

    // Displaying the inventory
    printf("Current Inventory:\n");
    displayInventory(inventory, itemCount);

    // Removing an item from the inventory
    removeItem(inventory, &itemCount, "Banana");

    // Displaying the updated inventory
    printf("\nUpdated Inventory (after removal):\n");
    displayInventory(inventory, itemCount);

    // Calculating the total value of the inventory
    float totalValue = calculateTotalValue(inventory, itemCount);
    printf("\nTotal Value of Inventory: $%.2f\n", totalValue);

    return 0;
}

// Function to display the inventory
void displayInventory(Item inventory[], int itemCount) {
    printf("%-20s%-10s%-10s\n", "Item Name", "Quantity", "Price");
    for (int i = 0; i < itemCount; i++) {
        printf("%-20s%-10d$%-9.2f\n", inventory[i].name, inventory[i].quantity, inventory[i].price);
    }
}

// Function to add an item to the inventory
void addItem(Item inventory[], int *itemCount, const char *name, int quantity, float price) {
    if (*itemCount < MAX_ITEMS) {
        strcpy(inventory[*itemCount].name, name);
        inventory[*itemCount].quantity = quantity;
        inventory[*itemCount].price = price;
        (*itemCount)++;
    } else {
        printf("Inventory is full! Cannot add more items.\n");
    }
}

// Function to remove an item from the inventory
void removeItem(Item inventory[], int *itemCount, const char *name) {
    int found = 0;
    for (int i = 0; i < *itemCount; i++) {
        if (strcmp(inventory[i].name, name) == 0) {
            for (int j = i; j < *itemCount - 1; j++) {
                inventory[j] = inventory[j + 1];  // Shift items to fill the gap
            }
            (*itemCount)--;  // Decrease the item count
            found = 1;
            printf("Item '%s' removed from inventory.\n", name);
            break;
        }
    }
    if (!found) {
        printf("Item '%s' not found in inventory.\n", name);
    }
}

// Function to calculate the total value of the inventory
float calculateTotalValue(Item inventory[], int itemCount) {
    float totalValue = 0;
    for (int i = 0; i < itemCount; i++) {
        totalValue += inventory[i].quantity * inventory[i].price;
    }
    return totalValue;
}
