#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// DATA STRUCTURES
typedef struct {
    int *data;    // Dynamic array to store stack elements
    int capacity; // Total capacity of the stack
    int top;      // Index of the top element
} Stack;

////////////////////////////////////////////////////////////////////////////////
// FUNCTION PROTOTYPES
void create(Stack *stack);
void push(Stack *stack, int x);
int pop(Stack *stack);
int peek(const Stack *stack);
int size(const Stack *stack);
int isEmpty(const Stack *stack);

////////////////////////////////////////////////////////////////////////////////
// MAIN ENTRY POINT
int main(int argc, char const *argv[]) {
    Stack stack;
    int x, y, z;

    create(&stack);
    push(&stack, 4);
    x = pop(&stack);
    // 4. Count: 0. Empty: 1.
    printf("%d.\t\tCount: %d.\tEmpty: %d.\n", x, size(&stack), isEmpty(&stack));

    push(&stack, 1);
    push(&stack, 2);
    push(&stack, 3);
    x = pop(&stack);
    y = pop(&stack);
    // 3, 2. Count: 1. Empty: 0;
    printf("%d, %d.\t\tCount: %d.\tEmpty: %d.\n", x, y, size(&stack), isEmpty(&stack));
    pop(&stack); // Empty the stack.

    push(&stack, 5);
    push(&stack, 6);
    x = peek(&stack);
    push(&stack, 7);
    y = pop(&stack);
    push(&stack, 8);
    z = pop(&stack);
    // 6, 7, 8. Count: 2. Empty: 0.
    printf("%d, %d, %d.\tCount: %d.\tEmpty: %d.\n", x, y, z, size(&stack), isEmpty(&stack));

    free(stack.data); // Cleanup allocated memory for the stack
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION IMPLEMENTATIONS

/**
 * Initialize the stack with a small capacity.
 */
void create(Stack *stack) {
    stack->capacity = 4; // Initial capacity
    stack->data = (int *)malloc(stack->capacity * sizeof(int));
    if (stack->data == NULL) {
        printf("ERROR: Memory allocation failed.\n");
        exit(1);
    }
    stack->top = -1; // Stack is empty
}

/**
 * Push data onto the stack, resizing if necessary.
 */
void push(Stack *stack, int x) {
    if (stack->top + 1 == stack->capacity) {
        // Resize the stack
        stack->capacity *= 2;
        stack->data = (int *)realloc(stack->data, stack->capacity * sizeof(int));
        if (stack->data == NULL) {
            printf("ERROR: Memory allocation failed during resize.\n");
            exit(1);
        }
    }
    stack->data[++stack->top] = x; // Add element to the stack
}

/**
 * Pop data from the stack.
 */
int pop(Stack *stack) {
    if (stack->top == -1) {
        printf("ERROR: Pop from empty stack.\n");
        exit(1);
    }
    return stack->data[stack->top--]; // Return top element and decrement `top`
}

/**
 * Returns the next value to be popped.
 */
int peek(const Stack *stack) {
    if (stack->top != -1)
        return stack->data[stack->top];
    else {
        printf("ERROR: Peeking from empty stack.\n");
        exit(1);
    }
}

/**
 * Returns the size of the stack.
 */
int size(const Stack *stack) {
    return stack->top + 1;
}

/**
 * Returns 1 if stack is empty, 0 otherwise.
 */
int isEmpty(const Stack *stack) {
    return stack->top == -1;
}
