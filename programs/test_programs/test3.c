#include <stdio.h>
#include <stdlib.h>
struct Element {
    int a; 
    double b; 
    char c; 
}; 
int main()
{
    int element_num = 30; 
    // struct Element elementsfixed[3]; 
    // elementsfixed[0].a = 1; 
    struct Element single_element; 
    // single_element.b = 2.1; 
    struct Element elements[element_num];
    int element_idx = 1; 
    // printf("Before entering loop\n");
    for(int i = 0;i<element_num;i++){ 
        // printf("Before access %d\n", i); 
        // elements[i].a = 3; 
        // printf("After access %d\n", i); 
        struct Element* ptr = &(elements[i]); 
        ptr->a = 3; 
        int* ptr_a = &(ptr->a); 
        // printf("ptr: %p ptr_a: %p\n", ptr, ptr_a); 
        printf("elements[i].a: %d\n", ptr->a); // this went wrong. 
    }
    return 0;
}
