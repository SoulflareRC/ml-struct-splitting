#include <stdio.h>
#include <stdlib.h>
struct Another{
    int i1, i2, i3, i4; 
}; 
struct Element {
    int a; 
    double b; 
    char c; 
    struct Another another; 
}; 
int main()
{
    struct Element element; 
    // struct Another another; 
    // another.i2 = 3; 
    // element.another = &another;
    struct Another* anotherPtr = &(element.another);
    // printf("Another Ptr: %p\n", anotherPtr);  
    int* i2Ptr = &(anotherPtr->i2); 
    // printf("i2 Ptr: %p\n", i2Ptr); 
    (*i2Ptr) = 3; 
    printf("*itPtr: %d\n", *i2Ptr);   
    // element.another.i2 = 3; 
    // printf("element.another.i2：%d", element.another.i2); 
    return 0;
}
