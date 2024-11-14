#include <stdio.h>

typedef struct {
    int a;
    int b;
    int c;
    int d;
    int e;
} Struct1;

typedef struct {
    int f;
    int g;
    int h;
    int i;
    int j;
} Struct2;

int main() {
    // Declare struct arrays (static arrays, no malloc)
    Struct1 s1[1000];
    Struct2 s2[1000];

    // Initialize struct arrays with some values
    for (int i = 0; i < 1000; i++) {
        s1[i].a = i;
        s1[i].b = i + 1;
        s1[i].c = i + 2;
        s1[i].d = i + 3;
        s1[i].e = i + 4;
        
        s2[i].f = i + 10;
        s2[i].g = i + 11;
        s2[i].h = i + 12;
        s2[i].i = i + 13;
        s2[i].j = i + 14;
    }

    // Loop 1: Access a, b, and c of Struct1
    long long sum1 = 0;
    for (int i = 0; i < 500; i++) {  // Varying iteration count
        sum1 += s1[i].a + s1[i].b + s1[i].c;
    }
    printf("Loop 1 sum of fields a, b, and c: %lld\n", sum1);

    // Loop 2: Access d, e of Struct1
    long long sum2 = 0;
    for (int i = 500; i < 1000; i++) {  // Varying iteration count
        sum2 += s1[i].d + s1[i].e;
    }
    printf("Loop 2 sum of fields d and e: %lld\n", sum2);

    // Loop 3: Access f, g of Struct2
    long long sum3 = 0;
    for (int i = 0; i < 300; i++) {  // Varying iteration count
        sum3 += s2[i].f + s2[i].g;
    }
    printf("Loop 3 sum of fields f and g: %lld\n", sum3);

    // Loop 4: Access h, i, j of Struct2
    long long sum4 = 0;
    for (int i = 300; i < 800; i++) {  // Varying iteration count
        sum4 += s2[i].h + s2[i].i + s2[i].j;
    }
    printf("Loop 4 sum of fields h, i, and j: %lld\n", sum4);

    // Loop 5: Perform a computation on all fields of both structs
    long long sum5 = 0;
    for (int i = 0; i < 1000; i++) {  // Varying iteration count
        sum5 += s1[i].a + s1[i].b + s1[i].c + s1[i].d + s1[i].e +
                s2[i].f + s2[i].g + s2[i].h + s2[i].i + s2[i].j;
    }
    printf("Loop 5 total sum of all fields: %lld\n", sum5);

    // Loop 6: Access fields across both structs
    long long sum6 = 0;
    for (int i = 0; i < 1000; i++) {  // Varying iteration count
        sum6 += s1[i].a + s1[i].b + s2[i].g + s2[i].h;
    }
    printf("Loop 6 sum of selected fields: %lld\n", sum6);

    return 0;
}
