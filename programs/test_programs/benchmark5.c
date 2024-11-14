#include <stdio.h>
#include <stdlib.h>
typedef struct
{
    int a;
    int b;
    long c;
    int d;
    long f;
    long g;
    long h;
} Test;

typedef struct
{
    int a;
    int b;
    long c;
    char d;
    long f;
    long g;
    long h;
} Hello;

int main()
{
    Test t[100];
    Hello h[100];
    for (int i = 0; i < 50; i++)
    {
        t[i].a = 0;
        t[i].d = 0;
        t[i].f = 0;
        h[i].a = 0;
        h[i].b = 0;
    }
    for (int i = 0; i < 50; i++)
    {
        t[i].a = t[i].a + 10; // => arr_1[i]->a = 10;
        t[i].d = t[i].a + 3;  // => arr_2[i]->c = arr_1[i]->a + 3;
        h[i].f = t[i].a + t[i].d;
        h[i].a = t[i].a + h[i].f;
        h[i].b = h[i].a + 1;
    }

    for (int i = 0; i < 50; i++)
    {
        t[i].a = t[i].a + 10; // => arr_1[i]->a = 10;
        t[i].d = t[i].a + 3;  // => arr_2[i]->c = arr_1[i]->a + 3;
        h[i].f = t[i].a + t[i].d;
        h[i].a = t[i].a + h[i].f;
    }

    for (int i = 0; i < 50; i++)
    {
        t[i].h = t[i].d + 10; // => arr_1[i]->a = 10;
    }

    for (int i = 0; i < 50; i++)
    {
        printf("%d %d %d %d %d\n", t[i].a, t[i].d, t[i].f, h[i].a, h[i].b);
    }
    return 0;
}
