#include <stdio.h>
#include <stdlib.h>
typedef struct
{
	int a;
	int b;
	int e[100];
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
	int e[100];
	long c;
	char d;
	long f;
	long g;
	long h;
} Hello;

int main()
{
	Test t;
	Hello h;
	t.a = 0;
	t.d = 0;
	t.f = 0;
	h.a = 0;
	h.b = 0;
	for (int i = 0; i < 50; i++)
	{
		t.a = t.a + 10; // => arr_1[i]->a = 10;
		t.d = t.a + 3;	// => arr_2[i]->c = arr_1[i]->a + 3;
		h.f = t.a + t.d;
		h.a = t.a + h.f;
		h.b = h.a + 1;
	}

	printf("%d %d %d %d %d\n", t.a, t.d, t.f, h.a, h.b);
	return 0;
}
