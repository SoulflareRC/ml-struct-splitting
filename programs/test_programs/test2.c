#include <stdio.h>
#include <stdlib.h>
struct Element{
	double x, y, z; 
	double multiplier; 
}; 

int main()
{
	
	struct Element elements[2]; 
	elements[1].multiplier=1.5; 
	printf("elements[1].multiplier=%f", elements[1].multiplier);
	// for(int i = 0;i<3;i++) {
	// 	elements[i].x=1;
	// 	printf( "elements[%d].x=%f", i, elements[i].x); 
	// }

	return 0;
}
