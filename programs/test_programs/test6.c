#include <stdio.h>
#include <math.h>

// Define a struct for matrix operations
struct Element 
{
    int a, b, c, d, e, f, g, h; 
};

struct Result{
    int res1, res2; 
}; 
int calcRes1(struct Element e){
    int res = 0; 
    for(int i = 0;i<100000;i++){
        if(i<30000){
            res += e.a + e.b;  
        } else {
            res -= (e.g + e.h);
        }
    }
    return res; 
}

int calcRes2(struct Element e){
    int res = 1; 
    for(int i = 0;i<150000;i++){
        if(i<60000){
            res += e.c + e.d;  
        } else {
            res %= (e.e + e.f);
            res -= e.a; 
        }
    }
    return res; 
}
struct Element genElement(){
    struct Element e; 
    e.a = 1; 
    e.b = 2; 
    e.c = 5; 
    e.d = 10; 
    e.e = 20; 
    e.f = 50; 
    e.g = 100; 
    e.h = 200; 

    return e; 
}
int main()
{
    struct Element e = genElement(); 
    printf("e.a: %d, e.b: %d, e.c: %d, e.d: %d, e.e: %d, e.f: %d, e.g: %d, e.h: %d\n", e.a, e.b, e.c, e.d, e.e, e.f, e.g, e.h);
    struct Result r; 
    r.res1 = calcRes1(e); 
    r.res2 = calcRes2(e); 
    printf("r.res1: %d r.res2: %d\n", r.res1, r.res2); 

    return 0;
}
