#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define pi 3.141592653589793
#define solar_mass (4 * pi * pi)
#define days_per_year 365.24

struct planet
{
    double x, y, z;
    double vx, vy, vz;
    double mass;
};
#define NBODIES 5

int main()
{
    int n = 10;
    int i, j;
    struct planet bodies[NBODIES];
    bodies[0].x = 0;
    bodies[0].y = 0;
    bodies[0].z = 0;
    bodies[0].vx = 0;
    bodies[0].vy = 0;
    bodies[0].vz = 0;
    bodies[0].mass = solar_mass;

    bodies[1].x = 4.84143144246472090e+00;
    bodies[1].y = -1.16032004402742839e+00;
    bodies[1].z = -1.03622044471123109e-01;
    bodies[1].vx = 1.66007664274403694e-03 * days_per_year;
    bodies[1].vy = 7.69901118419740425e-03 * days_per_year;
    bodies[1].vz = -6.90460016972063023e-05 * days_per_year;
    bodies[1].mass = 9.54791938424326609e-04 * solar_mass;

    bodies[2].x = 8.34336671824457987e+00;
    bodies[2].y = 4.12479856412430479e+00;
    bodies[2].z = -4.03523417114321381e-01;
    bodies[2].vx = -2.76742510726862411e-03 * days_per_year;
    bodies[2].vy = 4.99852801234917238e-03 * days_per_year;
    bodies[2].vz = 2.30417297573763929e-05 * days_per_year;
    bodies[2].mass = 2.85885980666130812e-04 * solar_mass;

    bodies[3].x = 1.28943695621391310e+01;
    bodies[3].y = -1.51111514016986312e+01;
    bodies[3].z = -2.23307578892655734e-01;
    bodies[3].vx = 2.96460137564761618e-03 * days_per_year;
    bodies[3].vy = 2.37847173959480950e-03 * days_per_year;
    bodies[3].vz = -2.96589568540237556e-05 * days_per_year;
    bodies[3].mass = 4.36624404335156298e-05 * solar_mass;

    bodies[4].x = 1.53796971148509165e+01;
    bodies[4].y = -2.59193146099879641e+01;
    bodies[4].z = 1.79258772950371181e-01;
    bodies[4].vx = 2.68067772490389322e-03 * days_per_year;
    bodies[4].vy = 1.62824170038242295e-03 * days_per_year;
    bodies[4].vz = -9.51592254519715870e-05 * days_per_year;
    bodies[4].mass = 5.15138902046611451e-05 * solar_mass;

    double px = 0.0, py = 0.0, pz = 0.0;
    for (i = 0; i < NBODIES; i++)
    {
        px += bodies[i].vx * bodies[i].mass;
        py += bodies[i].vy * bodies[i].mass;
        pz += bodies[i].vz * bodies[i].mass;
    }
    bodies[0].vx = -px / solar_mass;
    bodies[0].vy = -py / solar_mass;
    bodies[0].vz = -pz / solar_mass;

    double e;
    double dt = 0.01;

    e = 0.0;
    for (i = 0; i < NBODIES; i++)
    {
        e += 0.5 * bodies[i].mass * (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy + bodies[i].vz * bodies[i].vz);
        for (j = i + 1; j < NBODIES; j++)
        {
            double dx = bodies[i].x - bodies[j].x;
            double dy = bodies[i].y - bodies[j].y;
            double dz = bodies[i].z - bodies[j].z;
            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            e -= (bodies[i].mass * bodies[j].mass) / distance;
        }
    }

    printf("%.9f\n", e);
    for (i = 1; i <= n; i++)
    {
        int i, j;

        for (i = 0; i < NBODIES; i++)
        {
            for (j = i + 1; j < NBODIES; j++)
            {
                double dx = bodies[i].x - bodies[j].x;
                double dy = bodies[i].y - bodies[j].y;
                double dz = bodies[i].z - bodies[j].z;
                double distanced = dx * dx + dy * dy + dz * dz;
                double distance = sqrt(distanced);
                double mag = dt / (distanced * distance);
                bodies[i].vx -= dx * bodies[j].mass * mag;
                bodies[i].vy -= dy * bodies[j].mass * mag;
                bodies[i].vz -= dz * bodies[j].mass * mag;
                bodies[j].vx += dx * bodies[i].mass * mag;
                bodies[j].vy += dy * bodies[i].mass * mag;
                bodies[j].vz += dz * bodies[i].mass * mag;
            }
        }
        for (i = 0; i < NBODIES; i++)
        {

            bodies[i].x += dt * bodies[i].vx;
            bodies[i].y += dt * bodies[i].vy;
            bodies[i].z += dt * bodies[i].vz;
        }
    }

    e = 0.0;
    for (i = 0; i < NBODIES; i++)
    {
        e += 0.5 * bodies[i].mass * (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy + bodies[i].vz * bodies[i].vz);
        for (j = i + 1; j < NBODIES; j++)
        {
            double dx = bodies[i].x - bodies[j].x;
            double dy = bodies[i].y - bodies[j].y;
            double dz = bodies[i].z - bodies[j].z;
            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            e -= (bodies[i].mass * bodies[j].mass) / distance;
        }
    }
    printf("%.9f\n", e);
    return 0;
}
