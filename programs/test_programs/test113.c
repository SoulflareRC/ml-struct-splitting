#include <stdio.h>
#include <stdbool.h>

#define MAZE_SIZE 6
#define PATH -1   // Indicator for a traversable path
#define WALL -2   // Indicator for a wall
#define TARGET -3 // Indicator for the target
#define VISITED -4 // Indicator for visited cells

// Struct Definitions
typedef struct {
    int x, y;         // Position of the agent in the maze
    int stepsTaken;   // Number of steps taken by the agent
} Agent;

// Function Prototypes
void initializeMaze(int maze[MAZE_SIZE][MAZE_SIZE]);
bool moveAgent(int maze[MAZE_SIZE][MAZE_SIZE], Agent *agent);
void printMaze(int maze[MAZE_SIZE][MAZE_SIZE], Agent *agent);

int main() {
    int maze[MAZE_SIZE][MAZE_SIZE];
    Agent agent; 
    agent.x = 0; 
    agent.y = 0; 
    agent.stepsTaken = 0; 

    initializeMaze(maze);

    while (maze[agent.x][agent.y] != TARGET) {
        printf("Step %d:\n", agent.stepsTaken + 1);
        if (!moveAgent(maze, &agent)) {
            printf("Agent is stuck and cannot reach the target.\n");
            return 0;
        }
        printMaze(maze, &agent);
        printf("\n");
    }

    printf("Agent reached the target in %d steps!\n", agent.stepsTaken);
    return 0;
}

// Function Definitions

void initializeMaze(int maze[MAZE_SIZE][MAZE_SIZE]) {
    int tempMaze[MAZE_SIZE][MAZE_SIZE] = {
        {PATH, WALL, PATH, PATH, WALL, TARGET},
        {PATH, WALL, PATH, WALL, WALL, PATH},
        {PATH, PATH, PATH, WALL, PATH, PATH},
        {WALL, WALL, PATH, PATH, PATH, WALL},
        {PATH, PATH, PATH, WALL, PATH, WALL},
        {WALL, WALL, PATH, PATH, PATH, PATH},
    };

    for (int i = 0; i < MAZE_SIZE; i++) {
        for (int j = 0; j < MAZE_SIZE; j++) {
            maze[i][j] = tempMaze[i][j];
        }
    }
}

bool moveAgent(int maze[MAZE_SIZE][MAZE_SIZE], Agent *agent) {
    int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    bool moved = false;

    for (int i = 0; i < 4; i++) {
        int newX = agent->x + directions[i][0];
        int newY = agent->y + directions[i][1];

        if (newX >= 0 && newX < MAZE_SIZE && newY >= 0 && newY < MAZE_SIZE) {
            if (maze[newX][newY] == PATH || maze[newX][newY] == TARGET) {
                agent->x = newX;
                agent->y = newY;
                agent->stepsTaken = agent->stepsTaken+1;
                if (maze[newX][newY] == PATH) {
                    maze[newX][newY] = VISITED;
                }
                moved = true;
                break;
            }
        }
    }

    // If no move is possible, backtrack
    if (!moved) {
        for (int i = 0; i < 4; i++) {
            int backX = agent->x - directions[i][0];
            int backY = agent->y - directions[i][1];

            if (backX >= 0 && backX < MAZE_SIZE && backY >= 0 && backY < MAZE_SIZE) {
                if (maze[backX][backY] == VISITED) {
                    agent->x = backX;
                    agent->y = backY;
                    agent->stepsTaken = agent->stepsTaken+1;
                    return true;
                }
            }
        }
        return false; // Completely stuck
    }

    return true;
}

void printMaze(int maze[MAZE_SIZE][MAZE_SIZE], Agent *agent) {
    for (int i = 0; i < MAZE_SIZE; i++) {
        for (int j = 0; j < MAZE_SIZE; j++) {
            if (i == agent->x && j == agent->y) {
                printf("A "); // Agent's current position
            } else if (maze[i][j] == WALL) {
                printf("# "); // Wall
            } else if (maze[i][j] == PATH) {
                printf(". "); // Path
            } else if (maze[i][j] == TARGET) {
                printf("T "); // Target
            } else if (maze[i][j] == VISITED) {
                printf("v "); // Visited cells
            }
        }
        printf("\n");
    }
}
