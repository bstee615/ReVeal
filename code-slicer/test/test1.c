#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    char *x = argv[1];
    if (atoi(x) > 3) {
        printf("Hello, %s!\n", x);
    }
    else {
        printf("Salutations!\n");
    }
    int i = 0;
    while (i < strlen(x)) {
        i ++;
        printf("Length: %d\n", strlen(x));
    }
    printf("Goodbye, %s!\n", x);
}