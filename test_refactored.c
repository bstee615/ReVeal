int main()
{
    int y = 3;
    int goats = 0;
    goats = 10;

    char * vitiating = "butchers";

    int i = 0;
    while(i < goats)
     {
        y -= i;
        if (y == 5)
            {
            goats += 10;
            }
            else if (y == 3)
            {
            goats += 25;
            }
        i ++;
    }
    return goats * y;
}