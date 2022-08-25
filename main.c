#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <locale.h>
#include <time.h>
#include <pthread.h>

#include "board.h"
#include "bitboard.h"
#include "move.h"
#include "constants.h"
#include "net.h"
#include "search.h"
#include "game.h"
#include "random.h"

#include "pawn.h"
#include "knight.h"
#include "king.h"
#include "bishop.h"
#include "rook.h"
#include "queen.h"

int main()
{
    init_all();
    srand(time(NULL));

    time_t start_s, end_s;
    clock_t start, end;
    time(&start_s);
    start = clock();

    net_weights player;
    read_weights(&player, "trained_seed");
    single_match(&player, NULL, start_position, 4, 1);

    // simulate_challengers(1000, 2, 3, "trained_seed", 3, 0.05);


    time(&end_s);
    end = clock();
    printf("%s\ntime elapsed: %lds\n", horizontal_line, end_s-start_s);
    printf("CPU time: %lus\n", (int)(end-start)/CLOCKS_PER_SEC);

    return 0;
}
