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

    board_state board = {};
    parse_fen(&board, tricky_position);
    search_state search = {};
    search.board = &board;
    search.neg_nodes = 0;
    search.follow_pv = 0;
    search.score_pv = 0;
    memset(search.killer_moves, 0, sizeof(search.killer_moves));
    memset(search.history_moves, 0, sizeof(search.history_moves));
    memset(search.pv_table, 0, sizeof(search.pv_table));
    memset(search.pv_length, 0, sizeof(search.pv_length));

    net_weights weights = {};
    read_weights(&weights, "seed1");

    smp_search(&search, 2, &weights);

    time_t start_s, end_s;
    clock_t start, end;
    time(&start_s);
    start = clock();
}
