#include <stdio.h>
#include <string.h>
#include <locale.h>

#include "board.h"
#include "uci.h"
#include "bitboard.h"
#include "move.h"
#include "constants.h"
#include "perft.h"

#include "pawn.h"
#include "knight.h"
#include "king.h"
#include "bishop.h"
#include "rook.h"
#include "queen.h"

int main()
{
    init_attacks();

    int debug = 1;
    if (debug) {
        parse_fen(cmk_position);
        print_board();
        #ifndef WIN_64
            setlocale(LC_NUMERIC, "");
        #endif
        search_position(3);
        printf("move score P x q: %d\n", mvv_lva[P][q]);
        printf("move score q x P: %d\n", mvv_lva[q][P]);
    } else
        uci_loop();

    return 0;
}