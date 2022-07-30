#ifndef SEARCH_H
#define SEARCH_H

#include "bitboard.h"
#include "move.h"
#include "net.h"
#include "board.h"

#define net_weight 0
#define max_ply 64

typedef struct {
    board_state *board;
    int ply;
    int best_move;
    int neg_nodes;
    int killer_moves[2][max_ply];
    int history_moves[12][64];
    int pv_length[max_ply];
    int pv_table[max_ply][max_ply];
    int follow_pv;
    int score_pv;
    int full_depth_moves;
    int reduction_limit;
} search_state;

typedef struct {
    search_state *search;
    int alpha;
    int beta;
    int depth;
    const net_weights *weights;
} negamax_params;

/* ------------------------------- functions ------------------------------- */
// score move - determines priority of move
int score_move(search_state *search, const int move);

// sort moves in descending order by score
void sort_moves(search_state *search, move_list *moves);

// negamax alpha beta search
int negamax(search_state *search, const int alpha, const int beta, int depth, const net_weights *weights);

// quiescence search: similar to negamax but only examine captures
int quiescence(search_state *search, const int alpha, const int beta, const net_weights *weights);

// search position for best move and print it to UCI interface
void search_position(search_state *search, board_state *board, const int depth, const net_weights *weights);

// search at low depths and return best move
int quick_search(search_state *search, board_state *board, const int depth, const net_weights *weights);

// enable PV mopve scoring
void enable_pv_scoring(search_state *search, move_list *moves);

void smp_search(board_state *board, const int threads, const int depth, const net_weights *weights);

void *lazy_negamax(void *params);

/* ------------------------------- variables ------------------------------- */
// // best move
// extern int best_move;

// // negamax alpha beta nodes
// extern int neg_nodes;

// // killer moves [id][ply]
// // can use different number of killer moves but 2 is a good choice
// // see: https://www.chessprogramming.org/Killer_Heuristic
// extern int killer_moves[2][max_ply];

// // history moves [piece][square]
// // prioritizes moves that cause more cutoffs in search tree
// // see: https://www.chessprogramming.org/History_Heuristic
// extern int history_moves[12][64];

// // PV (principal variation) length [ply]
// extern int pv_length[max_ply];

// // PV table
// // see: https://www.reddit.com/r/chessprogramming/comments/m2m048/how_does_a_triangular_pvtable_work/
// extern int pv_table[max_ply][max_ply];
// /*
//     diagonal structure:
//     m1 m2 m3  ... <- principal variation
//     0  m2 m3  ...
//     0  0  m3  ...
//     .  .  .
// */

// // follow PV flag
// extern int follow_pv;

// // score PV PV flag
// extern int score_pv;

// //late move reduction variables
// extern int full_depth_moves;
// extern int reduction_limit;

#endif