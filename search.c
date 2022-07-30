#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>

#include "search.h"
#include "board.h"
#include "move.h"
#include "constants.h"
#include "net.h"
#include "hash.h"

// declaring global variables
// int best_move = 0;
// int neg_nodes = 0;
// int killer_moves[2][64];
// int history_moves[12][64];
// int pv_length[64];
// int pv_table[64][64];
// int follow_pv;
// int score_pv;
// int full_depth_moves = 4;
// int reduction_limit = 3;

// get priority of move
int score_move(search_state *search, const int move)
{
    if (search->score_pv){                                  // if PV move scoring is allowed
        if (search->pv_table[0][search->ply] == move) {     // make sure we are dealing with PV move
            search->score_pv = 0;                           // disable score PV flag
            return 20000;                                   // give PV move the highest score to search it first
        } 
    }
    if (get_move_capture(move)) {           // capture move, use mvv_lva table
        board_state *board = search->board;
        int target_piece = 0;
        for (int i = p; i <= k; ++i)
            if (get_bit(board->bitboards[i-6*board->side], get_move_target(move))) {
                target_piece = i-6*board->side;
                break;
            }
        return mvv_lva[get_move_piece(move)][target_piece] + 10000;
    } else {                                // score quiet move
        if (search->killer_moves[0][search->ply] == move) return 9000;
        if (search->killer_moves[1][search->ply] == move) return 8000;
        return search->history_moves[get_move_piece(move)][get_move_target(move)];
    }
    return 0;
}

// sort moves by priority
void sort_moves(search_state *search, move_list *moves)
{
    // make list of current scores
    int scores[moves->count];
    for (int i = 0; i < moves->count; ++i)
        scores[i] = score_move(search, moves->moves[i]);

    // insertion sort in descending order (fast for small arrays)
    for (int i = 0; i < moves->count-1; ++i) {
        // find index of largest element in unsorted part
        int max_idx = i;
        for (int j = i+1; j < moves->count; ++j)
            if (scores[j] > scores[max_idx]) max_idx = j;
        
        // swap scores
        int temp_score = scores[max_idx];
        scores[max_idx] = scores[i];
        scores[i] = temp_score;

        // swap moves
        int temp_move = moves->moves[max_idx];
        moves->moves[max_idx] = moves->moves[i];
        moves->moves[i] = temp_move;
    }
}

// negamax alpha beta search
// alpha: minimum score the maximizing player is assured of
// beta: maximum score the minimizing player is assured of
// goal is to get alpha as high as possible
// if beta <= alpha, then the minimizing player had a better option, so we can prune
int negamax(search_state *search, const int alpha, const int beta, int depth, const net_weights *weights)
{
    int score;
    board_state *board = search->board;
    int hash_flag = hash_flag_alpha;
    
    int pv_node = (beta - alpha > 1);
    if (search->ply && (score = read_shared_entry(board, alpha, beta, depth, search->ply)) != no_hash_entry && !pv_node)
    {
        return score;
    }

    // initialize pv length
    search->pv_length[search->ply] = search->ply;

    if (depth <= 0) {       // base case
        return quiescence(search, alpha, beta, weights);
    }
    
    ++search->neg_nodes;

    if (search->ply >= max_ply)     // too deep, danger of exceeding arrays
        return net_eval(board, weights);

    // if in check, increase search depth
    int in_check = square_attacked(board, ls1b(board->bitboards[(board->side==white) ? K : k]), (board->side)^1);
    if (in_check)
        ++depth;
    
    int legal_count = 0;
    int best_sofar = 0;
    int new_alpha = alpha;
    

    // null move pruning
    if (depth >= 3 && in_check == 0 && search->ply) {
        // copy_board
        board_state board_copy;
        memcpy(&board_copy, board, sizeof(board_state));
        // increment ply
        ++search->ply;
        if (board->enpassant != no_sq) board->hash_key ^= enpassant_keys[board->enpassant];
        // reset enpassant capture square
        board->enpassant = no_sq;
        // switch the side and give an opponent an extra move to make
        board->side ^= 1;
        board->hash_key ^= side_key;
        // search moves with reduced depth to find beta cutoffs
        int score = -negamax(search, -beta, -beta + 1, depth - 1, weights);
        //decrement ply
        --search->ply;
        //restore board state
        memcpy(board, &board_copy, sizeof(board_state));
        // fail-hard beta cutoff
        if (score >= beta)
            //move fails high
            return beta;
    }

    // loop through possible moves and narrow alpha and beta
    move_list moves;
    generate_moves(board, &moves);

    // if we are following PV line
    if (search->follow_pv)
        enable_pv_scoring(search, &moves);
    sort_moves(search, &moves);

    // moves searched in a move list
    int moves_searched = 0;

    for (int i = 0; i < moves.count; ++i) {
        board_state board_copy;
        memcpy(&board_copy, board, sizeof(board_state));
        ++search->ply;

        int move = moves.moves[i];
        if (!make_move(board, move, all_moves)) {      // skip illegal moves
            --search->ply;
            continue;
        }
        ++legal_count;

        // normal alpha beta search
        if (!moves_searched) score = -negamax(search, -beta, -new_alpha, depth - 1, weights); 

        else { // late move reduction (LMR)
            //condition to consider LMR
            if (moves_searched >= search->full_depth_moves && depth >= search->reduction_limit && !in_check 
            && !get_move_capture(moves.moves[i]) && !get_move_promoted(moves.moves[i]))
                //search current move with reduced depth
                score = -negamax(search, -new_alpha - 1, -new_alpha, depth - 2, weights);
            else // hack to ensure that full-depth search is done
                score = new_alpha + 1;
            // principle variation search PVS
            if (score > new_alpha) {
                // search the remaining moves to ensure they are all bad
                score = -negamax(search, -new_alpha -1, -new_alpha, depth - 1, weights);
                // if a better move is found in the search, search again with normal alpha beta score bounds
                if ((score > new_alpha) && (score < beta))
                    score = -negamax(search, -beta, -new_alpha, depth-1, weights);
            }
        }

        memcpy(board, &board_copy, sizeof(board_state));
        --search->ply;
        ++moves_searched;

        // variation is better than current best
        if (score > new_alpha) {
            // switch hash flag to one for storing PV node score
            hash_flag = hash_flag_exact;
            // if move is quiet, add move to history heuristic
            // nodes higher in the tree are valued more
            if (!get_move_capture(move))
                search->history_moves[get_move_piece(move)][get_move_target(move)] += 1 << depth;

            new_alpha = score;
            if (search->ply == 0) best_sofar = move;
            
            // update pv table and length
            search->pv_table[search->ply][search->ply] = move;
            // copy next row
            for (int next_ply = search->ply+1; next_ply < search->pv_length[search->ply+1]; ++next_ply)
                search->pv_table[search->ply][next_ply] = search->pv_table[search->ply+1][next_ply];
            search->pv_length[search->ply] = search->pv_length[search->ply+1];
                    // move failed hard beta cutoff
            if (score >= beta) {
                // store hash entry with score equal to beta
                write_shared_entry(board, beta, depth, hash_flag_beta, search->ply);
                // if move is quiet, store in killer moves cache so it has higher priority in analysis
                if (!get_move_capture(move)) {
                    search->killer_moves[1][search->ply] = search->killer_moves[0][search->ply];
                    search->killer_moves[0][search->ply] = move;
                }
                return beta;
            }
        }
    }

    // if no legal moves are possible, position is either checkmate or stalemate
    if (legal_count == 0) return in_check ? -mate_value + search->ply : 0;

    write_shared_entry(board, new_alpha, depth, hash_flag, search->ply);

    if (new_alpha > alpha)      // improvement was found
        search->best_move = best_sofar;
    
    return new_alpha;
}

// quiescence search, see negamax code for better documentation
int quiescence(search_state *search, const int alpha, const int beta, const net_weights *weights)
{
    ++search->neg_nodes;
    int evaluation = net_eval(search->board, weights);

    if (evaluation >= beta) return beta;                // failed hard beta cutoff
    int new_alpha = alpha;
    if (evaluation > alpha) new_alpha = evaluation;     // found better move

    // DFS over possible captures
    board_state *board = search->board;
    move_list moves;
    generate_moves(board, &moves);
    sort_moves(search, &moves);

    for (int i = 0; i < moves.count; ++i) {
        board_state board_copy;
        memcpy(&board_copy, board, sizeof(board_state));
        ++search->ply;

        // test out new move
        int move = moves.moves[i];
        if (make_move(board, move, captures_only) == 0) {      // skip illegal and quiet moves
            --search->ply;
            continue;
        }
        int score = -quiescence(search, -beta, -new_alpha, weights);     // now from opposite perspective

        --search->ply;
        memcpy(board, &board_copy, sizeof(board_state));
        if (score >= beta) return beta;
        if (score > new_alpha) new_alpha = score;
    }

    return new_alpha;
}

// search for best move and print it
void search_position(search_state *search, board_state *board, const int max_depth, const net_weights *weights)
{
    // initialize search state
    search->board = board;
    search->neg_nodes = 0;
    search->follow_pv = 0;
    search->score_pv = 0;
    memset(search->killer_moves, 0, sizeof(search->killer_moves));
    memset(search->history_moves, 0, sizeof(search->history_moves));
    memset(search->pv_table, 0, sizeof(search->pv_table));
    memset(search->pv_length, 0, sizeof(search->pv_length));

    // define intial alpha and beta bounds
    int alpha = -infinity;
    int beta = infinity;
    // iteratively deepen analysis
    int score = 0;
    for (int depth = 1; depth <= max_depth; ++depth) {
        search->follow_pv = 1;
        // find best move within a given position
        score = negamax(search, alpha, beta, depth, weights);
        // we fell outide the window, so try again with a full-wdth window
        if ((score <= alpha) || (score >= beta)) {
            alpha = -infinity;
            beta = infinity;
            continue;
        }
        alpha = score - 50;
        beta = score + 50;
    }
    // best move
    printf("bestmove ");
    print_move(search->pv_table[0][0]);
    printf("\n");
}

// enable PV move scoring
void enable_pv_scoring(search_state *search, move_list *moves)
{
    //disable following PV
    search->follow_pv = 0;
    // loop over the moves within a move list
    for (int i = 0; i < moves->count; ++i){
        // make sure we hit PV move
        if(search->pv_table[0][search->ply] == moves->moves[i]){
            //enable move scoring
            search->score_pv = 1;

            //enable following PV
            search->follow_pv = 1;
        }
    }
}

// search at low depths and return best move
int quick_search(search_state *search, board_state *board, const int depth, const net_weights *weights)
{
    // initialize search state
    search->board = board;
    search->neg_nodes = 0;
    search->follow_pv = 0;
    search->score_pv = 0;
    memset(search->killer_moves, 0, sizeof(search->killer_moves));
    memset(search->history_moves, 0, sizeof(search->history_moves));
    memset(search->pv_table, 0, sizeof(search->pv_table));
    memset(search->pv_length, 0, sizeof(search->pv_length));

    // find move
    negamax(search, -infinity, infinity, depth, weights);
    return search->pv_table[0][0];
}

void smp_search(board_state *board, const int threads, const int depth, const net_weights *weights)
{
    pthread_t pthreads[threads];
    negamax_params *searches[threads];
    
    for (int i = 1; i <= threads; i++)
    {
        // deep copy board state
        board_state *board_copy = calloc(1, sizeof(board_state));
        memcpy(board_copy, board, sizeof(board_state));

        searches[i-1] = calloc(1, sizeof(negamax_params));
        searches[i-1]->search = calloc(1, sizeof(search_state));
        search_state *search = searches[i-1]->search;
        search->ply = 0;
        search->best_move = 0;
        search->neg_nodes = 0;
        search->killer_moves[2][max_ply];
        search->history_moves[12][64];
        search->pv_length[max_ply];
        search->pv_table[max_ply][max_ply];
        search->follow_pv = 0;
        search->score_pv = 0;
        search->full_depth_moves = 0;
        search->reduction_limit = 0;
        searches[i-1]->search->board = board_copy;
        searches[i-1]->alpha = -infinity;
        searches[i-1]->beta = infinity;
        searches[i-1]->depth = depth + 1 + __builtin_ctz(i);
        searches[i-1]->weights = weights;

        pthread_create(&pthreads[i-1], NULL, lazy_negamax, (void *) &searches[i-1]);
        printf("created pthread searching at depth %d\n", searches[i-1]->depth);
    }
    printf("finished creating threads\n");
    for (int i = 0; i < threads; i++) {
        pthread_join(pthreads[i], NULL);
    }
    printf("joined all threads\n");
    printf("bestmove ");
    print_move(searches[threads-1]->search->pv_table[0][0]);
    printf("\n");

    for (int i = 0; i < threads; i++){
        free(searches[i]->search->board);
        free(searches[i]->search);
        free(searches[i]);
    }

}

void *lazy_negamax(void *params)
{
    printf("creating parameters\n");
    negamax_params *nparams = params;
    search_state *search = nparams->search;
    const int alpha = nparams->alpha;
    const int beta = nparams->beta;
    int depth = nparams->depth;
    const net_weights *weights = nparams->weights;
    printf("starting search at depth %d\n", depth);
    int score = negamax(search, alpha, beta, depth, weights);
    printf("finished search at depth %d\n", depth);
    printf("bestmove ");
    print_move(search->pv_table[0][0]);
    printf("\n");
}