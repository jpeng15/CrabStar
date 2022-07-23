#include "game.h"
#include "search.h"
#include "board.h"
#include "constants.h"
#include "net.h"
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

// returns result of match between player1 (white) and player2 (black)
// small bonus for using fewer nodes
float single_match(const net_weights *player1, const net_weights *player2, const char *start_fen, const int depth,
            const int verbose)
{
    int move_count = 0;
    int w_nodes = 0;
    int b_nodes = 0;
    int winner = 0;

    // initialize position
    board_state board = {};
    parse_fen(&board, start_fen);
    search_state search = {};

    // simulate game
    while (move_count++ < max_moves) {
        // white moves
        int w_move = quick_search(&search, &board, depth, player1);
        make_move(&board, w_move, all_moves);
        w_nodes += search.neg_nodes;

        if (verbose) {
            printf("%d ", move_count);
            print_move(w_move);
            printf(" ");
        }

        if (!has_legal_moves(&board)) {
            if (verbose) printf("\n");
            winner = (square_attacked(&board, ls1b(board.bitboards[k]), white) ? 1 : 0);
            break;
        }

        // black moves
        int b_move = quick_search(&search, &board, depth, player2);
        make_move(&board, b_move, all_moves);
        b_nodes += search.neg_nodes;

        if (verbose) {
            print_move(b_move);
            printf(" %d %d\n", w_nodes, b_nodes);
        }

        if (!has_legal_moves(&board)) {
            winner = (square_attacked(&board, ls1b(board.bitboards[K]), black) ? -1 : 0);
            break;
        }
    }
    // print_board(&board);
    return winner + ((w_nodes < b_nodes) ? node_bonus : -node_bonus);
}

// simulates match in round robin format between players, writing elo results to array
void *threaded_rr(void *params)
{
    rr_params *args = (rr_params *)params;
    args->result = single_match(args->player1, args->player2, args->start_fen, args->depth, 0);
    adjust_elos(args->elo1, args->elo2, args->result);
    return NULL;
}

// simulaters match in single elimination format between players, saves pointer to winner's net_weights
void *threaded_se(void *params)
{
    se_params *args = (se_params *)params;
    args->result = single_match(args->player1, args->player2, args->start_fen, args->depth, 0);
    return NULL;
}

// writes array of elo results from round robin tournament
void round_robin(net_weights **players, const int n_players, const int depth, int verbose)
{
    int n_pairings = n_players/2;
    int elos[n_players];
    memset(elos, 0, n_players * sizeof(int));

    // in each round, idxs1[i] faces idxs2[i]
    int idxs1[n_pairings], idxs2[n_pairings];
    for (int i = 0; i < n_pairings; ++i) {
        idxs1[i] = i;
        idxs2[i] = 2*n_pairings - i - 1;
    }

    // simulate matches
    int n_rounds = 2*n_pairings-1;
    int n_threads = 2*n_pairings;
    pthread_t tid[n_threads];
    rr_params params[n_threads];

    for (int round = 0; round < n_rounds; ++round) {
        for (int i = 0; i < n_pairings; ++i) {
            // first match player1 as white
            int w_idx = i;
            params[w_idx].player1 = players[idxs1[i]];
            params[w_idx].player2 = players[idxs2[i]];
            params[w_idx].player1_num = idxs1[i];
            params[w_idx].player2_num = idxs2[i];
            params[w_idx].start_fen = start_position;
            params[w_idx].depth = depth;
            params[w_idx].elo1 = &elos[idxs1[i]];
            params[w_idx].elo2 = &elos[idxs2[i]];
            pthread_create(&tid[w_idx], NULL, threaded_rr, (void *)&params[w_idx]);

            // second match player1 as black
            int b_idx = w_idx + n_pairings;
            params[b_idx].player1 = players[idxs2[i]];
            params[b_idx].player2 = players[idxs1[i]];
            params[b_idx].player1_num = idxs2[i];
            params[b_idx].player2_num = idxs1[i];
            params[b_idx].start_fen = start_position;
            params[b_idx].depth = depth;
            params[b_idx].elo1 = &elos[idxs2[i]];
            params[b_idx].elo2 = &elos[idxs1[i]];
            pthread_create(&tid[b_idx], NULL, threaded_rr, (void *)&params[b_idx]);
        }

        // join all threads
        for (int i = 0; i < n_pairings; ++i) {
            pthread_join(tid[i], NULL);
            pthread_join(tid[i+n_pairings], NULL);
            if (verbose) printf("%d vs %d: %0.2f\n", idxs1[i], idxs2[i],
                                params[i].result-params[i+n_pairings].result);
        }

        // cycle pairings
        for (int i = 1; i < n_pairings; ++i) idxs1[i] = (idxs1[i]>1) ? idxs1[i]-1 : 2*n_pairings-1;
        for (int i = 0; i < n_pairings; ++i) idxs2[i] = (idxs2[i]>1) ? idxs2[i]-1 : 2*n_pairings-1;

        if (verbose) printf("%s\n", horizontal_line);
    }
    if (verbose) {
        printf("ratings:\n");
        for (int i = 0; i < n_players; ++i) printf("player %d: %d\n", i, elos[i]);
    }
}

// simulates single elimination tournament, returns int containing indices of 1st and 2nd place
int single_elimination(net_weights **players, const int n_players, const int depth, int verbose)
{
    int n_remaining = n_players;
    int players_remaining[n_players*2];         // complete binary tree stored in flat array
    for (int i = 0; i < n_players; ++i) players_remaining[i+n_players] = i;

    // loop through rounds
    while (n_remaining > 1) {
        pthread_t tid[n_remaining];
        se_params params[n_remaining];
        
        // loop through pairings and create matches
        for (int i = 0; i < n_remaining; i += 2) {
            int p1 = players_remaining[n_remaining+i], p2 = players_remaining[n_remaining+i+1];
            params[i].player1 = players[p1];
            params[i].player2 = players[p2];
            params[i].depth = depth;
            params[i].start_fen = start_position;
            pthread_create(&tid[i], NULL, threaded_se, (void *)&params[i]);

            params[i+1].player1 = players[p2];
            params[i+1].player2 = players[p1];
            params[i+1].depth = depth;
            params[i+1].start_fen = start_position;
            pthread_create(&tid[i+1], NULL, threaded_se, (void *)&params[i+1]);
        }

        // wait for results and record winners
        for (int i = 0; i < n_remaining; i += 2) {
            pthread_join(tid[i], NULL);
            pthread_join(tid[i+1], NULL);

            int p1 = players_remaining[n_remaining+i], p2 = players_remaining[n_remaining+i+1];
            float result = params[i].result - params[i+1].result;
            if (verbose) printf("%d vs %d: %0.2f\n", p1, p2, result);
            if (n_remaining > 2)
                players_remaining[(n_remaining+i)/2] = (result >= 0) ? p1 : p2;
            else
                return encode_winners(((result >= 0) ? p1 : p2), ((result >= 0) ? p2 : p1));
        }

        n_remaining >>= 1;
        if (verbose) printf("%s\n", short_line);
    }

    return 0;
}

// adjust elo ratings for two players based on result
void adjust_elos(int *elo1, int *elo2, int result)
{
    float expected = 1.0 / (1.0 + pow(10, (*elo1-*elo2)*0.02));
    float actual = result*0.5 + 0.5;
    *elo1 += (int)(elo_k*(actual-expected));
    *elo2 -= (int)(elo_k*(actual-expected));
}

// simulate multiple generations of evolution
void simulate_generations(const int generations, const int n_players, const int depth,
                          const char *seed_path, const int inv_rate, const float std_dev)
{
    printf("%s\n", horizontal_line);
    net_weights *players[n_players];

    // create seed generation
    players[0] = calloc(1, sizeof(net_weights));
    read_weights(players[0], seed_path);
    for (int i = 1; i < n_players; ++i) {
        players[i] = duplicate_weights(players[0]);
        mutate(players[i], inv_rate/4, 4*std_dev);        // a bit more variety in seed generation
    }

    // loop through generations
    for (int gen = 0; gen <= generations; ++gen) {
        printf("generation %d\n", gen);

        // simulate tournament
        int winners = single_elimination(players, n_players, depth, 1);
        int first = get_first_place(winners);
        int second = get_second_place(winners);
        net_weights *first_net = players[first];
        net_weights *second_net = players[second];
        printf("first: %d, second: %d\n", first, second);

        // save best performing net every 10 generations and at completion
        if (gen == generations || (gen && gen % 10 == 0)) {
            // save weights
            char dir_path[100];
            sprintf(dir_path, "gen%d", gen);
            printf("%s\nsaving to %s\n", horizontal_line, dir_path);
            save_weights(first_net, dir_path);

            // if training completed, free all mallocs
            if (gen == generations) {
                for (int i = 0; i < n_players; ++i)
                    free(players[i]);
                return;
            }
        }

        // keep top 2 performing nets and replace rest
        for (int i = 0; i < n_players; ++i)         // free discarded nets
            if (i != first && i != second) free(players[i]);
        players[0] = first_net;
        players[n_players/2] = second_net;              // opposite bracket of tournament
        for (int i = 1; i < n_players/2; ++i) {
            players[i] = crossover(first_net, second_net, inv_rate, std_dev);
            players[i+n_players/2] = crossover(first_net, second_net, inv_rate, std_dev);
        }

        printf("%s\n", horizontal_line);
    }
}