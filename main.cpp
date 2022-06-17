#include <iostream>
#include "bitboard.hpp"
#include "sliders.hpp"
#include "leapers.hpp"

// init leaper pieces attacks
void init_leapers_attacks()
{
    // loop over 64 board squres
    for (int square = 0; square < 64; square++)
    {
        // inititialize pawn attacks
        pawn_attacks[white][square] = mask_pawn_attacks(white, square);
        pawn_attacks[black][square] = mask_pawn_attacks(black, square);

        // init knight attacks
        knight_attacks[square] = mask_knight_attacks(square);

        // init king attacks
        king_attacks[square] = mask_king_attacks(square);
    }

}

// init slider piece's attack tables
void init_sliders_attacks(int bishop)
{
    //loop over 64 board squares
    for (int square = 0; square < 64; square++)
    {
        //initialize bishop and rook masks
        bishop_masks[square] = mask_bishop_attacks(square);
        rook_masks[square] = mask_rook_attacks(square);

        //init current mask
        U64 attack_mask = bishop ? bishop_masks[square] : rook_masks[square];

        // init relevant occupancy bit count
        int relevant_bits_count = count_bits(attack_mask);

        // init occupancy indicies
        int occupancy_indicies = (1 << relevant_bits_count);

        // loop over occupancy indicies
        for (int index = 0; index < occupancy_indicies; index++)
        {
            // bishop
            if (bishop)
            {
                // init current occupancy variation
                U64 occupancy = set_occupancy(index, relevant_bits_count, attack_mask);

                // init matic index
                int magic_index = (occupancy * bishop_magic_numbers[square]) >>  (64 - bishop_relevant_bits[square]);

                // init bishop attacks
                bishop_attacks[square][magic_index] = generate_bishop_attacks(square, occupancy);
            }

            else //rook
            {
                // init current occupancy variation
                U64 occupancy = set_occupancy(index, relevant_bits_count, attack_mask);

                // init matic index
                int magic_index = (occupancy * rook_magic_numbers[square]) >>  (64 - rook_relevant_bits[square]);

                // init rook attacks
                rook_attacks[square][magic_index] = generate_rook_attacks(square, occupancy);
            }

        }
    }
}

void init_all()
{
    // initiliaze leaper pieces attacks
    init_leapers_attacks();

    //init slider pieces attacks
    // init_sliders_attacks(bishop);
    // init_sliders_attacks(rook);
    init_bishop_attacks();
    init_rook_attacks();

}

int main()
{
    init_all();
    
    set_bit(bitboards[P], e2);

    print_bitboard(bitboards[P]);

    std::cout << "piece: " << ascii_pieces[P] << "\n";
    std::cout << ascii_pieces[char_pieces[K]];
    

    return 0;
}