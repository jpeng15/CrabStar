#ifndef BISHOP_H
#define BISHOP_H

#include "move.h"
#include "bitboard.h"

// bishop attack masks
static U64 bishop_masks[64];

// bishop attacks table [square][occupancies]
static U64 bishop_attacks[64][512];

/* ------------------------------- functions ------------------------------- */
// get squares that could be obstacles in bishop's path
U64 mask_bishop_attacks(int square);

// generate bishop attacks on the fly
U64 generate_bishop_attacks(int square, U64 block);

// initialize bishop attack table
void init_bishop_attacks();

// get bishop attacks using magic bitboard
U64 get_bishop_attacks(int square, U64 occupancy);

// generate bishop moves
void generate_bishop_moves(move_list *moves, int side, U64 *bitboards, U64 *occupancies);

/* ------------------------------- constants ------------------------------- */
// relevancy occupancy bit count for every square on board
static const int bishop_relevant_bits[64] = {
 6, 5, 5, 5, 5, 5, 5, 6,
 5, 5, 5, 5, 5, 5, 5, 5,
 5, 5, 7, 7, 7, 7, 5, 5,
 5, 5, 7, 9, 9, 7, 5, 5,
 5, 5, 7, 9, 9, 7, 5, 5,
 5, 5, 7, 7, 7, 7, 5, 5,
 5, 5, 5, 5, 5, 5, 5, 5,
 6, 5, 5, 5, 5, 5, 5, 6
}; 

static const U64 bishop_magic_numbers[64] = {
    0x40040844404084ULL,
    0x2004208a004208ULL,
    0x10190041080202ULL,
    0x108060845042010ULL,
    0x581104180800210ULL,
    0x2112080446200010ULL,
    0x1080820820060210ULL,
    0x3c0808410220200ULL,
    0x4050404440404ULL,
    0x21001420088ULL,
    0x24d0080801082102ULL,
    0x1020a0a020400ULL,
    0x40308200402ULL,
    0x4011002100800ULL,
    0x401484104104005ULL,
    0x801010402020200ULL,
    0x400210c3880100ULL,
    0x404022024108200ULL,
    0x810018200204102ULL,
    0x4002801a02003ULL,
    0x85040820080400ULL,
    0x810102c808880400ULL,
    0xe900410884800ULL,
    0x8002020480840102ULL,
    0x220200865090201ULL,
    0x2010100a02021202ULL,
    0x152048408022401ULL,
    0x20080002081110ULL,
    0x4001001021004000ULL,
    0x800040400a011002ULL,
    0xe4004081011002ULL,
    0x1c004001012080ULL,
    0x8004200962a00220ULL,
    0x8422100208500202ULL,
    0x2000402200300c08ULL,
    0x8646020080080080ULL,
    0x80020a0200100808ULL,
    0x2010004880111000ULL,
    0x623000a080011400ULL,
    0x42008c0340209202ULL,
    0x209188240001000ULL,
    0x400408a884001800ULL,
    0x110400a6080400ULL,
    0x1840060a44020800ULL,
    0x90080104000041ULL,
    0x201011000808101ULL,
    0x1a2208080504f080ULL,
    0x8012020600211212ULL,
    0x500861011240000ULL,
    0x180806108200800ULL,
    0x4000020e01040044ULL,
    0x300000261044000aULL,
    0x802241102020002ULL,
    0x20906061210001ULL,
    0x5a84841004010310ULL,
    0x4010801011c04ULL,
    0xa010109502200ULL,
    0x4a02012000ULL,
    0x500201010098b028ULL,
    0x8040002811040900ULL,
    0x28000010020204ULL,
    0x6000020202d0240ULL,
    0x8918844842082200ULL,
    0x4010011029020020ULL
};

#endif