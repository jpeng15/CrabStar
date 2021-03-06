#ifndef NET_H
#define NET_H

#include "weights.h"

/*
    neural net structure:
    768 input nodes: 12 piece bitboards, all from perspective of player to move
    dense 128 node layer with ReLU activation
    dense 32 node layer with ReLU activation
    dense 32 node layer with ReLU activation
    1 output
*/

#define nodes0 768
#define nodes1 128
#define nodes2 32
#define nodes3 32
#define nodes4 1

// load weights from text file into 1d array
void load_weights(float *weights, const int dim, const char *path);

// initialize weight vectors from directory
void init_weights(const char *dir_path);

// evaluate position using neural network
int net_eval();

// weight vectors
// static float weights1[nodes0 * nodes1];
// static float weights2[nodes1 * nodes2];
// static float weights3[nodes2 * nodes3];
// static float weights4[nodes3 * nodes4];

// // bias vectors
// static float biases1[nodes1];
// static float biases2[nodes2];
// static float biases3[nodes3];
// static float biases4[nodes4];

#endif