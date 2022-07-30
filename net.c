#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "net.h"
#include "bitboard.h"
#include "board.h"
#include "constants.h"
#include "random.h"

// load weights from text file into 1d array
void load_vector(float *weights, const int dim, const char *path)
{
    FILE *fp;
    if ((fp = fopen(path, "r")) == NULL) return;

    // read in lines
    char buf[128];
    for (int idx = 0; idx < dim; ++idx) {
        fgets(buf, sizeof(buf), fp);
        float w = atof(buf);
        weights[idx] = w;
    }
    fclose(fp);
}

// initialize weight vectors
void read_weights(net_weights *weights, const char *dir_path)
{
    char file_path[100];

    sprintf(file_path, "%s%s", dir_path, "/layer1-weights.txt");
    load_vector(weights->weights1, nodes0*nodes1, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer2-weights.txt");
    load_vector(weights->weights2, nodes1*nodes2, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layr3-weights.txt");
    load_vector(weights->weights3, nodes2*nodes3, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer4-weights.txt");
    load_vector(weights->weights4, nodes3*nodes4, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer1-biases.txt");
    load_vector(weights->biases1, nodes1, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer2-biases.txt");
    load_vector(weights->biases2, nodes2, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer3-biases.txt");
    load_vector(weights->biases3, nodes3, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer4-biases.txt");
    load_vector(weights->biases4, nodes4, file_path);
}

// save vector of weights to given file path
void save_vector(const float *weights, const int dim, const char *path)
{
    FILE *fp;
    if ((fp = fopen(path, "w")) == NULL) return;
    for (int i = 0; i < dim; ++i)
        fprintf(fp, "%0.5f\n", weights[i]);
    fclose(fp);
}

// save weights to files in given folder path
void save_weights(const net_weights *weights, const char *dir_path)
{
    // make sure directory exists
    struct stat st;
    if (stat(dir_path, &st) == -1)
        mkdir(dir_path, 0700 );

    // save weights
    char file_path[100];

    sprintf(file_path, "%s%s", dir_path, "/layer1-weights.txt");
    save_vector(weights->weights1, nodes0*nodes1, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer2-weights.txt");
    save_vector(weights->weights2, nodes1*nodes2, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer3-weights.txt");
    save_vector(weights->weights3, nodes2*nodes3, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer4-weights.txt");
    save_vector(weights->weights4, nodes3*nodes4, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer1-biases.txt");
    save_vector(weights->biases1, nodes1, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer2-biases.txt");
    save_vector(weights->biases2, nodes2, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer3-biases.txt");
    save_vector(weights->biases3, nodes3, file_path);
    sprintf(file_path, "%s%s", dir_path, "/layer4-biases.txt");
    save_vector(weights->biases4, nodes4, file_path);
}

// evaluate position using neural network
int net_eval(const board_state *board, const net_weights *weights)
{
    // pass through first layer
    float prods1[nodes1];
    memcpy(prods1, weights->biases1, sizeof(weights->biases1));

    // loop through bitboards
    for (int piece = 0; piece < 12; ++piece) {
        U64 bitboard = board->bitboards[(piece + ((board->side==black)*6)) % 12];
        while (bitboard) {
            int loc = ls1b(bitboard);
            int idx = piece*64 + ((board->side==white) ? loc : 63-loc);
            for (int i = 0; i < nodes1; ++i)
                prods1[i] += weights->weights1[idx*nodes1 + i];
            pop_bit(bitboard, loc);
        }
    }

    // pass through second layer
    float prods2[nodes2];
    memcpy(prods2, weights->biases2, sizeof(weights->biases2));

    for (int i = 0; i < nodes1; ++i) {
        if (prods1[i] <= 0) continue;
        for (int j = 0; j < nodes2; ++j)
            prods2[j] += prods1[i] * weights->weights2[i*nodes2 + j];
    }

    // pass through third layer
    float prods3[nodes3];
    memcpy(prods3, weights->biases3, sizeof(weights->biases3));

    for (int i = 0; i < nodes2; ++i) {
        if (prods2[i] <= 0) continue;
        for (int j = 0; j < nodes2; ++j)
            prods3[j] += prods2[i] * weights->weights3[i*nodes2 + j];
    }

    // pass through final layer
    float final = weights->biases4[0];
    for (int i = 0; i < nodes3; ++i) {
        if (prods3[i] <= 0) continue;
        final += prods3[i] * weights->weights4[i];
    }
    
    return (int)(final * 100);
}

// returns copy of network
net_weights *duplicate_weights(const net_weights *weights)
{
    net_weights *copy = malloc(sizeof(net_weights));
    for (int i = 0; i < nodes0*nodes1; ++i) copy->weights1[i] = weights->weights1[i];
    for (int i = 0; i < nodes1*nodes2; ++i) copy->weights2[i] = weights->weights2[i];
    for (int i = 0; i < nodes2*nodes3; ++i) copy->weights3[i] = weights->weights3[i];
    for (int i = 0; i < nodes3*nodes4; ++i) copy->weights4[i] = weights->weights4[i];
    for (int i = 0; i < nodes1; ++i) copy->biases1[i] = weights->biases1[i];
    for (int i = 0; i < nodes2; ++i) copy->biases2[i] = weights->biases2[i];
    for (int i = 0; i < nodes3; ++i) copy->biases3[i] = weights->biases3[i];
    for (int i = 0; i < nodes4; ++i) copy->biases4[i] = weights->biases4[i];
    return copy;
}

// mutate weights in network
void mutate(net_weights *weights, const int inv_rate, const float std_dev)
{
    for (float *w = weights->weights1; w < weights->weights1 + nodes0*nodes1; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    for (float *w = weights->weights2; w < weights->weights2 + nodes1*nodes2; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    for (float *w = weights->weights3; w < weights->weights3 + nodes2*nodes3; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    for (float *w = weights->weights4; w < weights->weights4 + nodes3*nodes4; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    
    for (float *w = weights->biases1; w < weights->biases1 + nodes1; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    for (float *w = weights->biases2; w < weights->biases1 + nodes2; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    for (float *w = weights->biases3; w < weights->biases1 + nodes3; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
    for (float *w = weights->biases4; w < weights->biases1 + nodes4; ++w)
        if (rand() % inv_rate == 0) *w += gaussian_rand() * std_dev;
}

// mix weights from two input weights 50/50 and mutate weights
net_weights *crossover(const net_weights *weights1, const net_weights *weights2,
                       const int inv_rate, const float std_dev)
{
    net_weights *new_weights = malloc(sizeof(net_weights));

    // cross over weights
    for (int i = 0; i < nodes0*nodes1; ++i)
        new_weights->weights1[i] = ((rand()%2==0) ? weights1->weights1[i] : weights2->weights1[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    for (int i = 0; i < nodes1*nodes2; ++i)
        new_weights->weights2[i] = ((rand()%2==0) ? weights1->weights2[i] : weights2->weights2[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    for (int i = 0; i < nodes2*nodes3; ++i)
        new_weights->weights3[i] = ((rand()%2==0) ? weights1->weights3[i] : weights2->weights3[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    for (int i = 0; i < nodes3*nodes4; ++i)
        new_weights->weights4[i] = ((rand()%2==0) ? weights1->weights4[i] : weights2->weights4[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    
    // cross over biases
    for (int i = 0; i < nodes1; ++i)
        new_weights->biases1[i] = ((rand()%2==0) ? weights1->biases1[i] : weights2->biases1[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    for (int i = 0; i < nodes2; ++i)
        new_weights->biases2[i] = ((rand()%2==0) ? weights1->biases2[i] : weights2->biases2[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    for (int i = 0; i < nodes3; ++i)
        new_weights->biases3[i] = ((rand()%2==0) ? weights1->biases3[i] : weights2->biases3[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);
    for (int i = 0; i < nodes4; ++i)
        new_weights->biases4[i] = ((rand()%2==0) ? weights1->biases4[i] : weights2->biases4[i])
                                    + ((rand()%inv_rate==0) ? (gaussian_rand()*std_dev) : 0);

    return new_weights;
}