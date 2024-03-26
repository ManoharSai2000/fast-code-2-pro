//
// File:        inference.c
// Description: Implementation of inference function
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "alexnet.h"
#include "data.h"

void alexnet_inference(alexnet *net, char *filename)
{
    image img;
    FILE *fp = fopen("./images.list", "r");
    make_image(&img, FEATURE0_L, FEATURE0_L, IN_CHANNELS);
    img = load_image(filename, FEATURE0_L, FEATURE0_L, IN_CHANNELS, 0);
    // get_next_batch(net->batchsize, net->input, 2,
    //    net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units, fp);
    // printf("forward1\n");
    net->input = img.data;
    // printf("forward\n");
    forward_alexnet(net);
    printf("output\n");
    int pred = argmax(net->output, OUT_LAYER);
    printf("prediction: %d", pred);
    free_image(&img);
}
