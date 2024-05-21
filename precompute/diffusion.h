#ifndef DIFFUSION_H
#define DIFFUSION_H
#include<iostream>
#include<vector>

using namespace std;

double Diffusion (vector<vector<unsigned int>> &edges, vector<double> &weights, vector<unsigned int> &SeedSet, double T, unsigned int theta, unsigned int nNodes, unsigned int NUMTHREAD);

#endif
