
#include "base.h"

using namespace std;


double Diffusion (vector<vector<unsigned int>> &edges, vector<double> &weights, vector<unsigned int> &SeedSet, double T, unsigned int theta, unsigned int nNodes, unsigned int NUMTHREAD)
{    
    Base sp(edges, weights, SeedSet, T, theta, nNodes, NUMTHREAD);
    sp.parallel_sample();
    return sp.cover;
}