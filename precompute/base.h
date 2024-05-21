#ifndef BASE_H
#define BASE_H

#include<iostream>
#include<vector>
#include<string>
#include<thread>
#include <math.h>
#include <utility>
#include <algorithm>
#include <queue>
#include<sys/time.h>

#include<chrono>
#include<random>
#include "cnpy.h"

using namespace std;

typedef unsigned int uint;
typedef pair<double, uint> wn;
typedef pair<uint, double> nw;


uint seed = chrono::system_clock::now().time_since_epoch().count();
default_random_engine generator (seed);
uniform_real_distribution<double> distribution (0.0,1.0);

class Base{
    public:
        uint n;
        uint theta;
        double cover;
        double T;
        uint NUMTHREAD;
        vector<vector<pair<uint, double>>>graph;
        vector<uint> SeedSet;
        

        Base(vector<vector<uint>> &edges, vector<double> &weights, vector<uint> &_SeedSet, double _T, uint _theta, uint nNodes, uint _NUMTHREAD)
        {
            NUMTHREAD = _NUMTHREAD;
            n = nNodes;
            theta = _theta;
            T = _T;
            cover = 0.0;
            graph = vector<vector<nw>>(n, vector<nw>());
            unsigned int cnt=0;
            for(auto &edge:edges)
                graph[edge[0]].push_back(make_pair(edge[1], weights[cnt++]));
            SeedSet.assign(_SeedSet.begin(), _SeedSet.end());
            //uint seed = chrono::system_clock::now().time_since_epoch().count();
            //default_random_engine generator (seed);
            //uniform_real_distribution<double> distribution (0.0,1.0);
        }

        void parallel_sample()
        {
            vector<thread> threads;
            uint gap=theta/NUMTHREAD;
            uint start=0, end=0,ti=0;

            struct timeval t_start, t_end;
            gettimeofday(&t_start, NULL);

            end=0;
            for(ti=1;ti<=theta%NUMTHREAD;ti++){
                start = end;
                end += ceil((double)theta / NUMTHREAD);
                threads.push_back(thread(&Base::sample, this, start, end));
            }
            for(;ti<=NUMTHREAD;ti++){
                start = end;
                end += gap;
                threads.push_back(thread(&Base::sample, this, start, end));
            }
            for (uint t = 0; t < NUMTHREAD; t++)threads[t].join();  
            threads.clear(); 
            gettimeofday(&t_end, NULL);
            double timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
            cout << "Sampling time: " << timeCost << " s" << endl;
        }

        void sample(uint start, uint end)
        {                   
            vector<double>disV=vector<double>(n,T+1.0);            
            for(auto i=start;i<end;i++){
                priority_queue<wn, vector<wn>, greater<wn>> pq;
                fill(disV.begin(), disV.end(), T+1.0);
                for(auto &v: SeedSet){
                    pq.push(make_pair(0.0,v));
                    disV[v] = 0.0;                    
                }
                while(!pq.empty()){    
                    double cur_dis=pq.top().first;        
                    uint u=pq.top().second;
                    pq.pop();
                    if (cur_dis > T) 
                        break;
                    if (cur_dis > disV[u])  // This is necessary since for multiple sources, a neighbors can be added 
                        continue;           // into the priority queue multiple times in one loop with decreasing distance
                    ++cover;                     
                    for(auto &node_wei: graph[u]){
                        uint node = node_wei.first;
                        double weight = node_wei.second;
                        weight = -log(distribution(generator))/weight;
                        double new_dis = cur_dis + weight;
                        if(new_dis < disV[node] && new_dis <=T){
                            disV[node]= new_dis;
                            pq.push(make_pair(new_dis, node));                        
                        }                    
                    }
                }              
            }                            
        }
};

#endif