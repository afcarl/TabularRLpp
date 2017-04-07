#ifndef TABULARRL_CHAIN_MDP_H
#define TABULARRL_CHAIN_MDP_H

#include "env.h"
#include <vector>
#include <cmath>
#include <tuple>
#include <cassert>
#include <random>
#include <ctime>

class ChainMDP: public TabularEnv {
public:
    ChainMDP(int);
    int reset(void);
    std::tuple<int, double, bool> step(int);

private:
    bool done_;
    int n_, s_, t_;
    const int n_actions_ = 2;
    std::normal_distribution<double> normal_;
    std::default_random_engine generator;
};

ChainMDP::ChainMDP(int n):
        n_(n), s_(0), t_(0),
        done_(false), normal_(0.0, 1.0), generator((unsigned)time(0)) {
}

int ChainMDP::reset() {
    s_ = 0;
    t_ = 0;
    done_ = false;
    return s_;
}

std::tuple<int, double, bool> ChainMDP::step(int action) {
    assert(!done_);
    assert(action < n_actions_);
    assert(action >= 0);
    double reward = normal_(generator);
    if (s_ == n_ - 1) {
        reward += 1.0;
    }
    s_ += action? 1: -1;
    s_ = std::min(std::max(s_, 0), n_ - 1);

    if (t_ == n_ - 1) {
        done_ = true;
    } else {
        t_ += 1;
    }

    return std::make_tuple(s_, reward, done_);
};

#endif //TABULARRL_CHAIN_MDP_H
