#ifndef TABULARRL_UBEV_H
#define TABULARRL_UBEV_H
#define nid(s, a, t) (s * (a_ * h_) + a * h_ + t)
#define vid(s, a, t) (s * (h_ + 1) + t)
#define mid(ns, s, a, t) (ns * (s_ * a_ * h_) + s * (a_ * h_) + a * h_ + t)
#define piid(s, t) (s * h_ + t)

#include "env.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <limits>


const double inf = std::numeric_limits<double>::infinity();

double llnp(double x) {
    return std::log(std::log(std::max(std::exp(1.0), x)));
}

double vrng(std::vector<double> &v, int start,
            const int it, const int step = 1) {
    double max = v[start];
    double min = v[start];
    for (int i = start + step, j = 1; j < it; i += step, j++) {
        max = std::max(max, v[i]);
        min = std::min(min, v[i]);
    }
    return max - min;
}

double vmax(std::vector<double> &v, int start,
            const int it, const int step=1) {
    double max = v[start];
    for (int i = start + step, j = 1; j < it; i += step, j++) {
        max = std::max(max, v[i]);
    }
    return max;
}

double vdot(std::vector<double> &a, std::vector<double> &b, int sa, int sb,
            const int it, const int ta = 1, const int tb = 1) {
    double v = 0.0;
    for (int j = 0; j < it; j++, sa += ta, sb += tb) {
        v += a[sa] * b[sb];
    }
    return v;
}


class UBEV {
public:
    UBEV(int s, int a, int h, double delta, double beta, std::shared_ptr<TabularEnv> env);
    double execute();
    double confidence_bounds(int s, int a, int t);
    void update_policy();
    void train(int, std::vector<double>&);

private:
    int s_, a_, h_;
    double delta_, beta_;
    std::shared_ptr<TabularEnv> env_;
    std::vector<int> pi_;
    std::vector<double> n_, m_, l_, v_;
};

UBEV::UBEV(int s, int a, int h, double delta, double beta, std::shared_ptr<TabularEnv> env):
s_(s), a_(a), h_(h), delta_(delta), beta_(beta), env_(env),
n_((unsigned)(s * a * h), 0), m_((unsigned)(s * s * a * h), 0), l_((unsigned)(s * a * h), 0),
pi_((unsigned)(s * h), 0), v_((unsigned)(s * (h+1)), 0) {

}

double UBEV::execute() {
    int s = env_->reset();
    double cul_reward = 0.0;
    for (int t = 0; t < h_; t++) {
        int a = pi_[piid(s, t)];
        auto tuple = env_->step(a);
        int ns = std::get<0>(tuple);
        double reward = std::get<1>(tuple);
        cul_reward += reward;
        n_[nid(s, a, t)] += 1;
        l_[nid(s, a, t)] += reward;
        m_[mid(ns, s, a, t)] += 1;
        s = ns;
    }
    return cul_reward;
}

double UBEV::confidence_bounds(int s, int a, int t) {
    if (n_[nid(s, a, t)] == 0) {
        return inf;
    } else {
        double phi = beta_ * std::sqrt((2 * llnp(n_[nid(s, a, t)]) + std::log(18 * s_ * a_ * h_ / delta_)))
                     / std::sqrt(n_[nid(s, a, t)]);
        return phi;
    }
}

void UBEV::update_policy() {
    std::vector<double> ev((unsigned)a_, 0);
    for (int t = h_ - 1; t >= 0; t--) {
        for (int s = 0; s < s_; s++) {
            std::fill(ev.begin(), ev.end(), 0);
            for (int a = 0; a < a_; a++) {
                double phi = confidence_bounds(s, a, t);
                if (std::isinf(phi)) {
                    ev[a] = 1.0 + vmax(v_, t+1, s_, h_+1);
                } else {
                    double eva = std::min(
                            1.0,
                            l_[nid(s, a, t)] / std::max(1.0, n_[nid(s, a, t)]) + phi
                    );
                    double evb = std::min(
                            vmax(v_, t+1, s_, h_+1),
                            phi * vrng(v_, t+1, s_, h_+1) +
                                  vdot(m_, v_, mid(0, s, a, t), t+1, s_, s_ * a_ * h_, h_+1) / std::max(1.0, n_[nid(s, a, t)])
                    );
                    ev[a] = eva + evb;
                }
            }
            int amax = 0;
            double vmax = ev[0];
            for (int i = 1; i < a_; i++) {
                if (ev[i] > vmax) {
                    amax = i, vmax = ev[i];
                }
            }
            pi_[s * h_ + t] = amax;
            v_[s * (h_+1) + t] = vmax;
        }
    }
}

void UBEV::train(int num_episodes, std::vector<double> &rewards) {
    printf("\n");
    rewards.assign((unsigned)num_episodes, 0);
    double alpha = 0.005;
    double cul_reward = 0.0;
    for (int k = 0; k < num_episodes; k++) {
        update_policy();
        rewards[k] = execute();
        cul_reward = alpha * rewards[k] + (1 - alpha) * cul_reward;
        if (k % 20000 == 0) {
            //std::cout << cul_reward << std::endl;
            printf("%.2f: %.4lf\n", (float)k / num_episodes, cul_reward);
            fflush(stdout);
        }
    }
    return;
}

#endif //TABULARRL_UBEV_H
