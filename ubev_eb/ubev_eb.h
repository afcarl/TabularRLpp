#ifndef UBEV_EB_UBEV_EB_H
#define UBEV_EB_UBEV_EB_H

#define nid(s, a, t) (s * (a_ * h_) + a * h_ + t)
#define qid(s, a, t) (s * (a_ * (h_ + 1)) + a * (h_ + 1) + t)
#define mid(ns, s, a, t) (ns * (s_ * a_ * h_) + s * (a_ * h_) + a * h_ + t)
#define piid(s, t) (s * h_ + t)

#include "../env.h"
#include <limits>
#include <memory>
#include <vector>
#include <cmath>
#include <limits>

const double inf = std::numeric_limits<double>::infinity();

double llnp(double x) {
    return std::log(std::log(std::max(std::exp(1.0), x)));
}

class UBEV_EB {
public:
    UBEV_EB(int s, int a, int h, double delta, double beta, std::shared_ptr<TabularEnv> env);
    double execute();
    double confidence_bounds(int s, int a, int t);
    void update_policy();
    void train(int, std::vector<double> &);
private:
    int s_, a_, h_;
    double beta_, delta_;
    std::shared_ptr<TabularEnv> env_;
    std::vector<double> q_, p_;
    std::vector<double> n_, m_, l_;
    std::vector<int> pi_;
};

UBEV_EB::UBEV_EB(int s, int a, int h, double delta, double beta, std::shared_ptr<TabularEnv> env):
s_(s), a_(a), h_(h), beta_(beta), delta_(delta), env_(env),
q_((unsigned)(s * a * (h+1)), 0), p_((unsigned)(s * a * (h+1)), 0), pi_((unsigned)(s * h), 0),
n_((unsigned)(s * a * h), 0), m_((unsigned)(s * s * a * h), 0), l_((unsigned)(s * a * h), 0) {

}

double UBEV_EB::execute() {
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

double UBEV_EB::confidence_bounds(int s, int a, int t) {
    if (n_[nid(s, a, t)] > 0.0) {
        return beta_ * std::sqrt(2 * llnp(nid(s, a, t)) + std::log(18 * s_ * a_ * h_ / delta_))
               / std::sqrt(n_[nid(s, a, t)]);
    } else {
        return inf;
    }
}

void UBEV_EB::update_policy() {
    for (int i = 0; i < q_.size(); i++) {
        p_[i] = q_[i];
    }

    for (int t = h_ - 1; t >= 0; t--) {
        for (int s = 0; s < s_; s++) {
            for (int a = 0; a < a_; a++) {
                if (n_[nid(s, a, t)] == 0) {
                    q_[qid(s, a, t)] = inf;
                    continue;
                }
                q_[qid(s, a, t)] = l_[nid(s, a, t)] / n_[nid(s, a, t)] + confidence_bounds(s, a, t);
                for (int ns = 0; ns < s_; ns++) {
                    double T = m_[mid(ns, s, a, t)] / n_[nid(s, a, t)];
                    if (T <= 0.0) {
                        continue;
                    }
                    double qmax = -inf;
                    for (int i = 0; i < a_; i++) {
                        if (p_[qid(ns, i, t+1)] > qmax) {
                            qmax = p_[qid(ns, i, t+1)];
                        }
                    }
                    q_[qid(s, a, t)] += T * qmax;
                }
            }
        }
    }

    for (int s = 0; s < s_; s++) {
        for (int t = 0; t < h_; t++) {
            int a = 0; double qmax = q_[qid(s, a, t)];
            for (int i = 1; i < a_; i++) {
                if (q_[qid(s, i, t)] > qmax) {
                    qmax = q_[qid(s, i, t)];
                    a = i;
                }
            }
            pi_[piid(s, t)] = a;
        }
    }
}

void UBEV_EB::train(int num_episodes, std::vector<double> &rewards) {
    double alpha = 0.005;
    double cul_reward = 0.0;
    rewards.assign((unsigned)num_episodes, 0);
    for (int k = 0; k < num_episodes; k++) {
        rewards[k] = execute();
        update_policy();
        cul_reward = alpha * rewards[k] + (1 - alpha) * cul_reward;
        if (k % 20000 == 0) {
            //std::cout << cul_reward << std::endl;
            printf("%.2f: %.4lf\n", (float)k / num_episodes, cul_reward);
            fflush(stdout);
        }
    }
}

#undef nid
#undef qid
#undef piid
#undef mid
#endif //UBEV_EB_UBEV_EB_H
