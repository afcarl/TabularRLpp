//
// Created by Jiaming Song on 4/8/17.
//

#ifndef UBEV_EB_MBIE_EB_INFINITE_H
#define UBEV_EB_MBIE_EB_INFINITE_H

#define nid(s, a) (s * a_ + a)
#define qid(s, a) (s * a_ + a)
#define mid(ns, s, a) (ns * (s_ * a_) + s * a_ + a)
#define piid(s) (s)

#include "../env.h"
#include <limits>
#include <memory>
#include <vector>
#include <cmath>
#include <limits>

const double inf = 100000.0;//std::numeric_limits<double>::infinity();

class MBIE_EB_Infinite {
public:
    MBIE_EB_Infinite(int s, int a, int h, double beta, double gamma, std::shared_ptr<TabularEnv> env);
    double execute();
    double confidence_bounds(int s, int a);
    void update_policy();
    void train(int, std::vector<double> &);
private:
    int s_, a_, h_;
    double beta_, gamma_;
    std::shared_ptr<TabularEnv> env_;
    std::vector<double> q_, p_;
    std::vector<double> n_, m_, l_;
    std::vector<int> pi_;
};

MBIE_EB_Infinite::MBIE_EB_Infinite(int s, int a, int h, double beta, double gamma, std::shared_ptr<TabularEnv> env):
s_(s), a_(a), h_(h), beta_(beta), gamma_(gamma), env_(env),
q_((unsigned)(s * a), 0), p_((unsigned)(s * a), 0), pi_((unsigned)(s), 0),
n_((unsigned)(s * a), 0), m_((unsigned)(s * s * a), 0), l_((unsigned)(s * a), 0) {

}

double MBIE_EB_Infinite::execute() {
    int s = env_->state();
    double cul_reward = 0.0;
    for (int t = 0; t < h_; t++) {
        int a = pi_[piid(s)];
        auto tuple = env_->step(a);
        int ns = std::get<0>(tuple);
        double reward = std::get<1>(tuple);
        cul_reward += reward;
        n_[nid(s, a)] += 1;
        l_[nid(s, a)] += reward;
        m_[mid(ns, s, a)] += 1;
        s = ns;
    }
    return cul_reward;
}

double MBIE_EB_Infinite::confidence_bounds(int s, int a) {
    if (n_[nid(s, a)] > 0.0) {
        return beta_ / std::sqrt(n_[nid(s, a)]);
    } else {
        return inf;
    }
}

void MBIE_EB_Infinite::update_policy() {
    for (int i = 0; i < q_.size(); i++) {
        p_[i] = q_[i];
    }

    for (int s = 0; s < s_; s++) {
        for (int a = 0; a < a_; a++) {
            if (n_[nid(s, a)] == 0) {
                q_[qid(s, a)] = inf;
                continue;
            }
            q_[qid(s, a)] = l_[nid(s, a)] / n_[nid(s, a)] + confidence_bounds(s, a);
            for (int ns = 0; ns < s_; ns++) {
                double T = m_[mid(ns, s, a)] / n_[nid(s, a)];
                if (T <= 0.0) {
                    continue;
                }
                double qmax = -inf;
                for (int i = 0; i < a_; i++) {
                    if (p_[qid(ns, i)] > qmax) {
                        qmax = p_[qid(ns, i)];
                    }
                }
                q_[qid(s, a)] += gamma_ * T * qmax;
            }
            //if (!isinf(p_[qid(s, a)])) {
            //    q_[qid(s, a)] = 0.05 * q_[qid(s, a)] + p_[qid(s, a)];
            //}
        }
    }


    for (int s = 0; s < s_; s++) {
        int a = 0; double qmax = q_[qid(s, a)];
        for (int i = 1; i < a_; i++) {
            if (q_[qid(s, i)] > qmax) {
                qmax = q_[qid(s, i)];
                a = i;
            }
        }
        pi_[piid(s)] = a;
    }
}

void MBIE_EB_Infinite::train(int num_episodes, std::vector<double> &rewards) {
    double alpha = 0.005;
    double cul_reward = 0.0;
    rewards.assign((unsigned)num_episodes, 0);
    env_->reset();
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
    for (int s = 0; s < s_; s++) {
        printf("%d ", pi_[s]);
    }
    printf("\n");
}

#undef nid
#undef qid
#undef piid
#undef mid

#endif //UBEV_EB_MBIE_EB_INFINITE_H
