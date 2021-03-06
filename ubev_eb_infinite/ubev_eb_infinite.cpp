#include "../chain_mdp.h"
#include "ubev_eb_infinite.h"

//const int s_params[] = {10, 25, 50, 100};
//const double delta_params[] = {0.1, 0.01};
//const double beta_params[] = {1.0, 0.5, 0.1, 0.05, 0.01};

const int s_params[] = {100};
const double beta_params[] = {1, 0.8, 0.5, 0.2, 0.1};
const double delta_params[] = {0.1};
const double gamma_param = 0.99;

int main() {
    const int num_episodes = 1000000;
    for (auto s: s_params) {
        for (auto delta: delta_params) {
            for (auto beta: beta_params) {
                std::shared_ptr<ChainMDP> chain_mdp(new ChainMDP(s, false, 1.0));
                std::vector<double> rewards;
                UBEV_EB_Infinite ubev_eb(s, 2, s, delta, beta, gamma_param, chain_mdp);
                ubev_eb.train(num_episodes, rewards);
                std::string filename = "ubev_eb_inf_" + std::to_string(s) + "-" + std::to_string(beta) + ".txt";
                printf("%s\n", filename.c_str());
                FILE *f = fopen(filename.c_str(), "w");
                fprintf(f, "%d\n", s);
                fprintf(f, "%lf\n", beta);

                for (int i = 0; i < num_episodes; i++) {
                    fprintf(f, "%.2lf\n", rewards[i]);
                }
                fclose(f);
            }
        }
    }
    return 0;
}


