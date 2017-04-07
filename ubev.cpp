#include "chain_mdp.h"
#include "ubev.h"

//const int s_params[] = {10, 25, 50, 100};
//const double delta_params[] = {0.1, 0.01};
//const double beta_params[] = {1.0, 0.5, 0.1, 0.05, 0.01};

const int s_params[] = {50};
const double delta_params[] = {0.1};
const double beta_params[] = {0.1};

int main() {
    const int num_episodes = 1000000;
    for (auto s: s_params) {
        for (auto delta: delta_params) {
            for (auto beta: beta_params) {
                std::shared_ptr<ChainMDP> chain_mdp(new ChainMDP(s));
                std::vector<double> rewards;
                UBEV ubev(s, 2, s, delta, beta, chain_mdp);
                ubev.train(num_episodes, rewards);
                std::string filename = "ubev_" + std::to_string(s) + "-" + std::to_string(delta) + "-" + std::to_string(beta) + ".txt";
                printf("%s\n", filename.c_str());
                FILE *f = fopen(filename.c_str(), "w");
                fprintf(f, "%d\n", s);
                fprintf(f, "%lf\n", delta);
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