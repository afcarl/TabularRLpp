#include "../chain_mdp.h"
#include "ubev_eb.h"


int main(int argc, char **argv) {
    const int num_episodes = 1000000;
    if (argc < 3) {
        return -1;
    }
    int s = atoi(argv[1]);
    double beta = atof(argv[2]);
    double delta = 0.1;
    std::string filename =
            "ubev_eb_" + std::to_string(s) + "-" + std::to_string(delta) + "-" + std::to_string(beta) +
            ".txt";
    FILE *f = fopen(filename.c_str(), "w");
    for (int run = 0; run < 5; run++) {
        std::shared_ptr<ChainMDP> chain_mdp(new ChainMDP(s, true, 5.0));
        std::vector<double> rewards;
        UBEV_EB ubev(s, 2, s, delta, beta, chain_mdp);
        ubev.train(num_episodes, rewards);
        printf("%s\n", filename.c_str());

        for (int i = 0; i < num_episodes; i++) {
            fprintf(f, "%.2lf ", rewards[i]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}