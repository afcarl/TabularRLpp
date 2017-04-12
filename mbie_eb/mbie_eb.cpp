#include "../chain_mdp.h"
#include "mbie_eb.h"


int main(int argc, char **argv) {
    if (argc < 3) {
        return -1;
    }
    const int num_episodes = 1000000;
    int s = atoi(argv[1]);
    double beta = atof(argv[2]);
    std::string filename = "mbie_eb_" + std::to_string(s) + "-" + std::to_string(beta) + ".txt";
    FILE *f = fopen(filename.c_str(), "w");
    for (int run = 0; run < 5; run++) {
        std::shared_ptr<ChainMDP> chain_mdp(new ChainMDP(s, true, 5));
        std::vector<double> rewards;
        MBIE_EB mbie_eb(s, 2, s, beta, chain_mdp);
        mbie_eb.train(num_episodes, rewards);
        printf("%s\n", filename.c_str());
        for (int i = 0; i < num_episodes; i++) {
            fprintf(f, "%.2lf ", rewards[i]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

