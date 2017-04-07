#ifndef TABULARRL_ENV_H
#define TABULARRL_ENV_H

#include <tuple>

template <typename obs, typename action>
class Env {
public:
    Env() {}
    virtual obs reset() = 0;
    virtual std::tuple<obs, double, bool> step(action) = 0;
};

typedef Env<int, int> TabularEnv;

#endif //TABULARRL_ENV_H
