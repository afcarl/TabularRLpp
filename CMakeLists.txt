cmake_minimum_required(VERSION 3.7)
project(UBEV)
project(MBIE_EB)
project(UBEV_EB)

set(CMAKE_CXX_STANDARD 11)

set(UBEV_FILES chain_mdp.h ubev.h env.h ubev.cpp)
set(MBIE_EB_FILES chain_mdp.h env.h mbie_eb.h mbie_eb.cpp)
set(UBEV_EB_FILES chain_mdp.h env.h ubev_eb.h ubev_eb.cpp)
add_executable(UBEV ${UBEV_FILES})
add_executable(MBIE_EB ${MBIE_EB_FILES})
add_executable(UBEV_EB ${UBEV_EB_FILES})