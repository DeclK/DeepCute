// A simple pipeline states to identify its 0-cycle or 1-cycle data
#include <cute/tensor.hpp>
using namespace cute;

template <int Stages>
struct PipelineStates {
    static_assert(Stages > 0);
    uint32_t stage_idx = 0;    
    uint32_t phase = 0;

    CUTE_DEVICE
    void operator++(int){
        if ((++stage_idx) == Stages) {
            stage_idx = 0;
            phase ^= 1;
        }
    }

    CUTE_DEVICE
    void advance(uint32_t steps) {
        // directly advance more steps
        // see how many full Stages does the step advanced
        if (((stage_idx + steps) / Stages) % 2 == 1) {
            phase ^= 1
        }
        stage_idx = (stage_idx + steps) % Stages;
    }
};
