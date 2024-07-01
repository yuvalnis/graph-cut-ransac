#pragma once

namespace gcransac::sampler
{

enum class SamplerType
{
    Uniform,
    ProSaC,
    ProgressiveNapsac,
    Importance,
    AdaptiveReordering
};

}
