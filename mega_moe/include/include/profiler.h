#pragma once
#include "profile_stages.h"
#include <type_traits>
#ifdef ENABLE_PROFILING
#include "profiler_core.h"
#include "profile_events_generated.h"
#define MOE_PROFILE_BIND(buf) using namespace MoeProfile; const uint64_t moeProfileBuffer = static_cast<uint64_t>(buf)
#define MOE_PROFILE_BEGIN(stage, payload) PROF_MARK(moeProfileBuffer, MoeProfile::BeginId(stage), payload)
#define MOE_PROFILE_END(stage, payload) PROF_MARK(moeProfileBuffer, MoeProfile::EndId(stage), payload)
#else
#define PROF_INIT(buf) ((void)0)
#define MOE_PROFILE_BIND(buf) using namespace MoeProfile
#define MOE_PROFILE_BEGIN(stage, payload) static_assert(std::is_same<typename std::remove_cv<decltype(stage)>::type, MoeProfile::MoeProfileStage>::value, "profile stage must be an enum")
#define MOE_PROFILE_END(stage, payload) MOE_PROFILE_BEGIN(stage, payload)
#endif
