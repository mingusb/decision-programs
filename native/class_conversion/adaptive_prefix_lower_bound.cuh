#pragma once
#include "adaptive_engine.cuh"

namespace class_conversion_adaptive::prefix_lower_bound {
enum class Rejection : u32 {
  none=0,invalid_channel_shape,missing_source_arrays,nonfinite_prefix,
  invalid_channel,invalid_root,nonfinite_minimum,nonfinite_addition,below_window
};
struct Result {
  // complete means every source inventory entry was inspected and the selected
  // lower trace stayed finite. It does not mean that a classification is proved.
  bool complete=false,lower_window_met=false;
  u32 lower_word=0;
  u32 root_visits=0,additions=0;
  Rejection rejection=Rejection::none;
};

// Range primitive only; no native authority or cache admission is created.
// Caller binds the exact ordered residual inventory/positions and projected
// guard to an admitted source. minimum[root] must already be a sound lower
// enclosure of that residual tree on the entire guard, with finite source leaf
// operands. For unused-prefix reuse the caller additionally needs the SAME
// program's full-guard qualified donor gap/window, unchanged used/structural
// words, a target prefix no greater than the donor in this genuine channel,
// channel separability, and its finite donor actual UPPER trace. That upper
// trace plus this finite LOWER trace encloses each target RN32 step; this
// function alone does not establish finite target execution or donor authority.
// A zero native_margin_classes sentinel means all source score channels.
// Structural suffix words are never interpreted as genuine class prefixes.
// The -10 endpoint is the current proved native score window, not a heuristic.
// All arithmetic uses original source order; consumed entries (-1) are skipped.
// root_visits counts inspected inventory entries (including inactive/other
// channels); additions counts selected RN32 operations, including a failed step.
__device__ inline Result check(EngineView e,const std::int32_t* ordered_residual_roots,
                              u32 genuine_channel,u32 target_prefix_word) {
  Result out;
  const auto& source=e.source;
  const u32 score_channels=source.native_margin_classes?source.native_margin_classes:source.classes;
  if(!source.classes||!score_channels||score_channels>source.classes||genuine_channel>=score_channels){
    out.rejection=Rejection::invalid_channel_shape;return out;
  }
  if(source.trees&&(!ordered_residual_roots||!source.channels||!e.minimum)){
    out.rejection=Rejection::missing_source_arrays;return out;
  }
  float lower=__uint_as_float(target_prefix_word);out.lower_word=target_prefix_word;
  if(!isfinite(lower)){out.rejection=Rejection::nonfinite_prefix;return out;}
  for(u32 t=0;t<source.trees;++t){
    ++out.root_visits;
    const auto root=ordered_residual_roots[t],channel=source.channels[t];
    if(channel<0||u32(channel)>=source.classes){out.rejection=Rejection::invalid_channel;return out;}
    if(root < -1||(root>=0&&u32(root)>=source.nodes)){out.rejection=Rejection::invalid_root;return out;}
    if(root<0||u32(channel)!=genuine_channel)continue;
    const float minimum=__uint_as_float(e.minimum[u32(root)]);
    if(!isfinite(minimum)){out.rejection=Rejection::nonfinite_minimum;return out;}
    lower=__fadd_rn(lower,minimum);++out.additions;out.lower_word=__float_as_uint(lower);
    if(!isfinite(lower)){out.rejection=Rejection::nonfinite_addition;return out;}
  }
  out.complete=true;
  if(lower < -10.f){out.rejection=Rejection::below_window;return out;}
  out.lower_window_met=true;return out;
}
} // namespace class_conversion_adaptive::prefix_lower_bound
