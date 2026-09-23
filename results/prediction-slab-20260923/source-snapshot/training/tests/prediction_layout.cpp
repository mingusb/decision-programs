#include "ghb/detail/prediction_layout.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::size_t checks{};
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function, std::string_view diagnostic) {
  bool rejected{};
  try { function(); }
  catch (const std::invalid_argument& error) {
    rejected = true;
    require(std::string_view(error.what()).find(diagnostic) != std::string_view::npos,
            "overflow rejected at unexpected arithmetic stage");
  }
  require(rejected, "invalid size or memory budget was accepted");
}
using ghb::detail::PredictionSlabLayout;
using ghb::detail::PredictionSlabRegion;
void exact_regions(const PredictionSlabLayout& layout,
                   const std::array<PredictionSlabRegion, 4>& expected) {
  const std::array actual{layout.base, layout.nodes, layout.descriptors, layout.offsets};
  std::size_t end{}, payload{}, gaps{};
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto region = actual[i];
    require(region.offset == expected[i].offset && region.bytes == expected[i].bytes,
            "hand-derived slab region differs");
    if (!region.bytes) {
      require(region.offset == 0, "absent array has a fictitious storage region");
      continue;
    }
    require(region.offset % 256 == 0, "array start lost original CUDA allocation alignment");
    require(region.offset >= end, "regions overlap");
    require(region.offset <= layout.device_bytes && region.bytes <= layout.device_bytes - region.offset,
            "region exceeds allocated/uploaded extent");
    gaps += region.offset - end;
    payload += region.bytes;
    end = region.offset + region.bytes;
  }
  require(end == layout.device_bytes && payload + gaps == layout.device_bytes,
          "layout loses payload or hides trailing device padding");
  require(gaps <= 3 * 255, "four-array padding bound exceeded");
  require(layout.host_bytes >= layout.device_bytes && layout.host_bytes % 256 == 0 &&
          layout.host_bytes - layout.device_bytes < 256, "host capacity does not cover exactly the rounded slab");
  require(layout.host_blocks == layout.host_bytes / 256, "host block count differs from allocation extent");
  require(layout.reserved_bytes - layout.device_bytes == layout.prediction_bytes,
          "prediction reservation omits or double-counts the slab");
}
} // namespace

int main() {
  try {
    // This test intentionally has no model/CUDA dependency: 32/16 are explicit
    // fixture ABI sizes, also supplied through sizeof by the production caller.
    using ghb::detail::make_prediction_slab_layout;
    static_assert(sizeof(double) == 8 && sizeof(std::uint64_t) == 8);
    const auto odd = make_prediction_slab_layout<32, 16>(3, 13, 3, 96);
    exact_regions(odd, {{{0, 24}, {256, 416}, {768, 48}, {1024, 32}}});
    require(odd.output_offset_count == 4 && odd.device_bytes == 1056 && odd.host_bytes == 1280 &&
            odd.prediction_bytes == 768 && odd.reserved_bytes == 1824, "small measured-model accounting differs");

    const auto empty = make_prediction_slab_layout<32, 16>(1, 0, 0, 32);
    exact_regions(empty, {{{0, 8}, {0, 0}, {0, 0}, {256, 16}}});
    require(empty.device_bytes == 272 && empty.host_bytes == 512 && empty.reserved_bytes == 528,
            "empty forest accounting differs");

    const auto aligned = make_prediction_slab_layout<32, 16>(32, 16, 16, 32);
    exact_regions(aligned, {{{0, 256}, {256, 512}, {768, 256}, {1024, 264}}});
    require(aligned.device_bytes == 1288 && aligned.host_bytes == 1536 && aligned.reserved_bytes == 1544,
            "already aligned regions acquired extra padding");

    const auto wide = make_prediction_slab_layout<32, 16>(65, 3, 3, 65);
    exact_regions(wide, {{{0, 520}, {768, 96}, {1024, 48}, {1280, 528}}});
    require(wide.device_bytes == 1808 && wide.host_bytes == 2048 && wide.reserved_bytes == 2328,
            "wide sparse-forest accounting differs");

    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    using ghb::detail::prediction_layout_detail::product;
    using ghb::detail::prediction_layout_detail::sum;
    require(sum(maximum - 1, 1, "test") == maximum, "valid maximum addition rejected");
    require(product(maximum / 8, 8, "test") == maximum - 7, "valid near-maximum product changed");
    require(product(maximum, 0, "test") == 0, "zero-width product changed");
    rejects([&] { (void)sum(maximum, 1, "sum"); }, "sum size overflow");
    rejects([&] { (void)product(maximum / 8 + 1, 8, "product"); }, "product size overflow");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(maximum, 0, 0, 0); }, "inference output offsets");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(maximum / 8 + 1, 0, 0, 0); }, "inference slab region");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, maximum / 32 + 1, 1, 0); }, "inference slab region");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, 1, maximum / 16 + 1, 0); }, "inference slab region");
    // Synthetic count-only cases exercise different failing stages; they never
    // allocate their apparent model sizes or promise model semantic validity.
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, (maximum - 255) / 32, 1, 0); },
            "inference slab extent");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, (maximum - 255) / 32 - 1, 1, 0); },
            "inference slab alignment");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, (maximum - 255) / 32 - 16, 1, 0); },
            "inference host slab extent");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, 0, 0, maximum / 8 + 1); }, "inference predictions");
    rejects([&] { (void)make_prediction_slab_layout<32, 16>(1, 0, 0, maximum / 8); }, "inference slab workspace");

    rejects([&] { (void)odd.quantizer_budget(0); }, "exceeds available GPU memory");
    rejects([&] { (void)odd.quantizer_budget(1823); }, "exceeds available GPU memory");
    rejects([&] { (void)odd.quantizer_budget(1824); }, "exceeds available GPU memory");
    require(odd.quantizer_budget(1825) == 1, "strict free-memory boundary differs");
    require(odd.quantizer_budget(5920) == 4096, "quantizer remainder lost padded model bytes");
    require(odd.quantizer_budget(maximum) == maximum - 1824, "maximum free-memory remainder wraps");
    std::cout << "prediction layout CPU checks passed: " << checks << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prediction layout test failed: " << error.what() << '\n';
    return 1;
  }
}
