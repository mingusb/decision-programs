#include "common.hpp"
#include <iostream>

std::uint64_t digest(const std::vector<unsigned>& keys) {
  std::uint64_t value = 14695981039346656037ull;
  for (auto key : keys) { value ^= key; value *= 1099511628211ull; }
  return value;
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() try {
  // Golden values recorded from the unchanged historical generator before
  // adding located skew. This test links no CUDA runtime and uses no GPU.
  const std::pair<const char*, std::uint64_t> golden[] = {
      {"uniform", 10890433314986218069ull}, {"hot90", 16491493750015431021ull},
      {"hot99", 7225261831693095939ull}, {"single", 14846171880956040248ull}};
  for (const auto& [name, hash] : golden) {
    Dataset data(131071, 32768, gh::InputType::u32, name, "shuffled", 20260922);
    require(digest(data.keys) == hash, "historical input sequence changed");
  }
  for (unsigned bins : {24577u, 32768u, 1048576u}) {
    Dataset historical(100003, bins, gh::InputType::u32, "hot99", "shuffled", 715);
    Dataset explicit_last(100003, bins, gh::InputType::u32,
                          "hot99@" + std::to_string(bins - 1), "shuffled", 715);
    require(historical.keys == explicit_last.keys, "explicit last-bin sequence differs");
    for (unsigned hot : {24575u, 24576u, bins - 1}) {
      const auto distribution = "hot99@" + std::to_string(hot);
      Dataset shuffled(100003, bins, gh::InputType::u32, distribution, "shuffled", 715);
      Dataset sorted(100003, bins, gh::InputType::u32, distribution, "sorted", 715);
      std::vector<std::uint64_t> counted(bins);
      for (auto key : shuffled.keys) { require(key < bins, "invalid key"); ++counted[key]; }
      require(counted == shuffled.expected, "independent counts differ");
      require(counted == sorted.expected, "sorting changes counts");
      require(std::is_sorted(sorted.keys.begin(), sorted.keys.end()), "not sorted");
      require(counted[hot] > 98000 && counted[hot] < 100003, "dominant bin misplaced");
      auto reordered = shuffled.keys;
      std::sort(reordered.begin(), reordered.end());
      require(reordered == sorted.keys, "sorted input is not same multiset");
      for (std::size_t i = 0; i < historical.keys.size(); ++i)
        require(historical.keys[i] == shuffled.keys[i] ||
                (historical.keys[i] == bins - 1 && shuffled.keys[i] == hot),
                "moving dominant bin changes background input");
    }
  }
  for (const auto* bad : {"hot99@", "hot99@-1", "hot99@+1", "hot99@32", "hot99@1x",
                           "hot99@4294967296"}) {
    bool rejected = false;
    try { Dataset data(0, 32, gh::InputType::u32, bad, "shuffled", 715); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid hot-bin location accepted");
  }
  std::cout << "Historical sequence and located-skew checks passed.\n";
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n'; return 1;
}
