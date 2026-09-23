// Output-tile trainer. Included in booster.cpp after the frozen per-output implementation.
template<class Record, unsigned Order = 2> TrainingResult train_batch_impl(const Dataset& data, const TrainConfig& config) {
  static_assert(Order >= 2 && Order <= 4);
  using HistogramStats = std::conditional_t<Order == 2, gpu::Stats, gpu::HigherStats<Order>>;
  constexpr bool recording = !std::is_same_v<Record, gi::NullRecorder>;
  const auto total_start = Clock::now();
  TrainingResult result;
  auto& model = result.model;
  model.objective = config.objective;
  model.outputs = config.objective == Objective::multiclass_softmax ? config.classes : data.outputs;
  if (model.outputs > std::uint32_t(std::numeric_limits<std::int32_t>::max()) ||
      config.rounds > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("output/round count exceeds context index range");
  model.base_scores.resize(model.outputs);
  model.trees.reserve(product(config.rounds, model.outputs, "tree count"));
  result.training_loss.reserve(std::size_t(config.rounds) + 1);
  Record recorder(recording ? recorder_capacity(config, model.outputs) : 0, config.nvtx);
  gi::Context context;
  context.rows = data.rows; context.features = data.columns; context.stream_id = 1;
  Stream stream;
  QuantizedData quantized;
  auto phase = Clock::now();
  stage(recorder, gi::Stage::quantize, context, gi::Timing::host, nullptr, [&] {
    quantized = fit_quantize(data, config.max_bins, config.max_device_bytes, stream.value, config.quantize_policy);
    model.features = std::move(quantized.features);
  });
  result.quantize_ms = milliseconds(phase);
  const auto view = quantized.view;
  context.bins = view.total_bins;
  const bool multiclass = model.objective == Objective::multiclass_softmax;
  const bool reuse_counts = config.root_counts != RootCountPolicy::per_output;
  const unsigned levels = std::max(1U, config.max_depth);
  const unsigned history_levels = recording && config.tree_execution == TreeExecution::stream ? levels : 0U;
  const std::uint64_t leaves = std::max<std::uint64_t>(1, data.rows / config.min_leaf_rows);
  const auto desired_frontier = std::min<std::uint64_t>(1ULL << (config.max_depth ? config.max_depth - 1 : 0), leaves);
  const auto node_bound = std::min<std::uint64_t>((1ULL << (config.max_depth + 1)) - 1, leaves * 2 - 1);
  if (node_bound > unsigned(INT32_MAX)) invalid("tree capacity exceeds signed node indices");
  const unsigned tree_capacity = unsigned(node_bound);
  const auto predictions_count = product(data.rows, model.outputs, "predictions");
  const unsigned loss_blocks = std::min(4096U, 1 + (data.rows - 1) / 256);
  const unsigned init_chunks = std::min(256U, 1 + (data.rows - 1) / 1024);
  const auto init_count = product(init_chunks, std::size_t(model.outputs) + 1, "initialization partials");
  result.root_count_bytes = reuse_counts ? product(view.total_bins, sizeof(unsigned long long), "root counts") : 0;
  if (config.root_counts == RootCountPolicy::reuse_shared && !gpu::root_counts_shared_supported(view))
    invalid("shared root counts cannot hold the feature bins; select reuse-global");
  std::size_t free_bytes{}, total_bytes{};
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "query batch trainer memory budget");
  const auto budget = std::min(config.max_device_bytes, sum(quantized.resident_bytes, free_bytes, "available device budget"));
  struct Memory { std::size_t total{}, state{}, histogram{}, derivatives{}; };
  auto memory = [&](unsigned batch, unsigned capacity) {
    Memory m; m.total = quantized.resident_bytes;
    const auto bc = product(batch, capacity, "batched frontier");
    auto add = [&](std::size_t count, std::size_t element) { m.total = sum(m.total, product(count, element, "batch workspace"), "batch workspace"); };
    add(predictions_count, sizeof(double)); add(data.targets.size(), sizeof(float)); add(data.weights.size(), sizeof(float));
    add(model.outputs, sizeof(double)); add(loss_blocks + init_count + 2, sizeof(double));
    add(1, sizeof(unsigned) + sizeof(gpu::OutputBatch));
    m.derivatives = product(product(data.rows, multiclass ? model.outputs : batch, "derivatives"), Order * sizeof(double), "derivatives");
    add(m.derivatives, 1);
    m.histogram = sum(product(product(bc, view.total_bins, "batch histograms"), sizeof(HistogramStats), "batch histograms"), result.root_count_bytes, "histograms");
    add(m.histogram, 1); add(product(bc, data.columns, "split candidates"), sizeof(gpu::Split)); add(bc, sizeof(gpu::Split));
    const auto before = m.total;
    add(product(batch, data.rows, "assignments"), sizeof(int)); add(product(batch, tree_capacity, "tree nodes"), sizeof(Node));
    add(bc, 4 * sizeof(int) + sizeof(unsigned));
    add(product(batch, 1U + (capacity - 1) / 1024, "scan blocks"), sizeof(unsigned));
    add(batch, sizeof(gpu::TreeState) + sizeof(unsigned));
    m.state = m.total - before;
    add(product(batch, history_levels, "frontier history"), sizeof(unsigned));
    return m;
  };
  auto fits = [&](unsigned batch, unsigned capacity) {
    const auto m = memory(batch, capacity);
    return m.total <= budget && m.histogram <= config.max_histogram_bytes;
  };
  unsigned nodes_capacity = unsigned(desired_frontier), batch_capacity{};
  const unsigned requested_batch = std::min(model.outputs, config.output_tile_size);
  if (fits(1, nodes_capacity)) {
    unsigned lo = 1, hi = requested_batch;
    while (lo < hi) { const unsigned mid = lo + (hi - lo + 1) / 2; if (fits(mid, nodes_capacity)) lo = mid; else hi = mid - 1; }
    batch_capacity = lo;
  } else {
    if (!fits(1, 1)) invalid("memory budgets cannot hold one output and its root histogram");
    unsigned lo = 1, hi = nodes_capacity;
    while (lo < hi) { const unsigned mid = lo + (hi - lo + 1) / 2; if (fits(1, mid)) lo = mid; else hi = mid - 1; }
    nodes_capacity = lo; batch_capacity = 1;
  }
  const auto selected = memory(batch_capacity, nodes_capacity);
  const auto deeper_policy = config.histogram == HistogramPolicy::shared ? gpu::DeeperHistogramPolicy::shared1 : gpu::DeeperHistogramPolicy::global;
  if constexpr (Order == 2) {
    if (!gpu::deeper_histogram_supported(view, nodes_capacity, deeper_policy)) invalid("forced deeper histogram policy cannot hold configured frontier");
  }
  [[maybe_unused]] const unsigned chunks = deeper_policy == gpu::DeeperHistogramPolicy::global ? 0U : std::min(256U, 1 + (data.rows - 1) / 4096);
  const unsigned derivative_outputs = multiclass ? model.outputs : batch_capacity;
  const auto derivative_count = product(data.rows, derivative_outputs, "derivatives");
  const auto bc = product(batch_capacity, nodes_capacity, "batched frontier");
  const auto history_count = product(batch_capacity, history_levels, "batch history");
  const auto histogram_count = product(bc, view.total_bins, "histogram stats");
  const auto candidate_count = product(bc, data.columns, "split candidates");
  result.tree_batch_size = result.root_histogram_batch_size = batch_capacity;
  result.device_bytes = selected.total; result.tree_state_bytes = selected.state;
  result.gradient_bytes = selected.derivatives; result.histogram_bytes = selected.histogram;
  result.preparation_peak_bytes = quantized.peak_bytes;
  context.scratch_bytes = result.device_bytes - quantized.resident_bytes;
  constexpr std::size_t pinned_node_limit = 64ULL << 20;
  const auto tree_bytes = product(tree_capacity, sizeof(Node), "export tree");
  const unsigned export_batch = unsigned(std::min<std::size_t>(std::min(config.tree_export_batch_size, batch_capacity), pinned_node_limit / tree_bytes));
  result.tree_export_batch_size = export_batch;
  const auto export_node_count = product(export_batch, tree_capacity, "export nodes");
  result.pinned_export_bytes = sum(product(export_node_count, sizeof(Node), "pinned nodes"),
    sum(product(batch_capacity, sizeof(gpu::TreeState), "pinned states"),
      sum(sizeof(gpu::OutputBatch), product(history_count, sizeof(unsigned), "pinned history"), "pinned metadata"), "pinned metadata"), "pinned exports");
  HostScope preparation(recorder, gi::Stage::initialize, context);
  Device<double> predictions(predictions_count), gradient(derivative_count), hessian(derivative_count), base(model.outputs);
  // A zero-sized Device performs no CUDA allocation or free. Order 2 retains
  // its original G/H payload and every original compute entry point below.
  Device<double> third(Order >= 3 ? derivative_count : 0), fourth(Order >= 4 ? derivative_count : 0);
  Device<double> partial_loss(loss_blocks), scalar_loss(1), weight_sum(1), initial_partials(init_count);
  Device<float> targets(data.targets.size()), weights(data.weights.size());
  Device<int> assignments(product(batch_capacity, data.rows, "assignments")), frontier_a(bc), frontier_b(bc), left_map(bc), right_map(bc);
  Device<unsigned> scan_offsets(bc), scan_counts(product(batch_capacity, 1U + (nodes_capacity - 1) / 1024, "scan blocks"));
  Device<unsigned> initialization_status(1), active(batch_capacity), level_history(history_count);
  Device<HistogramStats> histograms(histogram_count);
  Device<unsigned long long> root_counts(reuse_counts ? view.total_bins : 0U);
  Device<gpu::Split> candidates(candidate_count), winners(bc);
  Device<Node> tree_nodes(product(batch_capacity, tree_capacity, "tree nodes"));
  Device<gpu::TreeState> tree_state(batch_capacity);
  Device<gpu::OutputBatch> selector(1);
  Pinned<gpu::TreeState> host_state(batch_capacity);
  Pinned<gpu::OutputBatch> host_selector(1);
  Pinned<Node> host_nodes(export_node_count);
  Pinned<double> host_loss(1), host_weight(1);
  Pinned<unsigned> host_status(1), host_history(history_count);
  std::unique_ptr<TreeGraph> graph;
  DrainStream drain{stream.value};
  std::vector<unsigned> round_history(product(model.outputs, history_levels, "round history"));
  preparation.finish();
  phase = Clock::now();
  stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] {
    targets.upload(data.targets.data(), data.targets.size(), stream.value);
    weights.upload(data.weights.data(), data.weights.size(), stream.value);
  });
  stream.wait(); result.upload_ms = milliseconds(phase);
  stage(recorder, gi::Stage::initialize, context, gi::Timing::gpu, stream.value, [&] {
    check(gpu::initialize_training(model.objective, targets.data, weights.data, data.rows, model.outputs,
                                  base.data, weight_sum.data, initial_partials.data, init_chunks,
                                  initialization_status.data, stream.value), "validate targets/weights and compute GPU base scores");
  });
  stage(recorder, gi::Stage::download, context, gi::Timing::host, nullptr, [&] {
    check(cudaMemcpyAsync(model.base_scores.data(), base.data, base.bytes(), cudaMemcpyDeviceToHost, stream.value), "export base scores");
    check(cudaMemcpyAsync(host_weight.data, weight_sum.data, sizeof(double), cudaMemcpyDeviceToHost, stream.value), "export total weight");
    check(cudaMemcpyAsync(host_status.data, initialization_status.data, sizeof(unsigned), cudaMemcpyDeviceToHost, stream.value), "export initialization status");
    stream.wait();
    if (*host_status.data) invalid("GPU target/weight/base validation failed, status=" + std::to_string(*host_status.data));
    if (!std::isfinite(*host_weight.data) || !(*host_weight.data > 0)) invalid("invalid GPU total weight");
  });
  stage(recorder, gi::Stage::initialize, context, gi::Timing::gpu, stream.value, [&] {
    // Full-capacity copies include the unused tail. Initialize it once, outside
    // boosting; later builders overwrite live nodes and leave a defined tail.
    // Compact exports also copy Node alignment padding, which member stores do
    // not initialize. Define the complete representation once for every mode.
    check(cudaMemsetAsync(tree_nodes.data, 0, tree_nodes.bytes(), stream.value), "initialize tree export storage");
    check(gpu::initialize_predictions(predictions.data, base.data, data.rows, model.outputs, stream.value), "initialize predictions");
  });
  auto evaluate = [&] {
    stage(recorder, gi::Stage::evaluate, context, gi::Timing::gpu, stream.value, [&] {
      check(gpu::loss(model.objective, predictions.data, targets.data, weights.data, data.rows, model.outputs,
                      partial_loss.data, loss_blocks, stream.value), "evaluate objective partial sums");
      check(gpu::finalize_loss(partial_loss.data, loss_blocks, weight_sum.data,
                              model.objective == Objective::multiclass_softmax ? 1U : model.outputs,
                              scalar_loss.data, stream.value), "reduce and normalize objective on GPU");
    });
    stage(recorder, gi::Stage::download, context, gi::Timing::host, nullptr, [&] {
      check(cudaMemcpyAsync(host_loss.data, scalar_loss.data, sizeof(double), cudaMemcpyDeviceToHost, stream.value), "export objective scalar");
      stream.wait();
      if (!std::isfinite(*host_loss.data) || *host_loss.data < 0) throw std::runtime_error("invalid GPU objective scalar");
      result.training_loss.push_back(*host_loss.data);
    });
  };
  evaluate(); collect(recorder, result);
  const gpu::SplitConfig split_config{config.min_leaf_rows, config.l2, config.min_child_hessian, config.min_gain, config.max_leaf_value};
  auto submit_batch = [&]<class R>(R& stages) {
    gpu::BatchResidentView resident{batch_capacity, nodes_capacity, tree_capacity,
      assignments.data, tree_nodes.data, frontier_a.data, frontier_b.data, left_map.data, right_map.data,
      tree_state.data, active.data, scan_offsets.data, scan_counts.data};
    auto work = context; work.depth = -1; work.active_nodes = 0;
    stage(stages, gi::Stage::initialize, work, gi::Timing::gpu, stream.value, [&] {
      check(gpu::resident_batch_initialize(data.rows, resident, selector.data, stream.value), "initialize output tree batch");
    });
    for (unsigned depth = 0; depth < levels; ++depth) {
      work.depth = int(depth);
      if (history_count) check(cudaMemcpyAsync(level_history.data + std::size_t(depth) * batch_capacity, active.data,
        active.bytes(), cudaMemcpyDeviceToDevice, stream.value), "record batch frontier sizes");
      stage(stages, gi::Stage::histogram, work, gi::Timing::gpu, stream.value, [&] {
        if constexpr (Order == 2) {
          if (!depth) check(gpu::root_histogram_batch(view, gradient.data, hessian.data, derivative_outputs,
            batch_capacity, product(nodes_capacity, view.total_bins, "root output stride"), selector.data,
            histograms.data, stream.value, root_counts.data), "root histogram into retained batch storage");
          else check(gpu::deeper_histogram(view, assignments.data, gradient.data, hessian.data, derivative_outputs, 0,
            batch_capacity, nodes_capacity, active.data, histograms.data, deeper_policy, chunks, stream.value, selector.data), "batched deeper histogram");
        } else {
          if (!depth) check(gpu::higher_root_histogram_batch<Order>(view, gradient.data, hessian.data, third.data, fourth.data,
            derivative_outputs, batch_capacity, product(nodes_capacity, view.total_bins, "root output stride"), selector.data,
            histograms.data, stream.value, root_counts.data), "higher-order root histogram into retained batch storage");
          else check(gpu::higher_deeper_histogram<Order>(view, assignments.data, gradient.data, hessian.data, third.data, fourth.data,
            derivative_outputs, batch_capacity, nodes_capacity, active.data, selector.data, histograms.data,
            stream.value), "higher-order batched deeper histogram");
        }
      });
      stage(stages, gi::Stage::split_search, work, gi::Timing::gpu, stream.value, [&] {
        if constexpr (Order == 2) {
          const auto search = config.split_policy == SplitPolicy::warp_wide ? gpu::find_splits_warp_wide_batched_active :
            config.split_policy == SplitPolicy::warp32 ? gpu::find_splits_warp_batched_active : gpu::find_splits_batched_active;
          check(search(view, histograms.data, batch_capacity, nodes_capacity, active.data, split_config, config.max_depth == 0,
            candidates.data, winners.data, stream.value, &selector.data->output_count), "batched independent split search");
        } else {
          check(gpu::higher_find_splits_batched_active<Order>(view, histograms.data, batch_capacity, nodes_capacity, active.data,
            split_config, config.max_depth == 0, candidates.data, winners.data, config.split_policy, stream.value,
            &selector.data->output_count), "higher-order batched independent split search");
        }
      });
      stage(stages, gi::Stage::route, work, gi::Timing::gpu, stream.value, [&] {
        check(gpu::resident_batch_materialize(view, winners.data, resident, selector.data, depth + 1 < config.max_depth,
          config.learning_rate, stream.value), "materialize independent tree frontiers");
        check(gpu::resident_batch_route(view, winners.data, resident, selector.data, stream.value), "route independent output rows");
        check(gpu::resident_batch_advance(resident, selector.data, stream.value), "advance independent tree frontiers");
      });
      std::swap(resident.frontier, resident.next_frontier);
    }
    work.depth = -1;
    stage(stages, gi::Stage::prediction, work, gi::Timing::gpu, stream.value, [&] {
      check(gpu::resident_batch_predict(view, resident, selector.data, model.outputs, predictions.data, stream.value), "apply independent tree batch");
    });
  };
  const auto training_start = Clock::now();
  if (reuse_counts && config.rounds) {
    auto work = context; work.depth = 0; work.active_nodes = 1; work.logical_write_bytes = result.root_count_bytes;
    stage(recorder, gi::Stage::histogram, work, gi::Timing::gpu, stream.value, [&] {
      check(gpu::root_counts(view, root_counts.data, config.root_counts == RootCountPolicy::reuse_shared ? gpu::RootCountKernel::shared : gpu::RootCountKernel::global,
        stream.value), "compute invariant root counts");
    });
  }
  for (unsigned round = 0; round < config.rounds; ++round) {
    context.round = int(round); context.output = -1; context.depth = -1; context.operations = 1;
    if constexpr (Order == 2) {
      if (multiclass) stage(recorder, gi::Stage::gradients, context, gi::Timing::gpu, stream.value, [&] {
        check(gpu::gradients(model.objective, predictions.data, targets.data, weights.data, gradient.data, hessian.data,
          data.rows, model.outputs, stream.value), "compute full pre-round softmax derivatives");
      });
    }
    for (unsigned tile_begin = 0; tile_begin < model.outputs;) {
      const unsigned tile_count = std::min(batch_capacity, model.outputs - tile_begin);
      context.output = int(tile_begin); context.operations = tile_count;
      if (!multiclass) stage(recorder, gi::Stage::gradients, context, gi::Timing::gpu, stream.value, [&] {
        if constexpr (Order == 2) {
          check(gpu::gradients_tile(model.objective, predictions.data, targets.data, weights.data, gradient.data, hessian.data,
            data.rows, model.outputs, tile_begin, tile_count, stream.value), "compute compact derivative tile");
        } else {
          check(gpu::higher_gradients_tile<Order>(predictions.data, targets.data, weights.data, gradient.data, hessian.data,
            third.data, fourth.data, data.rows, model.outputs, tile_begin, tile_count, stream.value), "compute compact higher-order derivative tile");
        }
      });
      *host_selector.data = gpu::OutputBatch{tile_begin, tile_count, multiclass ? tile_begin : 0U, multiclass ? model.outputs : tile_count};
      stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] { selector.upload(host_selector.data, 1, stream.value); });
      if (config.tree_execution == TreeExecution::graph && !graph) {
        stage(recorder, gi::Stage::initialize, context, gi::Timing::host, nullptr, [&] {
          graph = std::make_unique<TreeGraph>(); gi::NullRecorder disabled;
          graph->capture(stream.value, [&] { submit_batch(disabled); });
        });
      }
      stage(recorder, gi::Stage::tree_build, context, gi::Timing::gpu, stream.value, [&] {
        if (graph) check(cudaGraphLaunch(graph->executable, stream.value), "replay output tree batch");
        else submit_batch(recorder);
      });
      stage(recorder, gi::Stage::download, context, gi::Timing::host, nullptr, [&] {
        check(cudaMemcpyAsync(host_state.data, tree_state.data, product(tile_count, sizeof(gpu::TreeState), "state export"), cudaMemcpyDeviceToHost, stream.value), "export independent states");
        if (history_count) check(cudaMemcpyAsync(host_history.data, level_history.data, level_history.bytes(), cudaMemcpyDeviceToHost, stream.value), "export batch frontier history");
        // Full-capacity chunks combine state and first node export in one wait.
        if (export_batch) {
          for (unsigned begin = 0; begin < tile_count; begin += export_batch) {
            const unsigned count = std::min(export_batch, tile_count - begin);
            check(cudaMemcpyAsync(host_nodes.data, tree_nodes.data + std::size_t(begin) * tree_capacity,
              product(count, tree_bytes, "node export"), cudaMemcpyDeviceToHost, stream.value), "export bounded tree batch");
            stream.wait();
            for (unsigned slot = 0; slot < count; ++slot) {
              const auto& state = host_state.data[begin + slot];
              if (state.status) invalid("GPU tree construction failed, status=" + std::to_string(state.status));
              if (!state.node_count || state.node_count > tree_capacity) throw std::runtime_error("invalid exported tree size");
              Tree tree; tree.output = tile_begin + begin + slot;
              const auto* first = host_nodes.data + std::size_t(slot) * tree_capacity;
              tree.nodes.assign(first, first + state.node_count); model.trees.push_back(std::move(tree));
            }
          }
        } else {
          stream.wait();
          for (unsigned slot = 0; slot < tile_count; ++slot) {
            const auto& state = host_state.data[slot];
            if (state.status) invalid("GPU tree construction failed, status=" + std::to_string(state.status));
            if (!state.node_count || state.node_count > tree_capacity) throw std::runtime_error("invalid exported tree size");
            Tree tree; tree.output = tile_begin + slot; tree.nodes.resize(state.node_count);
            model.trees.push_back(std::move(tree));
            auto& retained = model.trees.back();
            check(cudaMemcpyAsync(retained.nodes.data(), tree_nodes.data + std::size_t(slot) * tree_capacity,
              product(state.node_count, sizeof(Node), "compact export"), cudaMemcpyDeviceToHost, stream.value), "export compact tree");
          }
          stream.wait();
        }
        for (unsigned depth = 0; depth < history_levels; ++depth)
          for (unsigned slot = 0; slot < tile_count; ++slot)
            round_history[std::size_t(tile_begin + slot) * history_levels + depth] = host_history.data[std::size_t(depth) * batch_capacity + slot];
      });
      tile_begin += tile_count;
    }
    context.output = -1; context.operations = 1;
    evaluate();
    if constexpr (recording) {
      auto samples = recorder.collect(false);
      if (!samples) throw std::logic_error("completed batch round has pending events");
      if (history_count) for (auto& sample : *samples) {
        auto& work = sample.context;
        if (work.output >= 0 && unsigned(work.output) < model.outputs && work.depth >= 0 && unsigned(work.depth) < levels) {
          std::uint64_t active_sum{};
          for (unsigned slot = 0; slot < work.operations && unsigned(work.output) + slot < model.outputs; ++slot)
            active_sum += round_history[std::size_t(unsigned(work.output) + slot) * history_levels + unsigned(work.depth)];
          if (active_sum > UINT32_MAX) throw std::runtime_error("recorded batch active-node count overflow");
          work.active_nodes = unsigned(active_sum);
        }
      }
      result.samples.insert(result.samples.end(), samples->begin(), samples->end()); recorder.reset();
    }
  }
  stream.wait(); result.training_ms = milliseconds(training_start);
  validate_model(model); result.total_ms = milliseconds(total_start);
  return result;
}
