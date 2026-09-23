# CMake generated Testfile for 
# Source directory: /home/b/gpu_histogram/training
# Build directory: /home/b/gpu_histogram/build/booster-level-batch
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[instrumentation_gpu]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_instrumentation_tests")
set_tests_properties([=[instrumentation_gpu]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;27;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[instrumentation_disabled]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_disabled_tests")
set_tests_properties([=[instrumentation_disabled]=] PROPERTIES  LABELS "cpu" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;34;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[instrumentation_observe]=] "/usr/bin/python3" "/home/b/gpu_histogram/training/tests/test_observe.py")
set_tests_properties([=[instrumentation_observe]=] PROPERTIES  LABELS "cpu" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;37;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[instrumentation_evaluate]=] "/usr/bin/python3" "/home/b/gpu_histogram/training/tests/test_evaluate.py")
set_tests_properties([=[instrumentation_evaluate]=] PROPERTIES  LABELS "cpu" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;37;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_booster]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_booster_tests")
set_tests_properties([=[trainer_booster]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_kernels]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_kernels_tests")
set_tests_properties([=[trainer_kernels]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_initialization]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_initialization_tests")
set_tests_properties([=[trainer_initialization]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_quantize]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_quantize_tests")
set_tests_properties([=[trainer_quantize]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_root_histogram]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_root_histogram_tests")
set_tests_properties([=[trainer_root_histogram]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_split_search]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_split_search_tests")
set_tests_properties([=[trainer_split_search]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_deeper_histogram]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_deeper_histogram_tests")
set_tests_properties([=[trainer_deeper_histogram]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_batch_resident]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_batch_resident_tests")
set_tests_properties([=[trainer_batch_resident]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
add_test([=[trainer_batch_training]=] "/home/b/gpu_histogram/build/booster-level-batch/ghb_batch_training_tests")
set_tests_properties([=[trainer_batch_training]=] PROPERTIES  LABELS "gpu" RUN_SERIAL "TRUE" SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/b/gpu_histogram/training/CMakeLists.txt;78;add_test;/home/b/gpu_histogram/training/CMakeLists.txt;0;")
