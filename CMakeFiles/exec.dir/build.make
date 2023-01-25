# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jonathas/pi_vision

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jonathas/pi_vision

# Include any dependencies generated for this target.
include CMakeFiles/exec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/exec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/exec.dir/flags.make

CMakeFiles/exec.dir/distance_good.cpp.o: CMakeFiles/exec.dir/flags.make
CMakeFiles/exec.dir/distance_good.cpp.o: distance_good.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jonathas/pi_vision/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/exec.dir/distance_good.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/exec.dir/distance_good.cpp.o -c /home/jonathas/pi_vision/distance_good.cpp

CMakeFiles/exec.dir/distance_good.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/exec.dir/distance_good.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jonathas/pi_vision/distance_good.cpp > CMakeFiles/exec.dir/distance_good.cpp.i

CMakeFiles/exec.dir/distance_good.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/exec.dir/distance_good.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jonathas/pi_vision/distance_good.cpp -o CMakeFiles/exec.dir/distance_good.cpp.s

CMakeFiles/exec.dir/am7x.cpp.o: CMakeFiles/exec.dir/flags.make
CMakeFiles/exec.dir/am7x.cpp.o: am7x.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jonathas/pi_vision/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/exec.dir/am7x.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/exec.dir/am7x.cpp.o -c /home/jonathas/pi_vision/am7x.cpp

CMakeFiles/exec.dir/am7x.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/exec.dir/am7x.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jonathas/pi_vision/am7x.cpp > CMakeFiles/exec.dir/am7x.cpp.i

CMakeFiles/exec.dir/am7x.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/exec.dir/am7x.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jonathas/pi_vision/am7x.cpp -o CMakeFiles/exec.dir/am7x.cpp.s

# Object files for target exec
exec_OBJECTS = \
"CMakeFiles/exec.dir/distance_good.cpp.o" \
"CMakeFiles/exec.dir/am7x.cpp.o"

# External object files for target exec
exec_EXTERNAL_OBJECTS =

exec: CMakeFiles/exec.dir/distance_good.cpp.o
exec: CMakeFiles/exec.dir/am7x.cpp.o
exec: CMakeFiles/exec.dir/build.make
exec: /usr/local/lib/libopencv_gapi.so.4.7.0
exec: /usr/local/lib/libopencv_stitching.so.4.7.0
exec: /usr/local/lib/libopencv_aruco.so.4.7.0
exec: /usr/local/lib/libopencv_barcode.so.4.7.0
exec: /usr/local/lib/libopencv_bgsegm.so.4.7.0
exec: /usr/local/lib/libopencv_bioinspired.so.4.7.0
exec: /usr/local/lib/libopencv_ccalib.so.4.7.0
exec: /usr/local/lib/libopencv_dnn_objdetect.so.4.7.0
exec: /usr/local/lib/libopencv_dnn_superres.so.4.7.0
exec: /usr/local/lib/libopencv_dpm.so.4.7.0
exec: /usr/local/lib/libopencv_face.so.4.7.0
exec: /usr/local/lib/libopencv_freetype.so.4.7.0
exec: /usr/local/lib/libopencv_fuzzy.so.4.7.0
exec: /usr/local/lib/libopencv_hfs.so.4.7.0
exec: /usr/local/lib/libopencv_img_hash.so.4.7.0
exec: /usr/local/lib/libopencv_intensity_transform.so.4.7.0
exec: /usr/local/lib/libopencv_line_descriptor.so.4.7.0
exec: /usr/local/lib/libopencv_mcc.so.4.7.0
exec: /usr/local/lib/libopencv_quality.so.4.7.0
exec: /usr/local/lib/libopencv_rapid.so.4.7.0
exec: /usr/local/lib/libopencv_reg.so.4.7.0
exec: /usr/local/lib/libopencv_rgbd.so.4.7.0
exec: /usr/local/lib/libopencv_saliency.so.4.7.0
exec: /usr/local/lib/libopencv_stereo.so.4.7.0
exec: /usr/local/lib/libopencv_structured_light.so.4.7.0
exec: /usr/local/lib/libopencv_superres.so.4.7.0
exec: /usr/local/lib/libopencv_surface_matching.so.4.7.0
exec: /usr/local/lib/libopencv_tracking.so.4.7.0
exec: /usr/local/lib/libopencv_videostab.so.4.7.0
exec: /usr/local/lib/libopencv_wechat_qrcode.so.4.7.0
exec: /usr/local/lib/libopencv_xfeatures2d.so.4.7.0
exec: /usr/local/lib/libopencv_xobjdetect.so.4.7.0
exec: /usr/local/lib/libopencv_xphoto.so.4.7.0
exec: /usr/local/lib/libopencv_shape.so.4.7.0
exec: /usr/local/lib/libopencv_highgui.so.4.7.0
exec: /usr/local/lib/libopencv_datasets.so.4.7.0
exec: /usr/local/lib/libopencv_plot.so.4.7.0
exec: /usr/local/lib/libopencv_text.so.4.7.0
exec: /usr/local/lib/libopencv_ml.so.4.7.0
exec: /usr/local/lib/libopencv_phase_unwrapping.so.4.7.0
exec: /usr/local/lib/libopencv_optflow.so.4.7.0
exec: /usr/local/lib/libopencv_ximgproc.so.4.7.0
exec: /usr/local/lib/libopencv_video.so.4.7.0
exec: /usr/local/lib/libopencv_videoio.so.4.7.0
exec: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
exec: /usr/local/lib/libopencv_objdetect.so.4.7.0
exec: /usr/local/lib/libopencv_calib3d.so.4.7.0
exec: /usr/local/lib/libopencv_dnn.so.4.7.0
exec: /usr/local/lib/libopencv_features2d.so.4.7.0
exec: /usr/local/lib/libopencv_flann.so.4.7.0
exec: /usr/local/lib/libopencv_photo.so.4.7.0
exec: /usr/local/lib/libopencv_imgproc.so.4.7.0
exec: /usr/local/lib/libopencv_core.so.4.7.0
exec: CMakeFiles/exec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jonathas/pi_vision/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable exec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/exec.dir/build: exec

.PHONY : CMakeFiles/exec.dir/build

CMakeFiles/exec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/exec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/exec.dir/clean

CMakeFiles/exec.dir/depend:
	cd /home/jonathas/pi_vision && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jonathas/pi_vision /home/jonathas/pi_vision /home/jonathas/pi_vision /home/jonathas/pi_vision /home/jonathas/pi_vision/CMakeFiles/exec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/exec.dir/depend

