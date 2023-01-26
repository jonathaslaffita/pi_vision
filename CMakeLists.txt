cmake_minimum_required(VERSION 3.10)
project( PI_VISION CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# find_package( Openmp Openmp )
# include_directories( ${Openmp_INCLUDE_DIRS} )
add_executable( exec distance_good.cpp am7x.cpp)
target_link_libraries( exec ${OpenCV_LIBS} )
target_link_libraries(exec m wiringPi pthread)



