cmake_minimum_required(VERSION 3.10)
project( PI_VISION CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# find_package( Openmp Openmp )
# include_directories( ${Openmp_INCLUDE_DIRS} )
add_executable( distance distance_good.cpp)
target_link_libraries( distance ${OpenCV_LIBS} )
target_link_libraries(distance m wiringPi pthread)



# cmake_minimum_required(VERSION 3.10)
# project( PI_VISION C CXX)

# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( exec distance_good.cpp am7x.cpp)
# target_link_libraries( exec ${OpenCV_LIBS} )
# target_link_libraries(exec m wiringPi)
# set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
# set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# find_package(Threads REQUIRED)
# target_link_libraries(exec Threads::Threads)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

