cmake_minimum_required(VERSION 3.0)
project(fsh)

find_package(PCL 1.9 REQUIRED)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like build/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()



include_directories(
    ${PCL_INCLUDE_DIRS}
    .
)

set(CMAKE_CXX_STANDARD 17)

set(filelists
	models/bunny.obj
	models/cube.obj
	models/dragon.obj
	models/fish_512.obj
	models/suzanne.obj
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(
    ${PCL_DEFINITIONS}
    -g
)

add_executable(fsh
    main.cpp
)

target_link_libraries (fsh ${PCL_LIBRARIES})

foreach(file ${filelists})
	configure_file(${PROJECT_SOURCE_DIR}/${file} ${PROJECT_BINARY_DIR}/${file} COPYONLY)
endforeach()
