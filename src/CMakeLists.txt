
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer/arm)

set(ncnn_SRCS
    allocator.cpp
    blob.cpp
    cpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    mat_pixel_resize.cpp
    modelbin.cpp
    net.cpp
    opencv.cpp
    paramdict.cpp
    benchmark.cpp
)

macro(ncnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp")

		# option teec
		if(NCNN_TEEC)
            set(arch teec)
            set(LAYER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/${arch}/${name}_${arch}.cpp)
            if(EXISTS ${LAYER_SRC})
                set(WITH_LAYER_${name}_${arch} 1)
                list(APPEND ncnn_SRCS "${LAYER_SRC}")
                list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/arm/${name}_arm.cpp")
            else()
                set(LAYER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/arm/${name}_arm.cpp)
                if(EXISTS ${LAYER_SRC})
                    set(WITH_LAYER_${name}_arm 1)
                    list(APPEND ncnn_SRCS "${LAYER_SRC}")
                endif()
            endif()

            if(WITH_LAYER_${name}_${arch})
                string(APPEND layer_declaration
                    "extern Layer* ${class}_${arch}_layer_creator();\n")
                string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_${arch}_layer_creator},\n#else\n{${class}_${arch}_layer_creator},\n#endif\n")
            elseif(WITH_LAYER_${name}_arm)
                string(APPEND layer_declaration
                    "extern Layer* ${class}_arm_layer_creator();\n")
                string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_arm_layer_creator},\n#else\n{${class}_arm_layer_creator},\n#endif\n")
            else()
                string(APPEND layer_declaration
                    "extern Layer* ${class}_layer_creator();\n")
                string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
            endif()
        else()
            set(arch arm)
            set(LAYER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/${arch}/${name}_${arch}.cpp)
            if(EXISTS ${LAYER_SRC})
                set(WITH_LAYER_${name}_${arch} 1)
                list(APPEND ncnn_SRCS "${LAYER_SRC}")
            endif()

            if(WITH_LAYER_${name}_${arch})
                string(APPEND layer_declaration
                    "extern Layer* ${class}_${arch}_layer_creator();\n")
                string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_${arch}_layer_creator},\n#else\n{${class}_${arch}_layer_creator},\n#endif\n")
            else()
                string(APPEND layer_declaration
                    "extern Layer* ${class}_layer_creator();\n")
                string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
            endif()
        endif()
        
    else()
        string(APPEND layer_registry "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    string(APPEND layer_type_enum "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

set(__LAYER_TYPE_ENUM_INDEX 0)

# layer implementation
ncnn_add_layer(AbsVal)
ncnn_add_layer(ArgMax OFF)
ncnn_add_layer(BatchNorm)
ncnn_add_layer(Bias)
ncnn_add_layer(BNLL)
ncnn_add_layer(Concat)
ncnn_add_layer(Convolution)
ncnn_add_layer(Crop)
ncnn_add_layer(Deconvolution)
ncnn_add_layer(Dropout)
ncnn_add_layer(Eltwise)
ncnn_add_layer(ELU)
ncnn_add_layer(Embed)
ncnn_add_layer(Exp)
ncnn_add_layer(Flatten)
ncnn_add_layer(InnerProduct)
ncnn_add_layer(Input)
ncnn_add_layer(Log)
ncnn_add_layer(LRN)
ncnn_add_layer(MemoryData)
ncnn_add_layer(MVN)
ncnn_add_layer(Pooling)
ncnn_add_layer(Power)
ncnn_add_layer(PReLU)
ncnn_add_layer(Proposal)
ncnn_add_layer(Reduction)
ncnn_add_layer(ReLU)
ncnn_add_layer(Reshape)
ncnn_add_layer(ROIPooling)
ncnn_add_layer(Scale)
ncnn_add_layer(Sigmoid)
ncnn_add_layer(Slice)
ncnn_add_layer(Softmax)
ncnn_add_layer(Split)
ncnn_add_layer(SPP OFF)
ncnn_add_layer(TanH)
ncnn_add_layer(Threshold)
ncnn_add_layer(Tile OFF)
ncnn_add_layer(RNN OFF)
ncnn_add_layer(LSTM OFF)
ncnn_add_layer(BinaryOp)
ncnn_add_layer(UnaryOp)
ncnn_add_layer(ConvolutionDepthWise)
ncnn_add_layer(Padding)
ncnn_add_layer(Squeeze)
ncnn_add_layer(ExpandDims)
ncnn_add_layer(Normalize)
ncnn_add_layer(Permute)
ncnn_add_layer(PriorBox)
ncnn_add_layer(DetectionOutput)
ncnn_add_layer(Interp)
ncnn_add_layer(DeconvolutionDepthWise)
ncnn_add_layer(ShuffleChannel)
ncnn_add_layer(InstanceNorm)
ncnn_add_layer(Clip)
ncnn_add_layer(Reorg)
ncnn_add_layer(YoloDetectionOutput)
ncnn_add_layer(Quantize)
ncnn_add_layer(Dequantize)
ncnn_add_layer(Yolov3DetectionOutput)
ncnn_add_layer(PSROIPooling)
ncnn_add_layer(ROIAlign OFF)

# create new
configure_file(layer_declaration.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
configure_file(layer_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
configure_file(layer_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)

if(NCNN_TEEC)
	list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/teec/teec.cpp")
endif()

add_library(ncnn STATIC ${ncnn_SRCS})
# link teec lib
include_directories(../ta/include
				 	../include)
target_link_libraries(ncnn teec)

if(NCNN_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(ncnn PUBLIC OpenMP::OpenMP_CXX)
endif()

if(COVERAGE)
    target_compile_options(ncnn PRIVATE --coverage)
endif()

#install(TARGETS ncnn ARCHIVE DESTINATION lib)
#install(FILES
#    allocator.h
#    blob.h
#    cpu.h
#    layer.h
#    layer_type.h
#    mat.h
#    modelbin.h
#    net.h
#    opencv.h
#    paramdict.h
#    benchmark.h
#    ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
#    ${CMAKE_CURRENT_BINARY_DIR}/platform.h
#    DESTINATION include
#)
