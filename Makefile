TENSOR_DIR=tensor
ARCH=compute_61
CODE=sm_61
OS := $(shell uname)

build:
	go build -o bin/main main.go

run:
	go run main.go

compile:
	nvcc ${TENSOR_DIR}/add.cu -o add.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/mul.cu -o mul.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/expand.cu -o expand.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/matmul.cu -o matmul.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/relu.cu -o relu.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/sum.cu -o sum.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/softmax.cu -o softmax.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/linear.cu ${TENSOR_DIR}/softmax-cross-entropy.cu ${TENSOR_DIR}/matmul.cu ${TENSOR_DIR}/sum.cu -o linear.dll --shared -DCUDADLL_EXPORTS -lcudart
	nvcc ${TENSOR_DIR}/softmax-cross-entropy.cu ${TENSOR_DIR}/sum.cu -o softmaxcrossentropy.dll --shared -DCUDADLL_EXPORTS -lcudart

#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libadd.so --shared ${TENSOR_DIR}/add.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libmul.so --shared ${TENSOR_DIR}/mul.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libexpand.so --shared ${TENSOR_DIR}/expand.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libmatmul.so --shared ${TENSOR_DIR}/matmul.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/librelu.so --shared ${TENSOR_DIR}/relu.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libsum.so --shared ${TENSOR_DIR}/sum.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libsoftmax.so --shared ${TENSOR_DIR}/softmax.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/liblinear.so --shared ${TENSOR_DIR}/linear.cu ${TENSOR_DIR}/softmax-cross-entropy.cu ${TENSOR_DIR}/matmul.cu ${TENSOR_DIR}/sum.cu
#	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libsoftmaxcrossentropy.so --shared ${TENSOR_DIR}/softmax-cross-entropy.cu
