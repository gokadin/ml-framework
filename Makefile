TENSOR_DIR=tensor
ARCH=compute_61
CODE=sm_61

build:
	go build -o bin/main main.go

run:
	go run main.go

compile:
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libadd.so --shared ${TENSOR_DIR}/add.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libmul.so --shared ${TENSOR_DIR}/mul.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libexpand.so --shared ${TENSOR_DIR}/expand.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libmatmul.so --shared ${TENSOR_DIR}/matmul.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/librelu.so --shared ${TENSOR_DIR}/relu.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libsum.so --shared ${TENSOR_DIR}/sum.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/libsoftmaxcrossentropy.so --shared ${TENSOR_DIR}/softmax-cross-entropy.cu ${TENSOR_DIR}/sum.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -arch=${ARCH} -code=${CODE} -o ${TENSOR_DIR}/liblinear.so --shared ${TENSOR_DIR}/linear.cu
