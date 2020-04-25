TENSOR_DIR=tensor

build:
	go build -o bin/main main.go

run:
	go run main.go

compile:
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/libadd.so --shared ${TENSOR_DIR}/add.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/libmul.so --shared ${TENSOR_DIR}/mul.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/libexpand.so --shared ${TENSOR_DIR}/expand.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/libmatmul.so --shared ${TENSOR_DIR}/matmul.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/librelu.so --shared ${TENSOR_DIR}/relu.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/libsum.so --shared ${TENSOR_DIR}/sum.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/libcrossentropy.so --shared ${TENSOR_DIR}/cross-entropy.cu ${TENSOR_DIR}/sum.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o ${TENSOR_DIR}/liblinear.so --shared ${TENSOR_DIR}/linear.cu
