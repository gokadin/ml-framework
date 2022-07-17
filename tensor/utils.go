package tensor

import "C"

func fromNativeArrayF32(nativeArray []C.float) []float32 {
	result := make([]float32, len(nativeArray))
	for i := 0; i < len(result); i++ {
		result[i] = float32(nativeArray[i])
	}
	return result
}

func fromNativeArrayInt(nativeArray []C.int) []int {
	result := make([]int, len(nativeArray))
	for i := 0; i < len(result); i++ {
		result[i] = int(nativeArray[i])
	}
	return result
}

func toNativeArrayInt(arr []int) []C.int {
	result := make([]C.int, len(arr))
	for i := 0; i < len(result); i++ {
		result[i] = C.int(arr[i])
	}
	return result
}
