package runners

import (
	"ml-framework/mat"
	"ml-framework/tensor"
)

func averageLoss(loss *tensor.Tensor) float32 {
	return mat.Sum(loss.ToMat32f(), 1).At(0) / float32(loss.Shape().D[1])
}

func accuracy(y, target *tensor.Tensor, validOutputRange float32) float32 {
	accuracyCounter := 0
	yData := y.ToFloat32()
	targetData := target.ToFloat32()
	for i := 0; i < y.Shape().D[0]*y.Shape().D[1]; i++ {
		if yData[i] <= targetData[i]+validOutputRange && yData[i] >= targetData[i]-validOutputRange {
			accuracyCounter++
		}
	}

	return float32(accuracyCounter*100) / float32(y.Shape().D[0]*y.Shape().D[1])
}

func accuracyOneHot(y, target *tensor.Tensor) float32 {
	accuracyCounter := 0
	yData := y.ToFloat32()
	targetData := target.ToFloat32()
	for i := 0; i < y.Shape().D[0]; i++ {
		var maxIndex int
		var maxValue float32
		var targetIndex int
		for j := 0; j < y.Shape().D[1]; j++ {
			index := i*y.Shape().D[1] + j
			if yData[index] > maxValue {
				maxValue = yData[index]
				maxIndex = j
			}
			if targetData[index] == 1 {
				targetIndex = j
			}
		}
		if maxIndex == targetIndex {
			accuracyCounter++
		}
	}

	return float32(accuracyCounter*100) / float32(y.Shape().D[0])
}
