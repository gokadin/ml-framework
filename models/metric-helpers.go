package models

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
)

func averageLoss(loss *tensor.Tensor) float32 {
	return mat.Sum(loss.Data(), 1).At(0) / float32(loss.Shape().Y)
}

func accuracy(y, target *tensor.Tensor, validOutputRange float32) float32 {
	accuracyCounter := 0
	for i := 0; i < y.Shape().X * y.Shape().Y; i++ {
		if y.Data().At(i) <= target.Data().At(i) + validOutputRange && y.Data().At(i) >= target.Data().At(i) - validOutputRange {
			accuracyCounter++
		}
	}

	return float32(accuracyCounter * 100) / float32(y.Shape().X * y.Shape().Y)
}

func accuracyOneHot(y, target *tensor.Tensor, validOutputRange float32) float32 {
	accuracyCounter := 0
	for i := 0; i < y.Shape().X; i++ {
		var maxIndex int
		var maxValue float32
		var targetIndex int
		for j := 0; j < y.Shape().Y; j++ {
			index := i * y.Shape().Y + j
			if y.Data().At(index) > maxValue {
				maxValue = y.Data().At(index)
				maxIndex = j
			}
			if target.Data().At(index) == 1 {
				targetIndex = j
			}
		}
		if maxIndex == targetIndex {
			accuracyCounter += y.Shape().Y
		}
	}

	return float32(accuracyCounter * 100) / float32(y.Shape().X * y.Shape().Y)
}
