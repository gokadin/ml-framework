package models

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
)

func averageLoss(loss *tensor.Tensor) float64 {
	return mat.Sum(loss.Data(), 1)[0][0] / float64(len(loss.Data()[0]))
}

func accuracy(y, target *tensor.Tensor, validOutputRange float64) float64 {
	accuracyCounter := 0.0
	for i := range y.Data() {
		for j := range y.Data()[i] {
			if y.Data()[i][j] <= target.Data()[i][j] + validOutputRange && y.Data()[i][j] >= target.Data()[i][j] - validOutputRange {
				accuracyCounter++
			}
		}
	}

	return accuracyCounter * 100.0 / float64(len(y.Data()) * len(y.Data()[0]))
}
