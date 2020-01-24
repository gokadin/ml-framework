package models

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
)

const (
	LossMeanSquared = "LossMeanSquared"
)

type criterion interface {
	forward(pred, target *tensor.Tensor) *tensor.Tensor
}

func newCriterion(loss string) criterion {
	switch loss {
	case LossMeanSquared:
		return newMeanSquaredCriterion()
	}

	log.Fatalf("unknown loss function selected: %s", loss)
	return nil
}

type meanSquaredCriterion struct {}

func newMeanSquaredCriterion() *meanSquaredCriterion {
	return &meanSquaredCriterion{}
}

func (msc *meanSquaredCriterion) forward(pred, target *tensor.Tensor) *tensor.Tensor {
	//return tensor.DivScalar(tensor.Sum(pred.Sub(c.target).Pow(2), 0), 2)
	return tensor.DivScalar(tensor.Sum(tensor.Pow(tensor.Sub(pred, target), 2), 0), 2)
}
