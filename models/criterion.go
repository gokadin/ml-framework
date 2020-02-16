package models

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
)

const (
	LossMeanSquared = "LossMeanSquared"
	LossCrossEntropy = "LossCrossEntropy"
)

type criterion interface {
	forward(pred, target *tensor.Tensor) *tensor.Tensor
}

func newCriterion(loss string) criterion {
	switch loss {
	case LossMeanSquared:
		return newMeanSquaredCriterion()
	case LossCrossEntropy:
		return newCrossEntropyCriterion()
	}

	log.Fatalf("unknown loss function selected: %s", loss)
	return nil
}

type meanSquaredCriterion struct {}

func newMeanSquaredCriterion() *meanSquaredCriterion {
	return &meanSquaredCriterion{}
}

func (msc *meanSquaredCriterion) forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.DivScalar(tensor.Sum(tensor.Pow(tensor.Sub(pred, target), 2), 0), float32(pred.Shape().X))
}

type crossEntropyCriterion struct {}

func newCrossEntropyCriterion() *crossEntropyCriterion {
	return &crossEntropyCriterion{}
}

func (cec *crossEntropyCriterion) forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.CrossEntropy(pred, target)
}
