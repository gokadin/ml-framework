package modules

import (
	"log"
	"ml-framework/tensor"
)

const (
	LossMeanSquared         = "LossMeanSquared"
	LossSoftmaxCrossEntropy = "LossSoftmaxCrossEntropy"
)

type Criterion interface {
	Forward(pred, target *tensor.Tensor) *tensor.Tensor
}

func NewCriterion(loss string) Criterion {
	switch loss {
	case LossMeanSquared:
		return newMeanSquaredCriterion()
	case LossSoftmaxCrossEntropy:
		return newCrossEntropyCriterion()
	}

	log.Fatalf("unknown loss function selected: %s", loss)
	return nil
}

type meanSquaredCriterion struct{}

func newMeanSquaredCriterion() *meanSquaredCriterion {
	return &meanSquaredCriterion{}
}

func (msc *meanSquaredCriterion) Forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.DivScalar(tensor.Sum(tensor.Pow(tensor.Sub(pred, target), 2), 0), float32(pred.Shape().X))
}

type crossEntropyCriterion struct{}

func newCrossEntropyCriterion() *crossEntropyCriterion {
	return &crossEntropyCriterion{}
}

func (cec *crossEntropyCriterion) Forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.SoftmaxCrossEntropy(pred, target)
}
