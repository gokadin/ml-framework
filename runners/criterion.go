package runners

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
)

const (
	lossFunctionMeanSquared = "lossFunctionMeanSquared"
)

type Criterion struct {
    lossFunctionName string
	target *tensor.Tensor
}

func NewCriterion(lossFunctionName string, target *tensor.Tensor) *Criterion {
	return &Criterion{
		lossFunctionName: lossFunctionName,
		target: target,
	}
}

func (c *Criterion) Forward(pred *tensor.Tensor) *tensor.Tensor {
    switch c.lossFunctionName {
	case lossFunctionMeanSquared:
		return c.forwardMeanSquared(pred)
	default:
		log.Fatal("loss function is not defined")
		return nil
	}
}

func (c *Criterion) forwardMeanSquared(pred *tensor.Tensor) *tensor.Tensor {
	return tensor.DivScalar(tensor.Sum(pred.Sub(c.target).Pow(2), 0), 2)
}
