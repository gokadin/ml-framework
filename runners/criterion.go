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
}

func NewCriterion(lossFunctionName string) *Criterion {
	return &Criterion{
		lossFunctionName: lossFunctionName,
	}
}

func (c *Criterion) Forward(pred, target *tensor.Tensor) *tensor.Tensor {
    switch c.lossFunctionName {
	case lossFunctionMeanSquared:
		return c.forwardMeanSquared(pred, target)
	default:
		log.Fatal("loss function is not defined")
		return nil
	}
}

func (c *Criterion) forwardMeanSquared(pred, target *tensor.Tensor) *tensor.Tensor {
	//return tensor.DivScalar(tensor.Sum(pred.Sub(c.target).Pow(2), 0), 2)
	return tensor.DivScalar(tensor.Sum(tensor.Pow(tensor.Sub(pred, target), 2), 0), 2)
}
