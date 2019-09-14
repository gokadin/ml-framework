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
	return tensor.SumX(tensor.Pow(tensor.Sub(pred, c.target), 2))
	//return tensor.DivScalar(tensor.SumX(tensor.Pow(tensor.Sub(pred, c.target), 2)), 2 * float64(len(pred.Data())))
}
