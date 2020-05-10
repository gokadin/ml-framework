package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationPow = "opPow"

type opPow struct {
	a *Tensor
	power float32
}

func (o *opPow) name() string {
	return operationPow
}

func (o *opPow) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opPow) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opPow) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opPow) forward(tensor *Tensor) {
	tensor.SetData(mat.Pow(o.a.ToMat32f(), float64(o.power)).Data())
}

func (o *opPow) backward(tensor *Tensor) {
	if o.power == 2 {
		o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.MulScalar(o.a.ToMat32f(), 2)).Data())
		return
	}
	o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.MulScalar(mat.Pow(o.a.ToMat32f(), float64(o.power) - 1), o.power)).Data())
}

func Pow(a *Tensor, power float32) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opPow{a, power}
	return result
}
