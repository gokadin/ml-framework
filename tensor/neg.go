package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationNeg = "opNeg"

type opNeg struct {
	a *Tensor
}

func (o *opNeg) name() string {
	return operationNeg
}

func (o *opNeg) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opNeg) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opNeg) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opNeg) forward(tensor *Tensor) {
	tensor.SetData(mat.Neg(o.a.ToMat32f()).Data())
}

func (o *opNeg) backward(tensor *Tensor) {
	o.a.SetGradient(tensor.GradientToFloat32())
}

func Neg(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opNeg{a}
	return result
}
