package tensor

import (
	"ml-framework/mat"
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
	return []Shape{tensorShape}
}

func (o *opNeg) forward(tensor *Tensor) {
	tensor.SetData(mat.Neg(o.a.ToMat32f()).Data())
}

func (o *opNeg) backward(tensor *Tensor) {
	o.a.SetGradient(tensor.GradientToFloat32())
}

func Neg(a *Tensor) *Tensor {
	o := &opNeg{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
