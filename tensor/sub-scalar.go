package tensor

import (
	"ml-framework/mat"
)

const operationSubScalar = "opSubScalar"

type opSubScalar struct {
	a      *Tensor
	scalar float32
}

func (o *opSubScalar) name() string {
	return operationSubScalar
}

func (o *opSubScalar) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opSubScalar) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opSubScalar) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opSubScalar) forward(tensor *Tensor) {
	tensor.SetData(mat.SubFromScalar(o.a.ToMat32f(), o.scalar).Data())
}

func (o *opSubScalar) backward(tensor *Tensor) {
	o.a.SetGradient(mat.MulScalar(tensor.GradientToMat32(), -o.scalar).Data())
}

func SubFromScalar(a *Tensor, scalar float32) *Tensor {
	o := &opSubScalar{a, scalar}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
