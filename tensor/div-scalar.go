package tensor

import (
	"ml-framework/mat"
)

const operationDivScalar = "opDivScalar"

type opDivScalar struct {
	a      *Tensor
	scalar float32
}

func (o *opDivScalar) name() string {
	return operationDivScalar
}

func (o *opDivScalar) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opDivScalar) forwardShape() mat.Shape {
	return o.a.Shape()
}

func (o *opDivScalar) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape}
}

func (o *opDivScalar) forward(tensor *Tensor) {
	tensor.SetData(mat.DivScalar(o.a.ToMat32f(), o.scalar).Data())
}

func (o *opDivScalar) backward(tensor *Tensor) {
	multiplier := 1.0 / o.scalar
	o.a.SetGradient(mat.MulScalar(tensor.GradientToMat32(), multiplier).Data())
}

func DivScalar(a *Tensor, scalar float32) *Tensor {
	o := &opDivScalar{a, scalar}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
