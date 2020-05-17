package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationMulScalar = "opMulScalar"

type opMulScalar struct {
	a *Tensor
	scalar float32
}

func (o *opMulScalar) name() string {
	return operationMulScalar
}

func (o *opMulScalar) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opMulScalar) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opMulScalar) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opMulScalar) forward(tensor *Tensor) {
	tensor.SetData(mat.MulScalar(o.a.ToMat32f(), o.scalar).Data())
}

func (o *opMulScalar) backward(tensor *Tensor) {
	o.a.SetGradient(mat.MulScalar(tensor.GradientToMat32(), o.scalar).Data())
}

func MulScalar(a *Tensor, scalar float32) *Tensor {
	o := &opMulScalar{a, scalar}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
