package tensor

import "github.com/gokadin/ml-framework/mat"

const operationSub = "opSub"

type opSub struct {
	a, b *Tensor
}

func (o *opSub) name() string {
	return operationSub
}

func (o *opSub) dependencies() []*Tensor {
	return []*Tensor{o.a, o.b}
}

func (o *opSub) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opSub) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opSub) forward(tensor *Tensor) {
	tensor.SetData(mat.Sub(o.a.ToMat32f(), o.b.ToMat32f()).Data())
}

func (o *opSub) backward(tensor *Tensor) {
	if o.a.isGradientEnabled {
		o.a.SetGradient(tensor.GradientToFloat32())
	}

	if o.b.isGradientEnabled {
		o.b.SetGradient(mat.Neg(tensor.GradientToMat32()).Data())
	}
}

func Sub(a, b *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opSub{a, b}
	return result
}
