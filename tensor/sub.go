package tensor

import "ml-framework/mat"

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
	if !o.a.Shape().Equals(o.b.Shape()) {
		handleIncompatibleShapes("sub", o.a.Shape(), o.b.Shape())
	}
	tensor.SetData(mat.Sub(o.a.ToMat32f(), o.b.ToMat32f()).Data())
}

func (o *opSub) backward(tensor *Tensor) {
	o.a.SetGradient(tensor.GradientToFloat32())
	o.b.SetGradient(mat.Neg(tensor.GradientToMat32()).Data())
}

func Sub(a, b *Tensor) *Tensor {
	o := &opSub{a, b}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
