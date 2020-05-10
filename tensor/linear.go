package tensor

//#include <linear.h>
import "C"
import "github.com/gokadin/ml-framework/mat"

const operationLinear = "opLinear"

type opLinear struct {
	a, x, b *Tensor
}

func (o *opLinear) name() string {
	return operationLinear
}

func (o *opLinear) dependencies() []*Tensor {
	return []*Tensor{o.a, o.x, o.b}
}

func (o *opLinear) forwardShape() Shape {
	return Shape{o.a.Shape().X, o.x.Shape().Y}
}

func (o *opLinear) backwardShapes(shape Shape) []Shape {
	return []Shape{
		{shape.X, o.a.Shape().Y},
		{o.x.Shape().X, shape.Y},
		{1, shape.Y},
	}
}

func (o *opLinear) forward(tensor *Tensor) {
	if o.a.Shape().Y != o.x.Shape().X || o.b.Shape().Y != o.x.Shape().Y {
		handleIncompatibleShapes("linear", o.a.Shape(), o.x.Shape(), o.b.Shape())
	}
	C.linear_forward(tensor._tensor, o.a._tensor, o.x._tensor, o.b._tensor)
}

func (o *opLinear) backward(tensor *Tensor) {
	tg := tensor.GradientToFloat32()
	o.b.SetGradient(mat.Sum(mat.NewMat32f(mat.WithShape(tensor.Shape().X, tensor.Shape().Y), tensor.GradientToFloat32()), 0).Data())
	_ = tg
	C.linear_backward(tensor._tensor, o.a._tensor, o.x._tensor, o.b._tensor)
}

func Linear(a, x, b *Tensor) *Tensor {
	o := &opLinear{a, x, b}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
