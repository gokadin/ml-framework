package tensor

//#include <linear.h>
import "C"

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
	return []Shape{} // finish
}

func (o *opLinear) forward(tensor *Tensor) {
	C.linear_forward(tensor._tensor, o.a._tensor, o.x._tensor, o.b._tensor)
}

func (o *opLinear) backward(tensor *Tensor) {
	C.linear_backward(tensor._tensor, o.a._tensor, o.x._tensor, o.b._tensor)
}

func Linear(a, x, b *Tensor) *Tensor {
	result := OfShape(a.Shape().X, b.Shape().Y)
	result.op = &opLinear{a, x, b}
	return result
}
