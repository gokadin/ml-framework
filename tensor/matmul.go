package tensor

//#include <matmul.h>
import "C"

const operationMatmul = "opMatmul"

type opMatmul struct {
	a, b *Tensor
}

func (o *opMatmul) name() string {
	return operationMatmul
}

func (o *opMatmul) dependencies() []*Tensor {
	return []*Tensor{o.a, o.b}
}

func (o *opMatmul) forwardShape() Shape {
	return Shape{o.a.Shape().X, o.b.Shape().Y}
}

func (o *opMatmul) backwardShapes(shape Shape) []Shape {
	return []Shape{
		{shape.X, o.a.Shape().Y},
		{o.b.Shape().X, shape.Y},
	}
}

func (o *opMatmul) forward(tensor *Tensor) {
	if o.a.Shape().Y != o.b.Shape().X {
		handleIncompatibleShapes("linear", o.a.Shape(), o.b.Shape())
	}
	C.matmul_forward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func (o *opMatmul) backward(tensor *Tensor) {
	C.matmul_backward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func Matmul(a, b *Tensor) *Tensor {
	o := &opMatmul{a, b}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
