package tensor

//#include <linear.h>
import "C"
import "ml-framework/mat"

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

func (o *opLinear) forwardShape() mat.Shape {
	return mat.Dim(o.a.Shape().D[0], o.x.Shape().D[1])
}

func (o *opLinear) backwardShapes(shape mat.Shape) []mat.Shape {
	return []mat.Shape{
		mat.Dim(shape.D[0], o.a.Shape().D[1]),
		mat.Dim(o.x.Shape().D[0], shape.D[1]),
		mat.Dim(1, shape.D[1]),
	}
}

func (o *opLinear) forward(tensor *Tensor) {
	if o.a.Shape().D[1] != o.x.Shape().D[0] || o.b.Shape().D[1] != o.x.Shape().D[1] {
		handleIncompatibleShapes("linear", o.a.Shape(), o.x.Shape(), o.b.Shape())
	}
	C.linear_forward(tensor._tensor, o.a._tensor, o.x._tensor, o.b._tensor)
}

func (o *opLinear) backward(tensor *Tensor) {
	C.linear_backward(tensor._tensor, o.a._tensor, o.x._tensor, o.b._tensor)
}

func Linear(a, x, b *Tensor) *Tensor {
	o := &opLinear{a, x, b}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
