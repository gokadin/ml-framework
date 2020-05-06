package tensor

//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -ladd
//#include <add.h>
import "C"

const operationAdd = "opAdd"

type opAdd struct {
	a, b *Tensor
}

func (o *opAdd) name() string {
	return operationAdd
}

func (o *opAdd) dependencies() []*Tensor {
	return []*Tensor{o.a, o.b}
}

func (o *opAdd) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opAdd) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opAdd) forward(tensor *Tensor) {
	if !o.a.Shape().Equals(o.b.Shape()) {
		handleIncompatibleShapes("add", o.a.Shape(), o.b.Shape())
	}
	C.add_forward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func (o *opAdd) backward(tensor *Tensor) {
	C.add_backward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func Add(a, b *Tensor) *Tensor {
	o := &opAdd{a, b}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
