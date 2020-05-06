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
	//C.forward(tensor._tensor)
}

func (o *opAdd) backward(tensor *Tensor) {
	//oa.a.SetGradient(tensor.GradientToFloat32())
	//oa.b.SetGradient(tensor.GradientToFloat32())
}

func Add(a, b *Tensor) *Tensor {
	result := OfShape(a.Shape().ToArray()...)
	result.op = &opAdd{a, b}
	result._tensor.op = C.alloc_add(a._tensor, b._tensor)
	return result
}
