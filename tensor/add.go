package tensor

//#include <add.h>
import "C"

const operationAdd = "opAdd"

type opAdd struct {
	a, b *Tensor
}

func (oa *opAdd) name() string {
	return operationAdd
}

func (oa *opAdd) dependencies() []*Tensor {
	return []*Tensor{oa.a, oa.b}
}

func (oa *opAdd) forward(tensor *Tensor) {
	//C.forward(tensor._tensor)
}

func (oa *opAdd) backward(tensor *Tensor) {
	//oa.a.SetGradient(tensor.GradientToFloat32())
	//oa.b.SetGradient(tensor.GradientToFloat32())
}

func Add(a, b *Tensor) *Tensor {
	result := OfShape(a.Shape().ToArray()...)
	result.op = &opAdd{a, b}
	result._tensor.op = C.alloc_add(a._tensor, b._tensor)
	return result
}
