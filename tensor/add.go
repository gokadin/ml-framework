package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -ladd
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
	tensor.adjustShape(oa.a.shape)
	C.add(oa.a._tensor, oa.b._tensor, tensor._tensor)
}

func (oa *opAdd) backward(tensor *Tensor) {
	if oa.a.isGradientEnabled {
		oa.a.SetGradient(tensor.GradientToFloat32())
	}

	if oa.b.isGradientEnabled {
		oa.b.SetGradient(tensor.GradientToFloat32())
	}
}

func Add(a, b *Tensor) *Tensor {
	result := Variable(a.shape.ToArray()...)
	result.op = &opAdd{a, b}
	return result
}
