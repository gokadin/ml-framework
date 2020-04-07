package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/google/uuid"
	"runtime"
)

//#cgo CFLAGS: -I.
//#include <tensor.cpp>
import "C"

type Tensor struct {
	id                string
	name              string
	op                op
	mat               *mat.Mat32f
	grad              *mat.Mat32f
	isGradientEnabled bool
	_data              []C.float
	_grad              []C.float
	_tensor           *C.TENSOR
}

func Variable(shape mat.Shape) *Tensor {
	t := &Tensor {
		id:   uuid.New().String(),
		mat:  mat.NewMat32f(shape, nil),
		_data: make([]C.float, shape.X * shape.Y),
	}

	t._tensor = C.alloc_tensor()
	t._tensor.data = &t._data[0]
	t._tensor.shapeX = C.int(shape.X)
	t._tensor.shapeY = C.int(shape.Y)

	runtime.SetFinalizer(t, free)

	return t
}

func free(t *Tensor) {
	C.free_tensor(t._tensor)
}

func (t *Tensor) Id() string {
	return t.id
}

func (t *Tensor) SetData(data []float32) *Tensor {
	t.mat = mat.NewMat32f(t.Shape(), data)
	for i := 0; i < len(data); i++ {
		t._data[i] = C.float(data[i])
	}
	return t
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) SetName(name string) *Tensor {
	t.name = name
	return t
}

func (t *Tensor) Data() *mat.Mat32f {
	return t.mat
}

func (t *Tensor) TempData() []float32 {
	result := make([]float32, t.Shape().X * t.Shape().Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return result
}

func (t *Tensor) Gradient() *mat.Mat32f {
	return t.grad
}

func (t *Tensor) Reduce(grad *mat.Mat32f) {
	t.mat.Sub(grad)
}

func (t *Tensor) Shape() mat.Shape {
	return t.mat.Shape()
}

func (t *Tensor) Reshape(shape mat.Shape) {
	t.mat.Reshape(shape)
}

func (t *Tensor) forward() {
	t.op.forward(t)
}

func (t *Tensor) backward() {
	t.op.backward(t)
}
