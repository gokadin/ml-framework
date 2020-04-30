package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"log"
	"runtime"
)

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -ladd
//#include <tensor.h>
import "C"

var nextId int

type Tensor struct {
	id                int
	name              string
	op                op
	isGradientEnabled bool
	ready             bool
	_mat_shape        *C.SHAPE
	_grad_shape       *C.SHAPE
	_data             []C.float
	_grad             []C.float
	_tensor           *C.TENSOR
}

func Variable(shapeArray ...int) *Tensor {
	if len(shapeArray) != 2 {
		log.Fatal("only shapes of 2 dimensions are supported for the moment")
	}
	shape := Shape{X: shapeArray[0], Y: shapeArray[1]}

	nextId++
	t := &Tensor{
		id: nextId,
	}

	t._tensor = C.alloc_tensor(C.int(nextId))
	t.initializeNativeTensor(shape)
	runtime.SetFinalizer(t, free)

	return t
}

func (t *Tensor) initializeNativeTensor(shape Shape) {
	t._data = make([]C.float, shape.X*shape.Y)
	t._grad = make([]C.float, shape.X*shape.Y)
	t._tensor.data = &t._data[0]
	t._tensor.grad = &t._grad[0]

	t._mat_shape = &C.SHAPE{}
	t._grad_shape = &C.SHAPE{}
	t._tensor.mat_shape = t._mat_shape
	t._tensor.grad_shape = t._grad_shape
	t.reshapeMat(shape.ToArray()...)
	t.reshapeGrad(shape.ToArray()...)
}

func (t *Tensor) reshapeMat(shape ...int) {
	t._mat_shape.x = C.int(shape[0])
	t._mat_shape.y = C.int(shape[1])
	t._mat_shape.size = C.int(shape[0] * shape[1])
}

func (t *Tensor) reshapeGrad(shape ...int) {
	t._grad_shape.x = C.int(shape[0])
	t._grad_shape.y = C.int(shape[1])
	t._grad_shape.size = C.int(shape[0] * shape[1])
}

func free(t *Tensor) {
	C.free_tensor(t._tensor)
}

func (t *Tensor) Id() int {
	return t.id
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) SetName(name string) *Tensor {
	t.name = name
	return t
}

func (t *Tensor) RunOnGpu(value bool) {
	t._tensor.run_on_gpu = C.bool(value)
}

func (t *Tensor) SetData(data []float32) *Tensor {
	for i := 0; i < len(data); i++ {
		t._data[i] = C.float(data[i])
	}
	return t
}

func (t *Tensor) Set(index int, value float32) {
	t._data[index] = C.float(value)
}

func (t *Tensor) SetGradient(grad []float32) {
	for i := 0; i < len(grad); i++ {
		t._grad[i] = C.float(grad[i])
	}
}

func (t *Tensor) ToFloat32() []float32 {
	result := make([]float32, t.Shape().Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return result
}

func (t *Tensor) ToFloat64() []float64 {
	result := make([]float64, t._mat_shape.size)
	for i := 0; i < len(result); i++ {
		result[i] = float64(t._data[i])
	}
	return result
}

func (t *Tensor) ToMat32f() *mat.Mat32f {
	result := make([]float32, t._mat_shape.size)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return mat.NewMat32f(mat.WithShape(t.Shape().X, t.Shape().Y), result)
}

func (t *Tensor) GradientToFloat32() []float32 {
	result := make([]float32, t._mat_shape.size)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return result
}

func (t *Tensor) GradientToMat32() *mat.Mat32f {
	result := make([]float32, t.Shape().Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return mat.NewMat32f(mat.WithShape(t.Shape().X, t.Shape().Y), result)
}

func (t *Tensor) Reduce(grad []float32) {
	for i := 0; i < t.Shape().Size(); i++ {
		t._data[i] -= C.float(grad[i])
	}
}

func (t *Tensor) Shape() Shape {
	return Shape{
		X: int(t._mat_shape.x),
		Y: int(t._mat_shape.y),
	}
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	if len(shape) != 2 {
		panic("only 2D shapes are supported")
	}

	t.reshapeMat(shape...)

	t._data = make([]C.float, shape[0]*shape[1])
	t._tensor.data = &t._data[0]

	return t
}

func (t *Tensor) forward() {
	if t.op.name() == operationAdd {
		_shape := C.calculate_op_shape(t._tensor)
		t.Reshape(int(_shape.x), int(_shape.y))
		handleOpResult(int(C.forward(t._tensor)))
	} else {
		t.op.forward(t)
	}
	t.ready = true
}

func (t *Tensor) backward() {
	if t.op.name() == operationAdd {
		handleOpResult(int(C.backward(t._tensor)))
	} else {
		t.op.backward(t)
	}
	t.ready = false
}

