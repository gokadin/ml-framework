package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"log"
	"runtime"
)

//#cgo CFLAGS: -I.
//#include <tensor.cpp>
import "C"

var nextId int

type Tensor struct {
	id                int
	name              string
	shape             Shape
	op                op
	isGradientEnabled bool
	ready             bool
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
		id:    nextId,
		shape: shape,
	}

	t._tensor = C.alloc_tensor()
	t.initializeNativeTensor(shape)
	runtime.SetFinalizer(t, free)

	return t
}

func (t *Tensor) initializeNativeTensor(shape Shape) {
	t._data = make([]C.float, shape.X*shape.Y)
	t._grad = make([]C.float, shape.X*shape.Y)

	t._tensor.grad = &t._grad[0]
	t._tensor.data = &t._data[0]
	t._tensor.shapeX = C.int(shape.X)
	t._tensor.shapeY = C.int(shape.Y)
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
	result := make([]float32, t.shape.X*t.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return result
}

func (t *Tensor) ToFloat64() []float64 {
	result := make([]float64, t.shape.X*t.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float64(t._data[i])
	}
	return result
}

func (t *Tensor) ToMat32f() *mat.Mat32f {
	result := make([]float32, t.shape.X*t.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return mat.NewMat32f(mat.WithShape(t.shape.X, t.shape.Y), result)
}

func (t *Tensor) GradientToFloat32() []float32 {
	result := make([]float32, t.shape.X*t.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return result
}

func (t *Tensor) GradientToMat32() *mat.Mat32f {
	result := make([]float32, t.shape.X*t.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return mat.NewMat32f(mat.WithShape(t.shape.X, t.shape.Y), result)
}

func (t *Tensor) Reduce(grad []float32) {
	for i := 0; i < len(t._data); i++ {
		t._data[i] -= C.float(grad[i])
	}
}

func (t *Tensor) Shape() Shape {
	return t.shape
}

func (t *Tensor) Size() int {
	return t.shape.X * t.shape.Y
}

func (t *Tensor) adjustShape(shape Shape) {
	if t.shape.X != shape.X || t.shape.Y != t.shape.Y {
		t.shape = shape
		t.initializeNativeTensor(shape)
	}
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	t.shape = Shape{X: shape[0], Y: shape[1]}
	t.initializeNativeTensor(t.shape)
	return t
}

func (t *Tensor) forward() {
	t.op.forward(t)
	t.ready = true
}

func (t *Tensor) backward() {
	t.op.backward(t)
	t.ready = false
}
