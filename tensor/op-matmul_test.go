package tensor

import (
	"fmt"
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_dot_forward_simple(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 3, 4})
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{5, 6, 7, 8})
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{19, 22, 43, 50}).Equals32f(c.mat))
}

func Test_dot_forward_square(t *testing.T) {
	size := 64
	aMat := make([]float32, size * size)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 1
	}
	bMat := make([]float32, size * size)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 1
	}
	a := Variable(mat.WithShape(size, size)).SetData(aMat)
	b := Variable(mat.WithShape(size, size)).SetData(bMat)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size), aMat), mat.NewMat32f(mat.WithShape(size, size), bMat))
	assert.True(t, expected.Equals32f(c.mat))
}

func Test_dot_forward_squareNotExponentialOfTwo(t *testing.T) {
	size := 50
	aMat := make([]float32, size * size)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 1
	}
	bMat := make([]float32, size * size)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 1
	}
	a := Variable(mat.WithShape(size, size)).SetData(aMat)
	b := Variable(mat.WithShape(size, size)).SetData(bMat)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size), aMat), mat.NewMat32f(mat.WithShape(size, size), bMat))
	assert.True(t, expected.Equals32f(c.mat))
}

func Test_dot_forward_simple2(t *testing.T) {
	size := 4
	size2 := 6
	aMat := make([]float32, size * size2)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 1
	}
	bMat := make([]float32, size2 * size)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 1
	}
	a := Variable(mat.WithShape(size, size2)).SetData(aMat)
	b := Variable(mat.WithShape(size2, size)).SetData(bMat)
	fmt.Println("ax", a.Shape().X)
	fmt.Println("by", b.Shape().Y)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size2), aMat), mat.NewMat32f(mat.WithShape(size2, size), bMat))
	assert.True(t, expected.Equals32f(c.mat))
}

func Test_dot_forward_differentSimple(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 0, 0, 1})
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{4, 1, 2, 2})
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{4, 1, 2, 2}).Equals32f(c.mat))
}

func Test_dot_forward_differentSizes(t *testing.T) {
	a := Variable(mat.WithShape(2, 3)).SetData([]float32{1, 0, 0, 1, 2, 3})
	b := Variable(mat.WithShape(3, 2)).SetData([]float32{4, 1, 2, 2, 0, 1})
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(mat.WithShape(2, 2), []float32{4, 1, 8, 8}).Equals32f(c.mat))
}

func Test_dot_backward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 0, 0, 1})
	a.isGradientEnabled = true
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{4, 1, 2, 2})
	b.isGradientEnabled = true
	c := Matmul(a, b)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.True(t, mat.MatMul(c.grad, mat.Transpose(b.mat)).Equals32f(a.grad))
	assert.True(t, mat.Transpose(mat.MatMul(mat.Transpose(c.grad), a.mat)).Equals32f(b.grad))
}

func Test_dot_backward_isGradientsAreDisabled(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 0, 0, 1})
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{4, 1, 2, 2})
	c := Matmul(a, b)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
