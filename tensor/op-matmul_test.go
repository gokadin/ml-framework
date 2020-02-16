package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_dot_forward_simple(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 3, 4}))
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{5, 6, 7, 8}))
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{19, 22, 43, 50}).Equals32f(c.mat))
}

func Test_dot_forward_differentSimple(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 0, 0, 1}))
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{4, 1, 2, 2}))
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{4, 1, 2, 2}).Equals32f(c.mat))
}

func Test_dot_backward(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 0, 0, 1}))
	a.isGradientEnabled = true
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{4, 1, 2, 2}))
	b.isGradientEnabled = true
	c := Matmul(a, b)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.True(t, mat.MatMul(c.grad, mat.Transpose(b.mat)).Equals32f(a.grad))
	assert.True(t, mat.Transpose(mat.MatMul(mat.Transpose(c.grad), a.mat)).Equals32f(b.grad))
}

func Test_dot_backward_isGradientsAreDisabled(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 0, 0, 1}))
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{4, 1, 2, 2}))
	c := Matmul(a, b)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
