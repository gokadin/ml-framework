package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_pow_forward(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 3, 4}))
	c := Pow(a, 3)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{1, 8, 27, 64}).Equals32f(c.mat))
}

func Test_pow_backward(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 3, 4}))
	a.isGradientEnabled = true
	c := Pow(a, 3)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 12, 27, 48}).Equals32f(a.grad))
}

func Test_pow_backward_forPowerOfTwo(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 3, 4}))
	a.isGradientEnabled = true
	c := Pow(a, 2)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{2, 4, 6, 8}).Equals32f(a.grad))
}
