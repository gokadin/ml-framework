package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_sub_forward(t *testing.T) {
	a := Variable(mat.WithShape(1, 2)).SetData([]float32{3, 4})
	b := Variable(mat.WithShape(1, 2)).SetData([]float32{1, 2})
	c := Sub(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{2, 2}).Equals32f(c.mat))
}

func Test_sub_backward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{3, 4, 5, 3})
	a.isGradientEnabled = true
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 1, 4})
	b.isGradientEnabled = true
	c := Sub(a, b)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{1, 1, 1, 1}).Equals32f(a.grad))
	assert.True(t, mat.NewMat32f(a.Shape(), []float32{-1, -1, -1, -1}).Equals32f(b.grad))
}

func Test_sub_backward_isGradientsAreDisabled(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{3, 4, 5, 3})
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 1, 4})
	c := Sub(a, b)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
