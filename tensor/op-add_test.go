package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_add_forward(t *testing.T) {
	a := Variable(mat.WithShape(1, 2)).SetData([]float32{1, 2})
	b := Variable(mat.WithShape(1, 2)).SetData([]float32{2, 3})
	c := Add(a, b)

	c.forward()

	assert.Equal(t, []float32{3, 5}, c.TempData())
}

func Test_add_forward_multipleAssociations(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 3})
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{2, 3, 1, 2})
	c := Add(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 5, 3, 5}).Equals32f(c.mat))
}

func Test_add_backward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 3})
	a.isGradientEnabled = true
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{2, 3, 1, 2})
	b.isGradientEnabled = true
	c := Add(a, b)
	c.forward()
	c.grad = mat.NewMat32fOnes(c.mat.Shape())

	c.backward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{1, 1, 1, 1}).Equals32f(a.grad))
	assert.True(t, mat.NewMat32f(a.Shape(), []float32{1, 1, 1, 1}).Equals32f(b.grad))
}

func Test_add_backward_isGradientsAreDisabled(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 3})
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{2, 3, 1, 2})
	c := Add(a, b)
	c.forward()
	c.grad = mat.NewMat32fOnes(c.mat.Shape())

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
