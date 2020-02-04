package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_add_forward(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(1, 2),[]float32{1, 2}))
	b := Constant(mat.NewMat32f(mat.WithShape(1, 2), []float32{2, 3}))
	c := Add(a, b)

	c.forward()

	assert.Equal(t, mat.NewMat32f(mat.WithShape(1, 2), []float32{3, 5}), c.mat)
}

func Test_add_forward_multipleAssociations(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 2, 3}))
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{2, 3, 1, 2}))
	c := Add(a, b)

	c.forward()

	assert.Equal(t, [][]float32{{3, 5}, {3, 5}}, c.mat)
}

func Test_add_backward(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 2, 3}))
	a.isGradientEnabled = true
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{2, 3, 1, 2}))
	b.isGradientEnabled = true
	c := Add(a, b)
	c.forward()
	c.grad = mat.NewMat32fOnes(c.mat.Shape())

	c.backward()

	assert.Equal(t, [][]float32{{1, 1}, {1, 1}}, a.grad)
	assert.Equal(t, [][]float32{{1, 1}, {1, 1}}, b.grad)
}

func Test_add_backward_isGradientsAreDisabled(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 2, 2, 3}))
	b := Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{2, 3, 1, 2}))
	c := Add(a, b)
	c.forward()
	c.grad = mat.NewMat32fOnes(c.mat.Shape())

	c.backward()

	assert.Nil(t, a.grad)
	assert.Nil(t, b.grad)
}
