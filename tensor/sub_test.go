package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_sub_forward(t *testing.T) {
	a := Variable(1, 2).SetData([]float32{3, 4})
	b := Variable(1, 2).SetData([]float32{1, 2})
	c := Sub(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.shape.Y), []float32{2, 2}).Equals32f(c.ToMat32f()))
}

func Test_sub_backward(t *testing.T) {
	a := Variable(2, 2).SetData([]float32{3, 4, 5, 3})
	a.isGradientEnabled = true
	b := Variable(2, 2).SetData([]float32{1, 2, 1, 4})
	b.isGradientEnabled = true
	c := Sub(a, b)
	c.SetGradient(mat.Ones32f(c.Size()))
	c.forward()

	c.backward()

	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.shape.Y), []float32{1, 1, 1, 1}).Equals32f(a.GradientToMat32()))
	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.shape.Y), []float32{-1, -1, -1, -1}).Equals32f(b.GradientToMat32()))
}

