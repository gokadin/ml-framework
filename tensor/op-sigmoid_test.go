package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func Test_sigmoid_forward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 3, 4})
	c := Sigmoid(a)

	c.forward()

	result := mat.Apply(a.mat, func(value float32) float32 {
		return float32(1 / (math.Exp(-float64(value)) + 1))
	})
	assert.True(t, result.Equals32f(c.mat))
}

func Test_sigmoid_backward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 3, 4})
	a.isGradientEnabled = true
	c := Sigmoid(a)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	result := mat.Apply(c.mat, func(value float32) float32 {
		return value * (1 - value)
	})
	assert.True(t, result.Equals32f(a.grad))
}

