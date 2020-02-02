package tensor

import (
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func Test_sigmoid_forward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	c := Sigmoid(a)

	c.forward()

	result := make([][]float64, 2)
	for i := range result {
		row := make([]float64, 2)
		for j := range row {
			row[j] = 1 / (math.Exp(-a.mat[i][j]) + 1)
		}
		result[i] = row
	}
	assert.Equal(t, result, c.mat)
}

func Test_sigmoid_backward(t *testing.T) {
	a := Constant([][]float64{{1, 2}, {3, 4}})
	a.isGradientEnabled = true
	c := Sigmoid(a)
	c.grad = generateIdentityGradient(len(c.mat), len(c.mat[0]))
	c.forward()

	c.backward()

	result := make([][]float64, 2)
	for i := range result {
		row := make([]float64, 2)
		for j := range row {
			row[j] = c.mat[i][j] * (1 - c.mat[i][j])
		}
		result[i] = row
	}
	assert.Equal(t, result, a.grad)
}

