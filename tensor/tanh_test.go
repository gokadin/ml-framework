package tensor

import (
	"github.com/stretchr/testify/assert"
	"math"
	"ml-framework/mat"
	"testing"
)

type tanhTestCases struct {
	name     string
	a        *Tensor
	runOnGpu bool
}

func buildTanhTestCases() []tanhTestCases {
	return []tanhTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), false},
		{"1200x2 ones GPU", From(mat.InitRandom, 1200, 2), true},
		{"1200x2 ones CPU", From(mat.InitRandom, 1200, 2), false},
		{"1200x2 zeros GPU", Zeros(1200, 2), true},
		{"1200x2 zeros CPU", Zeros(1200, 2), false},
		{"2x1200 ones GPU", From(mat.InitRandom, 2, 1200), true},
		{"2x1200 ones CPU", From(mat.InitRandom, 2, 1200), false},
	}
}

func Test_tanh_forward(t *testing.T) {
	for _, test := range buildTanhTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Apply(test.a.ToMat32f(), func(value float32) float32 {
				return float32(math.Tanh(float64(value)))
			}).Data()
			c := Tanh(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_tanh_backward(t *testing.T) {
	for _, test := range buildTanhTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Tanh(test.a)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Random32f(c.Shape().Size()))
			expectedGrad := mat.Mul(c.GradientToMat32(), mat.Apply(c.ToMat32f(), func(value float32) float32 {
				return float32(1 - math.Pow(math.Tanh(float64(value)), 2))
			})).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedGrad, test.a.GradientToFloat32(), 0.0001)
		})
	}
}
