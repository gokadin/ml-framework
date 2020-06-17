package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

type sigmoidTestCases struct {
	name string
	a *Tensor
	runOnGpu bool
}

func buildSigmoidTestCases() []sigmoidTestCases {
	return []sigmoidTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), false},
		{"1200x2 ones GPU", From(InitRandom, 1200, 2), true},
		{"1200x2 ones CPU", From(InitRandom, 1200, 2), false},
		{"1200x2 zeros GPU", Zeros(1200, 2), true},
		{"1200x2 zeros CPU", Zeros(1200, 2), false},
		{"2x1200 ones GPU", From(InitRandom, 2, 1200), true},
		{"2x1200 ones CPU", From(InitRandom, 2, 1200), false},
	}
}

func Test_sigmoid_forward(t *testing.T) {
	for _, test := range buildSigmoidTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Apply(test.a.ToMat32f(), func(value float32) float32 {
				return float32(1 / (math.Exp(-float64(value)) + 1))
			}).Data()
			c := Sigmoid(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_sigmoid_backward(t *testing.T) {
	for _, test := range buildSigmoidTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Sigmoid(test.a)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Random32f(c.Shape().Size()))
			expectedGrad := mat.Mul(c.GradientToMat32(), mat.Mul(c.ToMat32f(), mat.SubFromScalar(c.ToMat32f(), 1))).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedGrad, test.a.GradientToFloat32(), 0.0001)
		})
	}
}

