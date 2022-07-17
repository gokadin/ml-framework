package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type powTestCases struct {
	name     string
	a        *Tensor
	power    float32
	runOnGpu bool
}

func buildPowTestCases() []powTestCases {
	return []powTestCases{
		{"1x1 ^1 GPU", From(mat.InitRandom, 1, 1), 1, true},
		{"1x1 ^1 CPU", From(mat.InitRandom, 1, 1), 1, false},
		{"2x2 ^1 GPU", From(mat.InitRandom, 2, 2), 1, true},
		{"2x2 ^1 CPU", From(mat.InitRandom, 2, 2), 1, false},
		{"1x1 ^2 GPU", From(mat.InitRandom, 1, 1), 2, true},
		{"1x1 ^2 CPU", From(mat.InitRandom, 1, 1), 2, false},
		{"2x2 ^2 GPU", From(mat.InitRandom, 2, 2), 2, true},
		{"2x2 ^2 CPU", From(mat.InitRandom, 2, 2), 2, false},
		{"1x1 ^3 GPU", From(mat.InitRandom, 1, 1), 3, true},
		{"1x1 ^3 CPU", From(mat.InitRandom, 1, 1), 3, false},
		{"2x2 ^3 GPU", From(mat.InitRandom, 2, 2), 3, true},
		{"2x2 ^3 CPU", From(mat.InitRandom, 2, 2), 3, false},
	}
}

func Test_pow_forward(t *testing.T) {
	for _, test := range buildPowTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Pow(test.a.ToMat32f(), float64(test.power)).Data()
			c := Pow(test.a, test.power)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_pow_backward(t *testing.T) {
	for _, test := range buildPowTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Pow(test.a, test.power)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Random32f(c.Shape().Size()))
			expectedGrad := mat.Mul(c.GradientToMat32(), mat.MulScalar(mat.Pow(test.a.ToMat32f(), float64(test.power)-1), test.power)).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedGrad, test.a.GradientToFloat32(), 0.0001)
		})
	}
}
