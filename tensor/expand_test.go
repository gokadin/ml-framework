package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type expandTestCases struct {
	name string
	a *Tensor
	axis int
	copies int
	runOnGpu bool
}

func buildExpandTestCases() []expandTestCases {
	return []expandTestCases{
		{"1x1 axis 0 3 copies GPU", Ones(1, 1), 0, 3, true},
		{"1x1 axis 0 3 copies CPU", Ones(1, 1), 0, 3, false},
		{"1x1 axis 1 3 copies GPU", Ones(1, 1), 1, 3, true},
		{"1x1 axis 1 3 copies CPU", Ones(1, 1), 1, 3, false},
		{"1x2 axis 0 3 copies GPU", Ones(1, 2), 0, 3, true},
		{"1x2 axis 0 3 copies CPU", Ones(1, 2), 0, 3, false},
		{"2x1 axis 1 3 copies GPU", Ones(2, 1), 1, 3, true},
		{"2x1 axis 1 3 copies CPU", Ones(2, 1), 1, 3, false},
		{"1x1200 axis 0 12 copies GPU", Ones(1, 1200), 0, 12, true},
		{"1x1200 axis 0 12 copies CPU", Ones(1, 1200), 0, 12, false},
		{"1200x1 axis 1 12 copies GPU", Ones(1200, 1), 1, 12, true},
		{"1200x1 axis 1 12 copies CPU", Ones(1200, 1), 1, 12, false},
	}
}

func Test_expand_forward(t *testing.T) {
	for _, test := range buildExpandTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Expand(test.a.ToMat32f(), test.axis, test.copies).Data()
			c := Expand(test.a, test.axis, test.copies)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_expand_backward(t *testing.T) {
	for _, test := range buildExpandTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Expand(test.a, test.axis, test.copies)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))
			expectedGrad := mat.Sum(c.GradientToMat32(), test.axis).Data()

			c.backward()

			assert.Equal(t, expectedGrad, test.a.GradientToFloat32())
		})
	}
}
