package models

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
	"github.com/stretchr/testify/assert"
	"testing"
)

const (
	someNonDefaultLearningRate float32 = 5.0
	someNonDefaultMomentum float32 = 4.0
	someNonDefaultBeta1 float32 = 6.0
	someNonDefaultBeta2 float32 = 7.0
	someNonDefaultEpsStable float32 = 8.0
	someBatchSize = 3
)

func Test_defaultOptimizer_initializesWithDefaultLearningRate(t *testing.T) {
	o := newDefaultOptimizer([]float32{})

	assert.Equal(t, defaultLearningRate, o.learningRate)
}

func Test_defaultOptimizer_correctlyOverridesLearningRate(t *testing.T) {
	o := newDefaultOptimizer([]float32{someNonDefaultLearningRate})

	assert.Equal(t, someNonDefaultLearningRate, o.learningRate)
}

func Test_defaultOptimizer_update(t *testing.T) {
	o := newDefaultOptimizer([]float32{})
	t1 := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 2), []float32{1, 2}))
	t2 := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 2), []float32{1, 2}))
	t3 := tensor.Add(t1, t2)
	graph := tensor.NewGraph()
	graph.Forward(t3)
	graph.Backward(t3, t1)
	expected := mat.Sub(t1.Data(), mat.MulScalar(t1.Gradient(), o.learningRate / float32(someBatchSize)))

	o.Update(t1, someBatchSize, 0)

	assert.Equal(t, expected, t1.Data())
}

func Test_momentumOptimizer_initializesWithDefaultLearningRate(t *testing.T) {
	o := newMomentumOptimizer([]float32{})

	assert.Equal(t, defaultLearningRate, o.learningRate)
}

func Test_momentumOptimizer_initializesWithDefaultMomentum(t *testing.T) {
	o := newMomentumOptimizer([]float32{})

	assert.Equal(t, defaultMomentum, o.momentum)
}

func Test_momentumOptimizer_initializesAnEmptyVelocityMap(t *testing.T) {
	o := newMomentumOptimizer([]float32{})

	assert.Equal(t, make(map[string]*mat.Mat32f), o.velocityMap)
}

func Test_momentumOptimizer_correctlyOverridesDefaultLearningRate(t *testing.T) {
	o := newMomentumOptimizer([]float32{someNonDefaultLearningRate})

	assert.Equal(t, someNonDefaultLearningRate, o.learningRate)
}

func Test_momentumOptimizer_correctlyOverridesDefaultMomentum(t *testing.T) {
	o := newMomentumOptimizer([]float32{someNonDefaultLearningRate, someNonDefaultMomentum})

	assert.Equal(t, someNonDefaultMomentum, o.momentum)
}

func Test_adamOptimizer_initializesWithADefaultLearningRate(t *testing.T) {
	o := newAdamOptimizer([]float32{})

	assert.Equal(t, defaultLearningRate, o.learningRate)
}

func Test_adamOptimizer_initializesWithADefaultBeta1(t *testing.T) {
	o := newAdamOptimizer([]float32{})

	assert.Equal(t, defaultBeta1, o.beta1)
}

func Test_adamOptimizer_initializesWithADefaultBeta2(t *testing.T) {
	o := newAdamOptimizer([]float32{})

	assert.Equal(t, defaultBeta2, o.beta2)
}

func Test_adamOptimizer_initializesWithADefaultEpsStable(t *testing.T) {
	o := newAdamOptimizer([]float32{})

	assert.Equal(t, defaultEpsStable, o.epsStable)
}

func Test_adamOptimizer_initializesAnEmptyVelocityMap(t *testing.T) {
	o := newAdamOptimizer([]float32{})

	assert.Equal(t, make(map[string]*mat.Mat32f), o.velocityMap)
}

func Test_adamOptimizer_initializesAnEmptyMeanMap(t *testing.T) {
	o := newAdamOptimizer([]float32{})

	assert.Equal(t, make(map[string]*mat.Mat32f), o.meanMap)
}

func Test_adamOptimizer_correctlyOverridesDefaultLearningRate(t *testing.T) {
	o := newAdamOptimizer([]float32{someNonDefaultLearningRate})

	assert.Equal(t, someNonDefaultLearningRate, o.learningRate)
}

func Test_adamOptimizer_correctlyOverridesDefaultBeta1(t *testing.T) {
	o := newAdamOptimizer([]float32{someNonDefaultLearningRate, someNonDefaultBeta1})

	assert.Equal(t, someNonDefaultBeta1, o.beta1)
}

func Test_adamOptimizer_correctlyOverridesDefaultBeta2(t *testing.T) {
	o := newAdamOptimizer([]float32{someNonDefaultLearningRate, someNonDefaultBeta1, someNonDefaultBeta2})

	assert.Equal(t, someNonDefaultBeta2, o.beta2)
}

func Test_adamOptimizer_correctlyOverridesDefaultEpsStable(t *testing.T) {
	o := newAdamOptimizer([]float32{someNonDefaultLearningRate, someNonDefaultBeta1, someNonDefaultBeta2, someNonDefaultEpsStable})

	assert.Equal(t, someNonDefaultEpsStable, o.epsStable)
}
