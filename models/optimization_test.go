package models

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
	"github.com/stretchr/testify/assert"
	"testing"
)

const (
	someNonDefaultLearningRate = 5.0
	someNonDefaultMomentum = 4.0
	someNonDefaultBeta1 = 6.0
	someNonDefaultBeta2 = 7.0
	someNonDefaultEpsStable = 8.0
	someBatchSize = 3
)

func Test_defaultOptimizer_initializesWithDefaultLearningRate(t *testing.T) {
	o := newDefaultOptimizer([]float64{})

	assert.Equal(t, defaultLearningRate, o.learningRate)
}

func Test_defaultOptimizer_correctlyOverridesLearningRate(t *testing.T) {
	o := newDefaultOptimizer([]float64{someNonDefaultLearningRate})

	assert.Equal(t, someNonDefaultLearningRate, o.learningRate)
}

func Test_defaultOptimizer_update(t *testing.T) {
	o := newDefaultOptimizer([]float64{})
	t1 := tensor.NewTensor([][]float64{{1, 2}})
	t2 := tensor.NewTensor([][]float64{{1, 2}})
	t3 := tensor.Add(t1, t2)
	t3.Backward()
	expected := mat.Sub(t1.Data(), mat.MulScalar(t1.Gradient(), o.learningRate / float64(someBatchSize)))

	o.update(t1, "", someBatchSize, 0)

	assert.Equal(t, expected, t1.Data())
}

func Test_momentumOptimizer_initializesWithDefaultLearningRate(t *testing.T) {
	o := newMomentumOptimizer([]float64{})

	assert.Equal(t, defaultLearningRate, o.learningRate)
}

func Test_momentumOptimizer_initializesWithDefaultMomentum(t *testing.T) {
	o := newMomentumOptimizer([]float64{})

	assert.Equal(t, defaultMomentum, o.momentum)
}

func Test_momentumOptimizer_initializesAnEmptyVelocityMap(t *testing.T) {
	o := newMomentumOptimizer([]float64{})

	assert.Equal(t, make(map[string][][]float64), o.velocityMap)
}

func Test_momentumOptimizer_correctlyOverridesDefaultLearningRate(t *testing.T) {
	o := newMomentumOptimizer([]float64{someNonDefaultLearningRate})

	assert.Equal(t, someNonDefaultLearningRate, o.learningRate)
}

func Test_momentumOptimizer_correctlyOverridesDefaultMomentum(t *testing.T) {
	o := newMomentumOptimizer([]float64{someNonDefaultLearningRate, someNonDefaultMomentum})

	assert.Equal(t, someNonDefaultMomentum, o.momentum)
}

func Test_adamOptimizer_initializesWithADefaultLearningRate(t *testing.T) {
	o := newAdamOptimizer([]float64{})

	assert.Equal(t, defaultLearningRate, o.learningRate)
}

func Test_adamOptimizer_initializesWithADefaultBeta1(t *testing.T) {
	o := newAdamOptimizer([]float64{})

	assert.Equal(t, defaultBeta1, o.beta1)
}

func Test_adamOptimizer_initializesWithADefaultBeta2(t *testing.T) {
	o := newAdamOptimizer([]float64{})

	assert.Equal(t, defaultBeta2, o.beta2)
}

func Test_adamOptimizer_initializesWithADefaultEpsStable(t *testing.T) {
	o := newAdamOptimizer([]float64{})

	assert.Equal(t, defaultEpsStable, o.epsStable)
}

func Test_adamOptimizer_initializesAnEmptyVelocityMap(t *testing.T) {
	o := newAdamOptimizer([]float64{})

	assert.Equal(t, make(map[string][][]float64), o.velocityMap)
}

func Test_adamOptimizer_initializesAnEmptyMeanMap(t *testing.T) {
	o := newAdamOptimizer([]float64{})

	assert.Equal(t, make(map[string][][]float64), o.meanMap)
}

func Test_adamOptimizer_correctlyOverridesDefaultLearningRate(t *testing.T) {
	o := newAdamOptimizer([]float64{someNonDefaultLearningRate})

	assert.Equal(t, someNonDefaultLearningRate, o.learningRate)
}

func Test_adamOptimizer_correctlyOverridesDefaultBeta1(t *testing.T) {
	o := newAdamOptimizer([]float64{someNonDefaultLearningRate, someNonDefaultBeta1})

	assert.Equal(t, someNonDefaultBeta1, o.beta1)
}

func Test_adamOptimizer_correctlyOverridesDefaultBeta2(t *testing.T) {
	o := newAdamOptimizer([]float64{someNonDefaultLearningRate, someNonDefaultBeta1, someNonDefaultBeta2})

	assert.Equal(t, someNonDefaultBeta2, o.beta2)
}

func Test_adamOptimizer_correctlyOverridesDefaultEpsStable(t *testing.T) {
	o := newAdamOptimizer([]float64{someNonDefaultLearningRate, someNonDefaultBeta1, someNonDefaultBeta2, someNonDefaultEpsStable})

	assert.Equal(t, someNonDefaultEpsStable, o.epsStable)
}
