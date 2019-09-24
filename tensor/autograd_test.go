package tensor

import (
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
    "testing"
)

/** GENERAL **/

type AutogradTestSuite struct {
    suite.Suite
    autograd *Autograd
}

func (s *AutogradTestSuite) SetupTest() {
    s.autograd = NewAutograd()
}

func TestAutogradTestSuite(t *testing.T) {
    suite.Run(t, new(AutogradTestSuite))
}

/* DERIVATIVES */

func (s *AutogradTestSuite) Test_Autograd_Differentiate_Add() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    e := Add(a, b)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_Add_chain() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Pow(a, 2)
    e := Add(c, b)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{2}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_Sub() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    e := Sub(a, b)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_SubAndDeriveB() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    e := Sub(a, b)

    grad := s.autograd.Derivative(b, e)

    assert.Equal(s.T(), [][]float64{{-1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_Sub_chain() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Pow(b, 2)
    e := Sub(a, c)

    grad := s.autograd.Derivative(b, e)

    assert.Equal(s.T(), [][]float64{{-4}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_power() {
    a := NewTensor([][]float64{{2}})
    e := Pow(a, 2)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{4}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_powerHigherThanTwo() {
    a := NewTensor([][]float64{{2}})
    e := Pow(a, 4)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{32}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_power_chain() {
    a := NewTensor([][]float64{{2}})
    b := Pow(a, 2)
    e := Pow(b, 2)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{32}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_divScalar() {
    a := NewTensor([][]float64{{2}})
    e := DivScalar(a, 2)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{0.5}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_divScalar_chain() {
    a := NewTensor([][]float64{{2}})
    b := Pow(a, 2)
    e := DivScalar(b, 2)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{2}}, grad)
}

func Test_Autograd_Differentiate_sum_0(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}, {1, 1}})
    e := Sum(a, 0)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate(generateIdentityGradient(e.mat))

    assert.Equal(t, [][]float64{{3, 3}, {3, 3}, {3, 3}}, grad)
}

func Test_Autograd_Differentiate_sum_0_chain(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}, {1, 1}})
    b := Pow(a, 2)
    e := Sum(b, 0)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate(generateIdentityGradient(e.mat))

    assert.Equal(t, [][]float64{{6, 6}, {6, 6}, {6, 6}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_dotForA() {
    a := NewTensor([][]float64{{1, 2}, {2, 1}})
    b := NewTensor([][]float64{{0, 3}, {1, 1}})
    e := Dot(a, b)

    grad := s.autograd.Derivative(a, e)

    assert.Equal(s.T(), [][]float64{{3, 2}, {3, 2}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Differentiate_dotForB() {
    a := NewTensor([][]float64{{1, 2}, {2, 1}})
    b := NewTensor([][]float64{{0, 3}, {1, 1}})
    e := Dot(a, b)

    grad := s.autograd.Derivative(b, e)

    assert.Equal(s.T(), [][]float64{{3, 3}, {3, 3}}, grad)
}

/* DERIVATIVE COMBINATIONS */

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativePow() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Add(a, b)
    e := Pow(c, 2)

	grad := s.autograd.Derivative(b, e)

    assert.Equal(s.T(), [][]float64{{6}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativePowComplicated() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    d := NewTensor([][]float64{{3}})
    c := Sub(b, d)
    f := Pow(c, 3)
    e := Sub(a, f)

    grad := s.autograd.Derivative(d, e)

    assert.Equal(s.T(), [][]float64{{3}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativeErrorSimple() {
    y := NewTensor([][]float64{{1}})
    yHat := NewTensor([][]float64{{2}})
    e := DivScalar(Sum(Pow(Sub(y, yHat), 2), 0), 2)

    grad := s.autograd.Derivative(y, e)

    assert.Equal(s.T(), [][]float64{{-1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativeErrorSimpleMultipleValues() {
    y := NewTensor([][]float64{{1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := DivScalar(Sum(Pow(Sub(y, yHat), 2), 0), 2)

    grad := s.autograd.Derivative(y, e)

    assert.Equal(s.T(), [][]float64{{-1, -1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativeErrorFullForWWithSingleValue() {
    x := NewTensor([][]float64{{1}})
    w := NewTensor([][]float64{{1}})
    yHat := NewTensor([][]float64{{2}})
    e := DivScalar(Sum(Pow(Sub(Dot(x, w), yHat), 2), 0), 2)

    grad := s.autograd.Derivative(w, e)

    assert.Equal(s.T(), [][]float64{{-1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativeErrorFullForW() {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := DivScalar(Sum(Pow(Sub(Dot(x, w), yHat), 2), 0), 2)

    grad := s.autograd.Derivative(w, e)

    assert.Equal(s.T(), [][]float64{{0, 0}, {-1, -1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativeOfDotAndSub() {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := Sub(Dot(x, w), yHat)

    grad := s.autograd.Derivative(w, e)

    assert.Equal(s.T(), [][]float64{{0, 0}, {1, 1}}, grad)
}

func (s *AutogradTestSuite) Test_Autograd_Gradient_DerivativeOfDotSubAndPow() {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := Pow(Sub(Dot(x, w), yHat), 2)

    grad := s.autograd.Derivative(w, e)

    assert.Equal(s.T(), [][]float64{{0, 0}, {-2, -2}}, grad)
}

/* CACHE */

func (s *AutogradTestSuite) Test_Cache_createsAnEntryForAGeneratedGraph() {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := Pow(Sub(Dot(x, w), yHat), 2)

    s.autograd.Derivative(w, e)

    assert.Equal(s.T(), 1, len(s.autograd.cache))
    hash := calculateDerivativeHash(w, e)
	_, ok := s.autograd.cache[hash]
    assert.True(s.T(), ok)
}

func (s *AutogradTestSuite) Test_Cache_noNewCacheIsCreatedWhenCallingTheSameDerivativeTwice() {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := Pow(Sub(Dot(x, w), yHat), 2)

    s.autograd.Derivative(w, e)
    s.autograd.Derivative(w, e)

    assert.Equal(s.T(), 1, len(s.autograd.cache))
}

func (s *AutogradTestSuite) Test_Cache_thereAreTwoCacheEntriesForTwoDifferentDerivatives() {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := Pow(Sub(Dot(x, w), yHat), 2)

    s.autograd.Derivative(w, e)
    s.autograd.Derivative(x, e)

    assert.Equal(s.T(), 2, len(s.autograd.cache))
}

/* GRAPH OPTIMIZATION */

func (s *AutogradTestSuite) Test_optimization_removesAddOperations() {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Pow(b, 2)
    e := Add(c, a)

    s.autograd.Derivative(b, e)

    graph := s.autograd.getDerivativeGraph(b, e)
    assert.Equal(s.T(), 1, len(graph))
}