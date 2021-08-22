package tensor

//import (
//	"ml-framework/mat"
//	"github.com/stretchr/testify/assert"
//	"testing"
//)
//
//func Test_graph(t *testing.T) {
//	a := OfShape(1, 1).SetData([]float32{3})
//	b := OfShape(1, 1).SetData([]float32{4})
//	c := Add(a, b)
//
//	graph := NewGraph()
//
//	graph.Forward(c)
//
//	assert.Equal(t, 7., c.ToFloat64()[0])
//}
//
//func Test_hydra(t *testing.T) {
//	graph := NewGraph()
//	matA := []float32{3, 3, 3, 3}
//	matB := []float32{4, 4, 4, 4}
//	a := OfShape(2, 2).SetData(matA)
//	b := OfShape(2, 2).SetData(matB)
//	c := Pow(Add(a, b), 2)
//	head1 := Pow(Neg(c), 3)
//	head2 := Pow(c, 2)
//
//	graph.Forward(head1)
//	graph.Forward(head2)
//
//	expectedC := mat.Pow(mat.Add(mat.NewMat32f(mat.WithShape(2, 2), matA), mat.NewMat32f(mat.WithShape(2, 2), matB)), 2)
//	expectedHead1 := mat.Pow(mat.Neg(expectedC), 3)
//	expectedHead2 := mat.Pow(expectedC, 2)
//	assert.Equal(t, expectedHead1.Data(), head1.ToFloat32())
//	assert.Equal(t, expectedHead2.Data(), head2.ToFloat32())
//}
//
//func Test_hydraBackward(t *testing.T) {
//	graph := NewGraph()
//	matA := []float32{3, 3, 3, 3}
//	matB := []float32{4, 4, 4, 4}
//	a := OfShape(2, 2).SetData(matA)
//	b := OfShape(2, 2).SetData(matB)
//	c := Pow(Add(a, b), 2)
//	head1 := Pow(Neg(c), 2)
//	head2Start := Pow(c, 2)
//	head2 := Pow(head2Start, 2)
//
//	graph.Forward(head1)
//	graph.Forward(head2)
//
//	expectedC := mat.Pow(mat.Add(mat.NewMat32f(mat.WithShape(2, 2), matA), mat.NewMat32f(mat.WithShape(2, 2), matB)), 2)
//	expectedHead1 := mat.Pow(mat.Neg(expectedC), 2)
//	expectedHead2 := mat.Pow(mat.Pow(expectedC, 2), 2)
//	assert.Equal(t, expectedHead1.Data(), head1.ToFloat32())
//	assert.Equal(t, expectedHead2.Data(), head2.ToFloat32())
//
//	// ..... backward
//
//	graph.Backward(head1, a)
//	graph.Backward(head2, head2Start)
//
//	expectedGradHead2 := mat.MulScalar(head2Start.ToMat32f(), 2).Data()
//	assert.Equal(t, expectedGradHead2, head2Start.GradientToFloat32())
//}
//
////func Test_graph_some(t *testing.T) {
////	graph := NewGraph()
////	a := OfShape(mat.WithShape(1, 1)).SetData([]float32{2})
////	b := OfShape(mat.WithShape(1, 1)).SetData([]float32{5})
////	c := Add(a, b)
////	d := OfShape(mat.WithShape(1, 1)).SetData([]float32{4})
////	e := Add(c, d)
////
////	graph.Build(e)
////
////	assert.True(t, mat.NewMat32f(e.Shape(), []float32{11}).Equals32f(e.mat))
////}
////
////func Test_nonsnese(t *testing.T) {
////	a := OfShape(mat.WithShape(1, 1)).SetData([]float32{2})
////	b := OfShape(mat.WithShape(1, 1)).SetData([]float32{5})
////	c := OfShape(mat.WithShape(1, 1)).SetData([]float32{6})
////	d := OfShape(mat.WithShape(1, 1)).SetData([]float32{4})
////	e := Add(a, b)
////	f := Add(c, d)
////	g := Add(e, f)
////	graph := NewGraph()
////
////	graph.Build(g)
////
////	assert.True(t, mat.NewMat32f(g.Shape(), []float32{17}).Equals32f(g.mat))
////}
////
////func Test_nonsnese2(t *testing.T) {
////	x := OfShape(mat.WithShape(1, 1)).SetData([]float32{1})
////	y := OfShape(mat.WithShape(1, 1)).SetData([]float32{1})
////	a := Add(x, y)
////	b := OfShape(mat.WithShape(1, 1)).SetData([]float32{5})
////	c := OfShape(mat.WithShape(1, 1)).SetData([]float32{6})
////	d := OfShape(mat.WithShape(1, 1)).SetData([]float32{4})
////	e := Add(a, b)
////	f := Sub(c, d)
////	g := Add(e, f)
////	graph := NewGraph()
////
////	graph.Build(g)
////
////	assert.True(t, mat.NewMat32f(g.Shape(), []float32{9}).Equals32f(g.mat))
////}
////
////func Test_nonsnese3(t *testing.T) {
////	x := OfShape(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("x")
////	y := OfShape(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("y")
////	a := Add(x, y).SetName("a")
////	b := OfShape(mat.WithShape(1, 1)).SetData([]float32{5}).SetName("b")
////	c := OfShape(mat.WithShape(1, 1)).SetData([]float32{6}).SetName("c")
////	d := OfShape(mat.WithShape(1, 1)).SetData([]float32{4}).SetName("d")
////	e := Add(a, b).SetName("e")
////	f := Sub(c, d).SetName("f")
////	g := Add(e, f).SetName("g")
////	graph := NewGraph()
////
////	graph.Backward(g, y)
////
////	assert.True(t, mat.NewMat32f(y.Shape(), []float32{1}).Equals32f(y.grad))
////}
////
////func Test_nonsnese4(t *testing.T) {
////	x := OfShape(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("x")
////	y := OfShape(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("y")
////	a := Add(x, y).SetName("a")
////	b := OfShape(mat.WithShape(1, 1)).SetData([]float32{5}).SetName("b")
////	c := OfShape(mat.WithShape(1, 1)).SetData([]float32{6}).SetName("c")
////	d := OfShape(mat.WithShape(1, 1)).SetData([]float32{4}).SetName("d")
////	e := Add(a, b).SetName("e") //7
////	f := Sub(c, d).SetName("f") //2
////	g := Add(e, f).SetName("g") //9
////	h := Pow(g, 3)              //729
////	graph := NewGraph()
////
////	graph.Build(h)
////	graph.Backward(h, y)
////
////	assert.True(t, mat.NewMat32f(h.Shape(), []float32{729}).Equals32f(h.mat))
////	assert.True(t, mat.NewMat32f(y.Shape(), []float32{243}).Equals32f(y.grad))
////
////	graph.Backward(h, x)
////
////	assert.True(t, mat.NewMat32f(x.Shape(), []float32{243}).Equals32f(x.grad))
////}
////
////func Test_nonsnese5(t *testing.T) {
////	a := OfShape(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 1}).SetName("a")
////	b := OfShape(mat.WithShape(2, 2)).SetData([]float32{0, 3, 1, 1}).SetName("b")
////	e := Matmul(a, b).SetName("e")
////	graph := NewGraph()
////	graph.Build(e)
////
////	graph.Backward(e, a)
////
////	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 2, 3, 2}).Equals32f(a.grad))
////}
////
////func Test_nonsnese6(t *testing.T) {
////	a := OfShape(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 1}).SetName("a")
////	e := Sum(a, 0).SetName("e")
////	graph := NewGraph()
////	graph.Build(e)
////
////	graph.Build(e)
////
////	assert.True(t, mat.NewMat32f(mat.WithShape(1, 2), []float32{3, 3}).Equals32f(e.mat))
////}
////
////func Test_nonsnese7(t *testing.T) {
////	a := OfShape(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 1}).SetName("a")
////	b := OfShape(mat.WithShape(2, 2)).SetData([]float32{0, 3, 1, 1}).SetName("b")
////	e := Matmul(a, b).SetName("e")
////	graph := NewGraph()
////	graph.Build(e)
////
////	graph.Backward(e, a)
////
////	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 2, 3, 2}).Equals32f(a.grad))
////
////	graph.Backward(e, a)
////
////	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 2, 3, 2}).Equals32f(a.grad))
////}
////
////func Test_Autograd_Gradient_DerivativeOfDotSubAndPow2(t *testing.T) {
////	x := OfShape(mat.WithShape(1, 2)).SetData([]float32{0, 1}).SetName("x")
////	w := OfShape(mat.WithShape(2, 2)).SetData([]float32{1, 1, 1, 1}).SetName("w")
////	yHat := OfShape(mat.WithShape(1, 2)).SetData([]float32{2, 2}).SetName("yHat")
////	e := Pow(Sub(Matmul(x, w), yHat), 2)
////	graph := NewGraph()
////	graph.Build(e)
////
////	graph.Backward(e, w)
////
////	assert.True(t, mat.NewMat32f(mat.WithShape(1, 2), []float32{1, 1}).Equals32f(e.grad))
////}
