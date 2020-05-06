package tensor

type op interface {
	name() string
	dependencies() []*Tensor
	forwardShape() Shape
	backwardShapes(tensorShape Shape) []Shape
	forward(tensor *Tensor)
	backward(tensor *Tensor)
}
