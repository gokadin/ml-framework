package tensor

type op interface {
	name() string
	dependencies() []*Tensor
	forward(tensor *Tensor)
	backward(tensor *Tensor)
}
