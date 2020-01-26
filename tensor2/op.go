package tensor2

type op interface {
	name() string
	dependencies() []*Tensor
	forward(mat [][]float64)
	backward()
}
