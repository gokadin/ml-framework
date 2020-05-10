package tensor

type Graph struct {
	forwardGraphs map[int]*forwardGraph
	backwardGraphs map[int]*backwardGraph
}

func NewGraph() *Graph {
	return &Graph{
		forwardGraphs: make(map[int]*forwardGraph),
		backwardGraphs: make(map[int]*backwardGraph),
	}
}

func (g *Graph) Forward(tensor *Tensor) {
	if _, ok := g.forwardGraphs[tensor.id]; !ok {
		g.forwardGraphs[tensor.id] = buildForwardGraph(tensor)
	}

	g.forwardGraphs[tensor.id].run()
}

func (g *Graph) Backward(of *Tensor, derivatives ...*Tensor) {
	id := generateMapKey(append(derivatives, of))
	if _, ok := g.backwardGraphs[id]; !ok {
		g.backwardGraphs[id] = buildBackwardGraph(derivatives, of)
	}

	g.backwardGraphs[id].run()
}

func generateMapKey(tensors []*Tensor) int {
	key := 0
	for i := 0; i < len(tensors); i++ {
		partialSum := 0
		for j := 0; j < i + 1; j++ {
			partialSum += tensors[j].id
		}
		key += (partialSum + i) / (i + 1)
	}
	return key
}
