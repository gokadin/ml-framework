package tensor

import "log"

type Autograd struct {
	cache map[string][]*operation
}

func NewAutograd() *Autograd {
	return &Autograd{
		cache: make(map[string][]*operation),
	}
}

func (a *Autograd) Derivative(c, f *Tensor) [][]float64 {
	return computeDerivative(a.getDerivativeGraph(c, f))
}

func (a *Autograd) getDerivativeGraph(c, f *Tensor) []*operation {
	hash := calculateDerivativeHash(c, f)
	if _, ok := a.cache[hash]; ok {
		return a.cache[hash]
	}
	
    a.cache[hash] = createDerivativeGraph(c.operation, f.operation)
    return a.cache[hash]
}

func calculateDerivativeHash(c, f *Tensor) string {
	return c.id + f.id
}

func createDerivativeGraph(c, f *operation) []*operation {
    graph := make([]*operation, 0)
    
    current := f
	for !current.isLeaf() {
		graph = append(graph, current)
		for _, child := range current.children {
			if child.tensor.id == c.tensor.id {
				current = child
				break
			}
		}
	}
}

func findThatSpecialChild(c *operation, children []*operation) int {
    for i, child := range children {
        if leafIsInPath(c, child) {
        	return i
		}
	}
    
    log.Fatal("no child was special")
    return 0
}

func leafIsInPath(c, root *operation) bool {
    
}

func findComponentOperation(leaf, root *operation) *operation {
    var marked *operation
	for _, child := range root.children {
		if child.tensor.id == leaf.tensor.id {
            marked = child
		}
	}
	
	

	if marked == nil {
        log.Fatalf("did not find component")
	}
}

func optimize(f *operation) *operation {
	
}

func computeDerivative(graph []*operation) [][]float64 {

}

