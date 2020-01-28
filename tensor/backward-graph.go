package tensor

type backwardGraph struct {
	start     chan bool
	done      chan bool
	doneCount int
	root      *Tensor
	graph     map[string]*backwardMapping
}

type backwardMapping struct {
	tensor *Tensor
	in chan bool
	listeners []chan bool
}

func buildBackwardGraph(derivatives []*Tensor, of *Tensor) *backwardGraph {
	bg := &backwardGraph{
		start:     make(chan bool),
		done:      make(chan bool),
		root:      of,
		graph:     make(map[string]*backwardMapping),
	}

	bg.buildGraph(derivatives, of, nil)
	bg.bootGraph()

	return bg
}

func (bg *backwardGraph) buildGraph(derivatives []*Tensor, root, parent *Tensor) {
	if root.op == nil {
		return
	}

	gradientDependencies := make([]string, 0)
	for _, dependency := range root.op.dependencies() {
		if dependency == nil {
			continue
		}

		for _, derivative := range derivatives {
			if dependency.id == derivative.id {
				dependency.isGradientEnabled = true
				break
			}
		}

		bg.buildGraph(derivatives, dependency, root)

		if dependency.isGradientEnabled {
			root.isGradientEnabled = true
			gradientDependencies = append(gradientDependencies, dependency.id)
		}
	}

	if len(gradientDependencies) > 0 {
		listeners := make([]chan bool, len(gradientDependencies))
		for i, gradientDependencyId := range gradientDependencies {
			if mapping, ok := bg.graph[gradientDependencyId]; ok {
				in := make(chan bool)
				mapping.in = in
				listeners[i] = in
				continue
			}
			listeners[i] = bg.done
			bg.doneCount++
		}

		if parent == nil {
			bg.graph[root.id] = &backwardMapping{root, bg.start, listeners}
		} else {
			bg.graph[root.id] = &backwardMapping{root, nil, listeners}
		}
	}
}

func (bg *backwardGraph) bootGraph() {
	for _, mapping := range bg.graph {
		go executeBackwardOp(mapping.tensor, mapping.in, mapping.listeners)
	}
}

func (bg *backwardGraph) run() {
	bg.root.grad = generateIdentityGradient(len(bg.root.mat), len(bg.root.mat[0]))
	bg.start <- true

	doneCounter := 0
	for range bg.done {
		doneCounter++
		if doneCounter == bg.doneCount {
			break
		}
	}
}

func executeBackwardOp(tensor *Tensor, in chan bool, listeners []chan bool) {
	for range in {
		tensor.backward()
		for _, listener := range listeners {
			listener <- true
		}
	}
}

func generateIdentityGradient(shapeX, shapeY int) [][]float64 {
	grad := make([][]float64, shapeX)
	for i := range grad {
		grad[i] = make([]float64, shapeY)
		for j := range grad[i] {
			grad[i][j] = 1
		}
	}
	return grad
}
