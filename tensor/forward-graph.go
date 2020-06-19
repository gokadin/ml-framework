package tensor

type forwardGraph struct {
	startChannels []chan bool
	done          chan bool
	kill          chan bool
}

func buildForwardGraph(tensor *Tensor) *forwardGraph {
	fg := &forwardGraph{
		startChannels: make([]chan bool, 0),
		done:          make(chan bool),
		kill:          make(chan bool),
	}

	fg.build(tensor, fg.done)

	return fg
}

func (fg *forwardGraph) build(tensor *Tensor, lastChan chan bool) {
	c := make(chan bool)

	activeDependencyCount := 0
	for _, dependency := range tensor.op.dependencies() {
		if dependency.op != nil {
			activeDependencyCount++
			fg.build(dependency, c)
		}
	}

	if activeDependencyCount == 0 {
		activeDependencyCount = 1
		fg.startChannels = append(fg.startChannels, c)
	}

	go executeForwardOp(tensor, c, lastChan, fg.kill, activeDependencyCount)
}

func (fg *forwardGraph) run() {
	for _, startChannel := range fg.startChannels {
		startChannel <- true
	}

	<-fg.done
}

func (fg *forwardGraph) close() {
	close(fg.kill)
}

func executeForwardOp(tensor *Tensor, in, out, kill chan bool, threshold int) {
	counter := 0
	for {
		select {
		case <-in:
			counter++
			if counter != threshold {
				break
			}

			counter = 0
			if !tensor.ready {
				tensor.forward()
			}
			out <- true
		case <-kill:
			return
		}
	}
}
