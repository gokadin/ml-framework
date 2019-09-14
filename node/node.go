package node

import "github.com/google/uuid"

type Node struct {
	id 			string
	input       float64
	output      float64
	delta       float64
	connections []*connection
}

func NewNode() *Node {
	return &Node{
		id: uuid.New().String(),
		connections: make([]*connection, 0),
	}
}

func (n *Node) Id() string {
	return n.id
}

func (n *Node) ConnectTo(nextNode *Node) {
	n.connections = append(n.connections, newConnection(nextNode))
}

func (n *Node) ConnectToWithWeight(nextNode *Node, weight float64) {
	n.connections = append(n.connections, newConnectionWithWeight(nextNode, weight))
}

func (n *Node) Connections() []*connection {
	return n.connections
}

func (n *Node) Connection(index int) *connection {
	return n.connections[index]
}

func (n *Node) ResetInput() {
	n.input = 0.0
}

func (n *Node) Input() float64 {
	return n.input
}

func (n *Node) Output() float64 {
	return n.output
}

func (n *Node) SetOutput(value float64) {
    n.output = value
}

func (n *Node) SetInput(value float64) {
	n.input = value
}

func (n *Node) AddInput(value float64) {
	n.input += value
}

func (n *Node) SetDelta(delta float64) {
	n.delta = delta
}

func (n *Node) Delta() float64 {
    return n.delta
}

func (n *Node) Activate() {
	for _, connection := range n.connections {
		connection.nextNode.AddInput(n.output * connection.weight)
	}
}
