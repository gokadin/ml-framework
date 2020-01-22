package datasets

const trainingSetX = "trainingSetX"
const trainingSetY = "trainingSetY"
const validationSetX = "validationSetX"
const validationSetY = "validationSetY"

type dataset struct {
	name string
	sets map[string]*set
}

func NewDataset() *dataset {
	return &dataset{
		sets: make(map[string]*set),
	}
}

func (d *dataset) SetName(name string) {
	d.name = name
}

func (d *dataset) AddData(name string, data [][]float64) {
	d.sets[name] = &set{data}
}

func (d *dataset) Get(name string) *set {
	return d.sets[name]
}