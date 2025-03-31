from graphviz import Digraph

class DecisionTreePlotter:
    def __init__(self, tree, feature_names=None, label_names=None) -> None:
        self.tree = tree
        self.feature_names = feature_names
        self.label_names = label_names
        self.graph = Digraph('Decision Tree')

    def _build(self, dt_node):
        if dt_node.children:
            d = self.feature_names[dt_node.feature_index]
            if self.feature_names:
                label = d['name']
            else:
                label = str(dt_node.feature_index)

            self.graph.node(str(id(dt_node)), label = label, shape='box')

            for feature_value, dt_child in dt_node.children.items():
                self._build(dt_child)
                d_value = d.get('value_names')
                if d_value:
                    label = d_value[feature_value]
                else:
                    label = str(feature_value)
                
                self.graph.edge(str(id(dt_node)), str(id(dt_child)), label=label, fontsize='10')
        else:
            if self.label_names:
                label = self.label_names[dt_node.value]
            else:
                label = str(dt_node.value)
            
            self.graph.node(str(id(dt_node)), label=label, shape='')

    def plot(self):
        self._build(self.tree)
        self.graph.view()