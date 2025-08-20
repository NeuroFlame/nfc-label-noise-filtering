import numpy as np

class NodeValues:
	"""
	Class for holding labels and id's
	"""
	def __init__(self):
		self.datas: np.ndarray = []
		self.labels: list = []


class Node:
	"""
	Class for tree Node
	"""
	def __init__(self, value: NodeValues):
		self.value: NodeValues = value
		self.left_node: Node = None
		self.right_node: Node = None
