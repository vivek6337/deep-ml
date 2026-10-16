import math
import numpy as np

class Node:
    def __init__(self, prev_value=None, curr_attr=None, leaf_value=None, branches=None):
        self.prev_value = prev_value
        self.curr_attr = curr_attr
        self.leaf_value = leaf_value
        self.branches = branches if branches is not None else []


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    d = {}
    for label in labels:
        d[label] = d.get(label, 0) + 1

    n = len(labels)
    entropy = 0.0
    for v in d.values():
        p = v / n
        entropy -= p * math.log2(p)
    return entropy


def calculate_information_gain(examples: list[dict], attr: list, target_attr: str):
    """Calculate the information gain of splitting on attributes and find the best one."""
    best_gain = -1.0
    best_attr = None
    best_sorted_examples = None
    best_part_idx = None
    
    t = [i[target_attr] for i in examples]
    base_entropy = calculate_entropy(t)
    
    for att in attr:
        # Sort examples based on the current attribute values
        sorted_examples = sorted(examples, key=lambda x: x[att])
        a_f = [i[att] for i in sorted_examples]
        t_f = [i[target_attr] for i in sorted_examples]
        
        n = len(sorted_examples)
        if n == 0:
            continue
            
        p_idx = [0]
        g = 0.0
        prev_idx = 0
        
        for i in range(1, n):
            if a_f[i] != a_f[i - 1]:
                p_idx.append(i)
                g += ((i - prev_idx) / n) * calculate_entropy(t_f[prev_idx:i])
                prev_idx = i
        p_idx.append(n)
        g += ((n - prev_idx) / n) * calculate_entropy(t_f[prev_idx:])
        
        ig = base_entropy - g
        
        if ig > best_gain:
            best_gain = ig
            best_attr = att
            best_sorted_examples = sorted_examples
            best_part_idx = p_idx

    return best_gain, best_attr, best_sorted_examples, best_part_idx


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    d = {}
    for e in examples:
        val = e[target_attr]
        d[val] = d.get(val, 0) + 1
    return min(d, key=lambda k: (-d[k], k))


def grow_tree(prev_value, examples, attr, target_attr):
    """Recursively grow the decision tree."""
    t = [e[target_attr] for e in examples]
    
    # Base cases
    if calculate_entropy(t) == 0.0:
        return Node(prev_value=prev_value, leaf_value=t[0])
    if len(attr) == 0 or len(examples) == 0:
        return Node(prev_value=prev_value, leaf_value=majority_class(examples, target_attr))
    
    ig, a, sorted_examples, part_idx = calculate_information_gain(examples, attr, target_attr)
    

    node = Node(prev_value=prev_value, curr_attr=a)
    new_attr = [i for i in attr if i != a]

    # Create branches for each partition
    for i in range(len(part_idx) - 1):
        subset = sorted_examples[part_idx[i]:part_idx[i + 1]]
        if subset:
            p_value = subset[0][a]
            child_node = grow_tree(p_value, subset, new_attr, target_attr)
            node.branches.append(child_node)

    return node


def dict_f(node):
    """Convert the Node tree into a nested dictionary representation."""
    if not node.branches:
        return node.leaf_value
    
    child = {}
    for n in node.branches:
        child[n.prev_value] = dict_f(n)

    return {node.curr_attr: child}


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm and return as a nested dict."""
    root = grow_tree(None, examples, attributes, target_attr)
    
    # If the root has no branches (e.g. pure dataset from the start)
    if not root.branches:
        return root.leaf_value
        
    return dict_f(root)