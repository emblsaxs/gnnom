from anytree import Node, PreOrderIter

mLevel = 13

with open('115.kdtree_no_NULL', 'r') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]

root = Node(lines[0].split("\t")[0])
nodes = {}
nodes[root.name] = root

num = len(lines)
for line in lines:
    line = line.split("\t")
    child1 = "".join(line[1]).strip()
    child2 = "".join(line[2]).strip()
    nodes[child1] = Node(child1, parent=nodes[line[0]])
    nodes[child2] = Node(child2, parent=nodes[line[0]])

# print("Done growing the tree.")
names_a = ([node.name for node in PreOrderIter(root, maxlevel=mLevel)])
names_b = ([node.name for node in PreOrderIter(root, maxlevel=mLevel + 1)])

for name in names_b:
    if (name not in names_a):
        print(name)
