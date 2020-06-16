import graphviz

dotFilePath = "/home/xinhos/Desktop/4-dot.dot"
with open(dotFilePath, "r") as file:
    treeGraphContent = file.read()
treeGraph = graphviz.Source(treeGraphContent)
treeGraph.view()