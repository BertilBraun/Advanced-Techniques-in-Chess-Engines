from __future__ import annotations
import os

from src.mcts.MCTSNode import MCTSNode


def draw_mcts_graph(root: MCTSNode) -> None:
    with open('mcts_graph.dot', 'w') as f:
        f.write('digraph G {\n')
        f.write('node [shape=box];\n')

        def draw_node(node: MCTSNode) -> None:
            f.write(f'"{repr(node)}" [label="{repr(node)}"];\n')
            for child in node.children:
                f.write(f'"{repr(node)}" -> "{repr(child)}";\n')
                draw_node(child)

        draw_node(root)
        f.write('}\n')
    os.system('dot -Tpng mcts_graph.dot -o mcts_graph.png')
