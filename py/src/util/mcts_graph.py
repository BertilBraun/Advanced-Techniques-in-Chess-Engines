from __future__ import annotations
import os

from src.mcts.MCTSNode import MCTSNode


def _node_repr(node: MCTSNode) -> str:
    return f"""{repr(node.board) if node.board else None}
visits: {node.number_of_visits}
score: {node.result_score:.2f}
move: {node.encoded_move_to_get_here}
policy: {node.parent.children_policies[node.my_child_index] if node.parent else 0.0:.2f}"""


def draw_mcts_graph(root: MCTSNode, out_file: str = 'mcts_graph.png') -> None:
    with open('mcts_graph.dot', 'w') as f:
        f.write('digraph G {\n')
        f.write('node [shape=box];\n')

        def draw_node(node: MCTSNode) -> None:
            f.write(f'"{_node_repr(node)}" [label="{_node_repr(node)}"];\n')
            for child in node.children:
                if not child.is_fully_expanded and child.board and not child.is_terminal_node:
                    continue
                f.write(f'"{_node_repr(node)}" -> "{_node_repr(child)}";\n')
                draw_node(child)

        draw_node(root)
        f.write('}\n')
    os.system(f'dot -Tpng mcts_graph.dot -o {out_file}')
