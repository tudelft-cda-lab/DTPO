import pydot

import numpy as np


def export_tree(
    tree,
    filename,
    feature_names,
    action_names,
    integer_features=None,
    colors=None,
    fontname="helvetica",
    continuous_actions=False,
):
    """
    Visualizes the decision tree and exports it using graphviz.
    """
    dot_string = sklearn_tree_to_graphviz(
        tree,
        feature_names,
        action_names,
        integer_features,
        colors,
        fontname,
        continuous_actions,
    )
    graph = pydot.graph_from_dot_data(dot_string)[0]

    if filename.endswith(".png"):
        graph.write_png(filename)
    elif filename.endswith(".pdf"):
        graph.write_pdf(filename)
    elif filename.endswith(".dot"):
        graph.write_dot(filename)
    else:
        raise ValueError(f"Unkown file extension {filename.split('.')[-1]}")


def sklearn_tree_to_graphviz(
    tree,
    feature_names,
    action_names,
    integer_features=None,
    colors=None,
    fontname="helvetica",
    continuous_actions=False,
):
    # If no features are specified as integer then assume they are continuous.
    # this means that if you have integers and don't specify it splits will
    # be printed as <= 4.500 instead of <= 4
    if integer_features is None:
        integer_features = [False for _ in range(len(feature_names))]

    # If no colors are defined then create a default palette
    if colors is None:
        # Seaborn color blind palette
        palette = [
            "#0173b2",
            "#de8f05",
            "#029e73",
            "#d55e00",
            "#cc78bc",
            "#ca9161",
            "#fbafe4",
            "#949494",
            "#ece133",
            "#56b4e9",
        ]
        if continuous_actions:
            colors = palette
        else:
            colors = []
            for i in range(len(action_names)):
                colors.append(palette[i % len(palette)])

    header = f"""digraph Tree {{
node [shape=box, style=\"filled, rounded\", color=\"gray\", fillcolor=\"white\" fontname=\"{fontname}\"] ;
edge [fontname=\"{fontname}\"] ;
"""

    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value

    def sklearn_tree_to_graphviz_rec(node_id=0):
        left_id = children_left[node_id]
        right_id = children_right[node_id]
        if left_id != right_id:
            left_dot = sklearn_tree_to_graphviz_rec(left_id)
            right_dot = sklearn_tree_to_graphviz_rec(right_id)

            if node_id == 0:
                edge_label_left = "yes"
                edge_label_right = "no"
            else:
                edge_label_left = ""
                edge_label_right = ""

            feature_i = feature[node_id]
            threshold_value = threshold[node_id]

            feature_name = feature_names[feature_i]

            if integer_features[feature_i]:
                split_condition = int(threshold_value)
            else:
                split_condition = f"{threshold_value:.3f}"

            predicate = (
                f'{node_id} [label="if {feature_name} <= {split_condition}"] ;\n'
            )
            yes = left_id
            no = right_id

            edge_left = (
                f'{node_id} -> {yes} [label="{edge_label_left}", fontcolor="gray"] ;\n'
            )
            edge_right = (
                f'{node_id} -> {no} [label="{edge_label_right}", fontcolor="gray"] ;\n'
            )

            return f"{predicate}{left_dot}{right_dot}{edge_left}{edge_right}"

        if continuous_actions:
            label = ", ".join(f"{x[0]:.2f}" for x in value[node_id])
            color = colors[0]
            return f'{node_id} [label="{label}", fillcolor="{color}", color="{color}", fontcolor=white] ;\n'

        action_i = np.argmax(value[node_id])
        label = f"{action_names[action_i]}"
        color = colors[action_i]
        return f'{node_id} [label="{label}", fillcolor="{color}", color="{color}", fontcolor=white] ;\n'

    body = sklearn_tree_to_graphviz_rec()

    footer = "}"

    return header + body.strip() + footer
