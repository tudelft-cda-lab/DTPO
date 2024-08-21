import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
from sklearn.utils import check_random_state

from .utils import Node, Leaf


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))


class DTPolicy:
    def __init__(self, max_depth=5, max_leaf_nodes=None, random_state=None):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes
        )
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, obss_test, acts_train, acts_test = train_test_split(
            obss, acts, train_size=train_frac, random_state=self.random_state
        )
        self.fit(obss_train, acts_train)
        print("Train accuracy: {}".format(accuracy(self, obss_train, acts_train)))
        print("Test accuracy: {}".format(accuracy(self, obss_test, acts_test)))
        print("Number of nodes: {}".format(self.tree.tree_.node_count))

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth, self.max_leaf_nodes, self.random_state)
        clone.tree = self.tree
        return clone


def identity_state_transformer(obs):
    return obs


# TODO: possibly refactor this into a util function so that dtpo.py can use it too


def prune_sklearn_tree(sklearn_tree: DecisionTreeClassifier):
    """
    Returns a scikit-learn Tree object with the pruned and
    discretized decision tree policy.
    """
    n_actions = sklearn_tree.n_classes_
    n_features_in = sklearn_tree.n_features_in_

    tree = Tree(n_features_in, np.array([n_actions]), 1)

    feature = sklearn_tree.tree_.feature
    threshold = sklearn_tree.tree_.threshold
    children_left = sklearn_tree.tree_.children_left
    children_right = sklearn_tree.tree_.children_right
    value = sklearn_tree.tree_.value

    def sklearn_to_pruned_tree_rec(node_id):
        # If this is a leaf node
        left_id = children_left[node_id]
        right_id = children_right[node_id]
        if left_id == right_id:
            chosen_action = np.argmax(value[node_id])
            discretized_value = np.zeros(n_actions)
            discretized_value[chosen_action] = 1
            return Leaf(discretized_value.reshape(1, -1))

        left_subtree = sklearn_to_pruned_tree_rec(left_id)
        right_subtree = sklearn_to_pruned_tree_rec(right_id)

        if (
            isinstance(left_subtree, Leaf)
            and isinstance(right_subtree, Leaf)
            and np.all(np.isclose(left_subtree.value, right_subtree.value))
        ):
            return Leaf(left_subtree.value)

        return Node(feature[node_id], threshold[node_id], left_subtree, right_subtree)

    pruned_tree_object = sklearn_to_pruned_tree_rec(0)

    def count_and_assign_ids(subtree, depth, node_count, max_depth):
        if depth > max_depth:
            max_depth = depth

        subtree.id = node_count
        node_count += 1

        if isinstance(subtree, Node):
            node_count, max_depth = count_and_assign_ids(
                subtree.left, depth + 1, node_count, max_depth
            )
            return count_and_assign_ids(subtree.right, depth + 1, node_count, max_depth)

        return node_count, max_depth

    node_count, max_depth = count_and_assign_ids(pruned_tree_object, 0, 0, 0)

    nodes = [None] * node_count
    values = [np.zeros((1, n_actions))] * node_count

    def pruned_tree_to_state_nodes_values(subtree, nodes, values):
        if isinstance(subtree, Leaf):
            nodes[subtree.id] = (-1, -1, -2, -2, 0, 0, 0)
            values[subtree.id] = subtree.value
        else:
            nodes[subtree.id] = (
                subtree.left.id,
                subtree.right.id,
                subtree.feature,
                subtree.threshold,
                0,
                0,
                0,
            )
            pruned_tree_to_state_nodes_values(subtree.left, nodes, values)
            pruned_tree_to_state_nodes_values(subtree.right, nodes, values)

    pruned_tree_to_state_nodes_values(pruned_tree_object, nodes, values)

    nodes = np.array(
        nodes,
        dtype=[
            ("left_child", "<i8"),
            ("right_child", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
        ],
    )
    values = np.array(values)

    state = {
        "n_features_": n_features_in,
        "max_depth": max_depth,
        "node_count": node_count,
        "nodes": nodes,
        "values": values,
    }
    tree.__setstate__(state)
    return tree


# TODO: add verbose setting


class ViperLearner:
    def __init__(
        self,
        max_depth=4,
        max_leaf_nodes=None,
        n_batch_rollouts=10,
        max_samples=200000,
        max_iters=80,
        train_frac=0.8,
        is_reweight=True,
        n_test_rollouts=50,
        is_train=True,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.n_batch_rollouts = n_batch_rollouts
        self.max_samples = max_samples
        self.max_iters = max_iters
        self.train_frac = train_frac
        self.is_reweight = is_reweight
        self.n_test_rollouts = n_test_rollouts
        self.is_train = is_train
        self.random_state = random_state

    def learn(self, env, teacher):
        # TODO: probably remove state transformers (also from deeper in the code)

        random_state = check_random_state(self.random_state)

        student = DTPolicy(self.max_depth, self.max_leaf_nodes, random_state)

        # Train student
        if self.is_train:
            student = train_dagger(
                env,
                teacher,
                student,
                identity_state_transformer,
                self.max_iters,
                self.n_batch_rollouts,
                self.max_samples,
                self.train_frac,
                self.is_reweight,
                self.n_test_rollouts,
                random_state,
            )
            # TODO: save the policy somewhere
        else:
            raise NotImplementedError()

        # Test student
        rew = test_policy(
            env, student, identity_state_transformer, self.n_test_rollouts
        )
        print("Final reward: {}".format(rew))
        print("Number of nodes: {}".format(student.tree.tree_.node_count))

        student.tree.tree_ = prune_sklearn_tree(student.tree)

        self.tree_policy_ = student.tree


class TransformerPolicy:
    def __init__(self, policy, state_transformer):
        self.policy = policy
        self.state_transformer = state_transformer

    def predict(self, obss):
        return self.policy.predict(
            np.array([self.state_transformer(obs) for obs in obss])
        )


def get_rollout(env, policy, render):
    # obs, done = np.array(env.reset()), False
    obs, info = env.reset()
    done = False
    trunc = False
    rollout = []

    while not done and not trunc:
        # Render
        if render:
            env.unwrapped.render()

        # Action
        act = policy.predict(np.array([obs]))[0]

        # Step
        next_obs, rew, done, trunc, info = env.step(act)

        # Rollout (s, a, r)
        rollout.append((obs, act, rew))

        # Update (and remove LazyFrames)
        obs = np.array(next_obs)

    return rollout


def get_rollouts(env, policy, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy, render))
    return rollouts


def _sample(obss, acts, qs, max_pts, is_reweight, random_state):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1) + 1e-8
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = random_state.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = random_state.choice(
            len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False
        )

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx]


def test_policy(env, policy, state_transformer, n_test_rollouts):
    wrapped_student = TransformerPolicy(policy, state_transformer)
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout(env, wrapped_student, False)
        cum_rew += sum((rew for _, _, rew in student_trace))
    return cum_rew / n_test_rollouts


def identify_best_policy(env, policies, state_transformer, n_test_rollouts):
    print("Initial policy count: {}".format(len(policies)))
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1) / 2)
        print("Current policy count: {}".format(n_policies))

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env, policy, state_transformer, n_test_rollouts)
            new_policies.append((policy, new_rew))
            print("Reward update: {} -> {}".format(rew, new_rew))

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0]


def train_dagger(
    env,
    teacher,
    student,
    state_transformer,
    max_iters,
    n_batch_rollouts,
    max_samples,
    train_frac,
    is_reweight,
    n_test_rollouts,
    random_state,
):
    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []
    wrapped_student = TransformerPolicy(student, state_transformer)

    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((state_transformer(obs) for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))
    qs.extend(teacher.predict_q(np.array([obs for obs, _, _ in trace])))

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        print("Iteration {}/{}".format(i, max_iters))

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(
            np.array(obss),
            np.array(acts),
            np.array(qs),
            max_samples,
            is_reweight,
            random_state,
        )
        print("Training student with {} points".format(len(cur_obss)))
        student.train(cur_obss, cur_acts, train_frac)

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, wrapped_student, False, n_batch_rollouts)
        student_obss = np.array([obs for obs, _, _ in student_trace])

        # Step 2c: Query the oracle for supervision
        teacher_qs = teacher.predict_q(
            student_obss
        )  # at the interface level, order matters, since teacher.predict may run updates
        teacher_acts = teacher.predict(student_obss)

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((state_transformer(obs) for obs in student_obss))
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        print("Student reward: {}".format(cur_rew))

        students.append((student.clone(), cur_rew))

    max_student = identify_best_policy(
        env, students, state_transformer, n_test_rollouts
    )

    return max_student
