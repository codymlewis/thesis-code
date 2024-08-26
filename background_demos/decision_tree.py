from sklearn import (
    tree,
    datasets,
    model_selection,
    metrics,
)
import graphviz


if __name__ == "__main__":
    data = datasets.load_iris()
    X, Y = data.data, data.target
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=42,
    )
    model = tree.DecisionTreeClassifier(random_state=42)
    print("Training a decision tree on the iris dataset...")
    model.fit(train_X, train_Y)
    accuracy_val = metrics.accuracy_score(test_Y, model.predict(test_X))
    print(f"Done. Testing accuracy: {accuracy_val:.5%}")
    gv_source_str = tree.export_graphviz(
        model,
        feature_names=data.feature_names,
        class_names=data.target_names,
    )
    graph = graphviz.Source(gv_source_str)
    tree_render_fn = "iris_tree.gv"
    graph.render(tree_render_fn)
    print(f"Rendered tree to {tree_render_fn}")
