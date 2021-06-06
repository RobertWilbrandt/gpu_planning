#pragma once

#include <vector>

namespace gpu_planning {

template <typename V, typename E>
class Graph {
 public:
  Graph();

  size_t add_node(const V& data);
  size_t add_edge(const E& data, size_t v1_id, size_t v2_id);

  size_t num_nodes() const;
  size_t num_edges() const;

  const V& node(size_t id) const;
  const E& edge(size_t id) const;

  size_t edge_from(size_t id) const;
  size_t edge_to(size_t id) const;

 private:
  struct Node {
    V data;
    std::vector<size_t> edge_ids;
  };

  struct Edge {
    E data;
    size_t from_id;
    size_t to_id;
  };

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
};

template <typename V, typename E>
Graph<V, E>::Graph() : nodes_{}, edges_{} {}

template <typename V, typename E>
size_t Graph<V, E>::add_node(const V& data) {
  Node node;
  node.data = data;

  nodes_.push_back(node);
  return nodes_.size() - 1;
}

template <typename V, typename E>
size_t Graph<V, E>::add_edge(const E& data, size_t v1_id, size_t v2_id) {
  Edge edge;
  edge.data = data;
  edge.from_id = v1_id;
  edge.to_id = v2_id;

  edges_.push_back(edge);
  const size_t edge_id = edges_.size() - 1;

  nodes_[v1_id].edge_ids.push_back(edge_id);
  nodes_[v2_id].edge_ids.push_back(edge_id);

  return edge_id;
}

template <typename V, typename E>
size_t Graph<V, E>::num_nodes() const {
  return nodes_.size();
}

template <typename V, typename E>
size_t Graph<V, E>::num_edges() const {
  return edges_.size();
}

template <typename V, typename E>
const V& Graph<V, E>::node(size_t id) const {
  return nodes_[id].data;
}

template <typename V, typename E>
const E& Graph<V, E>::edge(size_t id) const {
  return edges_[id].data;
}

template <typename V, typename E>
size_t Graph<V, E>::edge_from(size_t id) const {
  return edges_[id].from_id;
}

template <typename V, typename E>
size_t Graph<V, E>::edge_to(size_t id) const {
  return edges_[id].to_id;
}

}  // namespace gpu_planning
