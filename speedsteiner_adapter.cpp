#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dst/dst.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }
    int root = 0;
    int edge_count = 0;
    int terminal_count = 0;
    if (!(std::cin >> root >> edge_count >> terminal_count)) {
        return 2;
    }
    std::vector<int> terminals(terminal_count);
    for (int& terminal : terminals) {
        std::cin >> terminal;
    }
    std::vector<std::pair<int, int>> edges;
    std::vector<float> weights;
    edges.reserve(edge_count);
    weights.reserve(edge_count);
    for (int position = 0; position < edge_count; ++position) {
        int tail = 0;
        int head = 0;
        float weight = 0;
        if (!(std::cin >> tail >> head >> weight)) {
            return 2;
        }
        edges.emplace_back(tail, head);
        weights.push_back(weight);
    }
    dst::DST instance(edges, weights, root, terminals);
    std::shared_ptr<dst::Tree> tree;
    const std::string method(argv[1]);
    if (method == "fast_level2") {
        tree = instance.fast_level2_alg()->to_tree();
    } else if (method == "level2") {
        tree = instance.level2_alg()->to_tree();
    } else {
        return 2;
    }
    if (!tree->is_valid(instance.terms_dm)) {
        return 3;
    }
    const auto selected = tree->edges();
    std::vector<std::pair<int, int>> original_edges;
    for (const auto& edge : *selected) {
        if (!instance.terms_dm.count(edge.second)) {
            original_edges.push_back(edge);
        }
    }
    std::cout << std::setprecision(17) << tree->cost << '\n';
    std::cout << original_edges.size() << '\n';
    for (const auto& edge : original_edges) {
        std::cout << edge.first << ' ' << edge.second << '\n';
    }
    return 0;
}
