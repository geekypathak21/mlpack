/**
 * @file core/tree/rectangle_tree/r_plus_plus_tree_descent_heuristic.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of RPlusPlusTreeDescentHeuristic, a class that chooses the best
 * child of a node in an R++ tree when inserting a new point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_DESCENT_HEURISTIC_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_DESCENT_HEURISTIC_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

class RPlusPlusTreeDescentHeuristic
{
 public:
  /**
   * Evaluate the node using the R++-tree heuristic.
   *
   * @param node The node that is being evaluated.
   * @param point The number of the point that is being inserted.
   */
  template<typename TreeType>
  static size_t ChooseDescentNode(TreeType* node, const size_t point);

  /**
   * Evaluate the node using the R++-tree heuristic.
   *
   * @param node The node that is being evaluated.
   * @param insertedNode The node that is being inserted.
   */
  template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node,
                                  const TreeType* insertedNode);
};

} // namespace mlpack

#include "r_plus_plus_tree_descent_heuristic_impl.hpp"

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_DESCENT_HEURISTIC_HPP
