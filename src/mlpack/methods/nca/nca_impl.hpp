/**
 * @file methods/nca/nca_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated NCA class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NCA_NCA_IMPL_HPP
#define MLPACK_METHODS_NCA_NCA_IMPL_HPP

// In case it was not already included.
#include "nca.hpp"

namespace mlpack {

// Just set the internal matrix reference.
template<typename DistanceType, typename OptimizerType>
NCA<DistanceType, OptimizerType>::NCA(const arma::mat& dataset,
                                    const arma::Row<size_t>& labels,
                                    DistanceType distance) :
    dataset(dataset),
    labels(labels),
    distance(distance),
    errorFunction(dataset, labels, distance)
{ /* Nothing to do. */ }

template<typename DistanceType, typename OptimizerType>
template<typename... CallbackTypes>
void NCA<DistanceType, OptimizerType>::LearnDistance(arma::mat& outputMatrix,
    CallbackTypes&&... callbacks)
{
  // See if we were passed an initialized matrix.
  if ((outputMatrix.n_rows != dataset.n_rows) ||
      (outputMatrix.n_cols != dataset.n_rows))
    outputMatrix.eye(dataset.n_rows, dataset.n_rows);

  optimizer.Optimize(errorFunction, outputMatrix, callbacks...);
}

} // namespace mlpack

#endif
