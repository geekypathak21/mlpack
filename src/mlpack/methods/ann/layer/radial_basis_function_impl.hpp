/**
 * @file radial_basis_impl.hpp
 * @author Himanshu Pathak
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RBF_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RBF_IMPL_HPP

// In case it hasn't yet been included.
#include "dropout.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
RBF<InputDataType, OutputDataType>::RBF() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
RBF<InputDataType, OutputDataType>::RBF(
    const size_t inSize,
    const size_t outSize) :
    inSize(inSize),
    outSize(outSize)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RBF<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  centres = arma::randu(outSize, inSize);
  centres = arma::normcdf(centres, 0, 1);
  sigmas = arma::ones(outSize);
  arma::cube x = arma::cube(outSize, input.n_rows, input.n_rows);
  distances = arma::mat(input.n_rows, outSize);
  arma::cube c = arma::cube(outSize, input.n_rows, input.n_rows);

  for(size_t i=0;i < input.n_rows;i++)
  {
    for (size_t j = 0;j < outSize; j++)
    {
      x.slice(i).row(j) = input.row(i);
    }
  }

  for(size_t i=0; i < input.n_rows; i++)
  {
    c.slice(i)= centres;
  }

  for(size_t i=0;i < input.n_rows;i++)
  { 
  distances.row(i) = arma::pow(arma::sum (arma::pow ((x.slice(i) - c.slice(i)), 2), 1), 0.5).t() * sigmas(i);
  }

  output = distances;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RBF<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = distances.t() * gy;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void RBF<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(distances);
  ar & BOOST_SERIALIZATION_NVP(sigmas);
  ar & BOOST_SERIALIZATION_NVP(centres);
}

} // namespace ann
} // namespace mlpack
#endif
