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
    outSize(0),
    reset(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
RBF<InputDataType, OutputDataType>::RBF(
    const size_t inSize,
    const size_t outSize) :
    inSize(inSize),
    outSize(outSize),
    reset(false)
{
  weights.set_size(outSize * inSize + outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
void RBF<InputDataType, OutputDataType>::Reset()
{
  centres = arma::mat(weights.memptr(), inSize, outSize, false, false);
  sigmas = arma::mat(weights.memptr() + centres.n_elem,
      1, outSize, false, false);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RBF<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  if(!reset)
  {
    centres = arma::normcdf(centres, 0,1);
    sigmas.ones();
    sigmas = sigmas / outSize;
    reset = true;
  }
  distances = arma::mat(outSize, input.n_cols);

  for (size_t i = 0; i < input.n_cols; i++)
  {
    arma::mat temp = centres.each_col() - input.col(i);
    distances.col(i) = arma::pow(arma::sum(
                                 arma::pow((temp),
                                 2), 0), 0.5).t();
  }
  output = distances.each_col() % sigmas.t();
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RBF<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = centres.t() * gy;
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
