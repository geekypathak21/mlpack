/**
 * @file gaussian_function.hpp
 * @author Himanshu Pathak
 *
 * Definition and implementation of the logistic function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The logistic function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \frac{1}{1 + e^{-x}} \\
 * f'(x) &=& f(x) * (1 - f(x)) \\
 * f^{-1}(y) &=& ln(\frac{y}{1-y})
 * @f}
 */
class GaussianFunction
{
 public:
  /**
   * Computes the gaussian function.
   *
   * @param x Input data.
   * @return f(x).
   */
  template<typename eT>
  static double Fn(const eT x)
  {
    return std::exp(-1 * std::pow(x, 2));
  }

  /**
   * Computes the gaussian function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = arma::exp(-1 * arma::pow(x, 2));
  }

  /**
   * Computes the first derivative of the logistic function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return 2 * -y * std::exp(-1 * std::pow(y, 2));
  }

  /**
   * Computes the first derivatives of the logistic function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x = 2 * -y % arma::exp(-1 * arma::pow(y, 2));
  }
}; // class LogisticFunction

} // namespace ann
} // namespace mlpack

#endif
