/**
 * @file methods/radical/radical.hpp
 * @author Nishant Mehta
 *
 * Declaration of Radical class (RADICAL is Robust, Accurate, Direct ICA
 * aLgorithm).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RADICAL_RADICAL_HPP
#define MLPACK_METHODS_RADICAL_RADICAL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * An implementation of RADICAL, an algorithm for independent component
 * analysis (ICA).
 *
 * Let X be a matrix where each column is a point and each row a dimension.
 * The goal is to find a square unmixing matrix W such that Y = W X and
 * the rows of Y are independent components.
 *
 * For more details, see the following paper:
 *
 * @code
 * @article{learned2003ica,
 *   title = {ICA Using Spacings Estimates of Entropy},
 *   author = {Learned-Miller, E.G. and Fisher III, J.W.},
 *   journal = {Journal of Machine Learning Research},
 *   volume = {4},
 *   pages = {1271--1295},
 *   year = {2003}
 * }
 * @endcode
 */
class Radical
{
 public:
  /**
   * Set the parameters to RADICAL.
   *
   * @param noiseStdDev Standard deviation of the Gaussian noise added to the
   *    replicates of the data points during Radical2D
   * @param replicates Number of Gaussian-perturbed replicates to use (per
   *    point) in Radical2D
   * @param angles Number of angles to consider in brute-force search during
   *    Radical2D
   * @param sweeps Number of sweeps.  Each sweep calls Radical2D once for each
   *    pair of dimensions
   * @param m The variable m from Vasicek's m-spacing estimator of entropy.
   */
  Radical(const double noiseStdDev = 0.175,
          const size_t replicates = 30,
          const size_t angles = 150,
          const size_t sweeps = 0,
          const size_t m = 0);

  /**
   * Run RADICAL.
   *
   * @param matX Input data into the algorithm - a matrix where each column is
   *    a point and each row is a dimension.
   * @param matY Estimated independent components - a matrix where each column
   *    is a point and each row is an estimated independent component.
   * @param matW Estimated unmixing matrix, where matY = matW * matX.
   * @param timers Optional Timers struct for storing timing information.
   */
  template<typename MatType = arma::mat>
  [[deprecated("Will be removed in mlpack 5.0.0.  Use Apply() instead.")]]
  void DoRadical(const MatType& matX,
                 MatType& matY,
                 MatType& matW,
                 util::Timers& timers = IO::GetTimers());

  /**
   * Run RADICAL.
   *
   * @param matX Input data into the algorithm - a matrix where each column is
   *    a point and each row is a dimension.
   * @param matY Estimated independent components - a matrix where each column
   *    is a point and each row is an estimated independent component.
   * @param matW Estimated unmixing matrix, where matY = matW * matX.
   * @param timers Optional Timers struct for storing timing information.
   */
  template<typename MatType = arma::mat>
  void Apply(const MatType& matX,
             MatType& matY,
             MatType& matW,
             util::Timers& timers = IO::GetTimers());

  /**
   * Vasicek's m-spacing estimator of entropy, with overlap modification from
   * (Learned-Miller and Fisher, 2003).
   *
   * @param x Empirical sample (one-dimensional) over which to estimate entropy.
   * @param m The variable m from Vasicek's m-spacing estimator of entropy.
   */
  template<typename VecType>
  typename VecType::elem_type Vasicek(VecType& x, const size_t m) const;

  /**
   * Make replicates of each data point (the number of replicates is set in
   * either the constructor or with Replicates()) and perturb data with Gaussian
   * noise with standard deviation noiseStdDev.
   */
  template<typename MatType>
  void CopyAndPerturb(MatType& xNew, const MatType& x) const;

  //! Two-dimensional version of RADICAL.
  template<typename MatType>
  typename MatType::elem_type Apply2D(
      const MatType& matX,
      const size_t m,
      MatType& perturbed, // auxiliary memory
      MatType& candidate, // auxiliary memory
      util::Timers& timers = IO::GetTimers());

  //! Get the standard deviation of the additive Gaussian noise.
  double NoiseStdDev() const { return noiseStdDev; }
  //! Modify the standard deviation of the additive Gaussian noise.
  double& NoiseStdDev() { return noiseStdDev; }

  //! Get the number of Gaussian-perturbed replicates used per point.
  size_t Replicates() const { return replicates; }
  //! Modify the number of Gaussian-perturbed replicates used per point.
  size_t& Replicates() { return replicates; }

  //! Get the number of angles considered during brute-force search.
  size_t Angles() const { return angles; }
  //! Modify the number of angles considered during brute-force search.
  size_t& Angles() { return angles; }

  //! Get the number of sweeps.
  size_t Sweeps() const { return sweeps; }
  //! Modify the number of sweeps.
  size_t& Sweeps() { return sweeps; }

  //! Get the value of m to use for Vasicek's m-spacing estimator of entropy.
  size_t M() const { return m; }
  //! Set the value of m to use for Vasicek's m-spacing estimator of entropy.
  size_t& M() { return m; }

  // Serialize the Radical object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Standard deviation of the Gaussian noise added to the replicates of
  //! the data points during Radical2D.
  double noiseStdDev;

  //! Number of Gaussian-perturbed replicates to use (per point) in Radical2D.
  size_t replicates;

  //! Number of angles to consider in brute-force search during Radical2D.
  size_t angles;

  //! Number of sweeps; each sweep calls Radical2D once for each pair of
  //! dimensions.
  size_t sweeps;

  //! Value of m to use for Vasicek's m-spacing estimator of entropy.
  size_t m;
};

template<typename MatType>
void WhitenFeatureMajorMatrix(const MatType& matX,
                              MatType& matXWhitened,
                              MatType& matWhitening);

} // namespace mlpack

// Include implementation.
#include "radical_impl.hpp"

#endif
