/**
 * @file methods/ann/gan/wgangp_impl.hpp
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_WGANGP_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_WGANGP_IMPL_HPP

#include "gan.hpp"

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/network_init.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
template<typename Policy>
std::enable_if_t<std::is_same_v<Policy, WGANGP>, typename MatType::elem_type>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::Evaluate(
    const MatType& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  using ElemType = typename MatType::elem_type;

  if ((parameter.is_empty()))
  {
    Reset();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  MatType realInput, realTarget;
  MakeAlias(realInput, predictors, predictors.n_rows, batchSize,
      i * predictors.n_rows);
  MakeAlias(realTarget, responses, responses.n_rows, batchSize,
      i * responses.n_rows);

  discriminator.Forward(realInput, discriminator.networkOutput);
  ElemType res = discriminator.outputLayer.Forward(discriminator.networkOutput,
      realTarget);

  noise.imbue([&]() { return noiseFunction(); });
  generator.Forward(noise, generator.networkOutput);

  MatType fakeInput, fakeTarget;
  MakeAlias(fakeInput, predictors, predictors.n_rows, batchSize,
      numFunctions * predictors.n_rows);
  fakeInput = generator.networkOutput;

  MakeAlias(fakeTarget, responses, responses.n_rows, batchSize,
      numFunctions * responses.n_rows);
  fakeTarget.fill(ElemType(-1));

  discriminator.Forward(fakeInput, discriminator.networkOutput);
  res += discriminator.outputLayer.Forward(discriminator.networkOutput,
      fakeTarget);

  // Gradient Penalty is calculated here.
  const ElemType epsilon = ElemType(Random());
  MatType interpolated = epsilon * realInput +
      (ElemType(1) - epsilon) * generator.networkOutput;
  fakeInput = interpolated;
  fakeTarget.fill(ElemType(-1));

  MatType normGradientDiscriminator;
  discriminator.Gradient(discriminator.Parameters(), numFunctions,
      normGradientDiscriminator, batchSize);
  const ElemType gradNorm = arma::norm(normGradientDiscriminator, 2);
  const ElemType penalty = gradNorm - ElemType(1);
  res += lambda * penalty * penalty;

  fakeInput = generator.networkOutput;

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
template<typename GradType, typename Policy>
std::enable_if_t<std::is_same_v<Policy, WGANGP>, typename MatType::elem_type>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::
EvaluateWithGradient(const MatType& /* parameters */,
                     const size_t i,
                     GradType& gradient,
                     const size_t /* batchSize */)
{
  using ElemType = typename MatType::elem_type;

  if (parameter.is_empty())
  {
    Reset();
  }

  if (gradient.is_empty())
  {
    gradient.set_size(parameter.n_elem, 1);
    gradient.zeros();
  }
  else
    gradient.zeros();

  if (deterministic)
  {
    deterministic = false;
    ResetDeterministic();
  }

  gradientGenerator = MatType(gradient.memptr(),
      generator.Parameters().n_elem, 1, false, false);

  gradientDiscriminator = MatType(gradient.memptr() +
      gradientGenerator.n_elem,
      discriminator.Parameters().n_elem, 1, false, false);

  noiseGradientDiscriminator.set_size(gradientDiscriminator.n_rows,
      gradientDiscriminator.n_cols);
  noiseGradientDiscriminator.zeros();

  MatType realInput;
  MakeAlias(realInput, predictors, predictors.n_rows, batchSize,
      i * predictors.n_rows);

  // Get the gradients of the Discriminator.
  ElemType res = discriminator.EvaluateWithGradient(discriminator.Parameters(),
      i, gradientDiscriminator, batchSize);

  noise.imbue([&]() { return noiseFunction(); });
  generator.Forward(noise, generator.networkOutput);

  MatType fakeInput, fakeTargets;
  MakeAlias(fakeInput, predictors, predictors.n_rows, batchSize,
      numFunctions * predictors.n_rows);
  fakeInput = generator.networkOutput;

  MakeAlias(fakeTargets, responses, responses.n_rows, batchSize,
      numFunctions * responses.n_rows);
  fakeTargets.fill(ElemType(-1));

  // Gradient Penalty is calculated here.
  const ElemType epsilon = ElemType(Random());
  MatType interpolated = epsilon * realInput +
      (ElemType(1) - epsilon) * generator.networkOutput;
  fakeInput = interpolated;
  fakeTargets.fill(ElemType(-1));
  MatType normGradientDiscriminator;
  discriminator.Gradient(discriminator.Parameters(), numFunctions,
      normGradientDiscriminator, batchSize);
  const ElemType gradNorm = arma::norm(normGradientDiscriminator, 2);
  const ElemType penalty = gradNorm - ElemType(1);
  res += lambda * penalty * penalty;

  // Restore generated samples for discriminator update.
  fakeInput = generator.networkOutput;

  res += discriminator.EvaluateWithGradient(discriminator.Parameters(),
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -D(G(noise)).
    // Pass the error from Discriminator to Generator.
    fakeTargets.fill(ElemType(1));

    discriminator.Forward(fakeInput, discriminator.networkOutput);
    discriminator.outputLayer.Backward(discriminator.networkOutput,
        fakeTargets, discriminator.error);
    discriminator.networkDelta.set_size(fakeInput.n_rows, fakeInput.n_cols);
    discriminator.network.Backward(fakeInput, discriminator.networkOutput,
        discriminator.error, discriminator.networkDelta);

    generator.error = discriminator.networkDelta;
    generator.networkDelta.set_size(noise.n_rows, noise.n_cols);
    generator.network.Backward(noise, generator.networkOutput,
        generator.error, generator.networkDelta);

    gradientGenerator.zeros();
    generator.network.Gradient(noise, generator.error, gradientGenerator);
    gradientGenerator *= multiplier;
  }

  currentBatch++;

  if (preTrainSize > 0)
  {
    preTrainSize--;
  }

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
template<typename Policy>
std::enable_if_t<std::is_same_v<Policy, WGANGP>, void>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::
Gradient(const MatType& parameters,
         const size_t i,
         MatType& gradient,
         const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

} // namespace mlpack
# endif
