/**
 * @file methods/ann/gan/gan_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_GAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_GAN_IMPL_HPP

#include "gan.hpp"

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/network_init.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/layer/identity.hpp>

namespace mlpack {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::GAN(
    Model generator,
    Model discriminator,
    InitializationRuleType initializeRule,
    Noise noiseFunction,
    const size_t noiseDim,
    const size_t batchSize,
    const size_t generatorUpdateStep,
    const size_t preTrainSize,
    const typename MatType::elem_type multiplier,
    const typename MatType::elem_type clippingParameter,
    const typename MatType::elem_type lambda):
    generator(std::move(generator)),
    discriminator(std::move(discriminator)),
    initializeRule(std::move(initializeRule)),
    noiseFunction(std::move(noiseFunction)),
    noiseDim(noiseDim),
    numFunctions(0),
    batchSize(batchSize),
    currentBatch(0),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    multiplier(multiplier),
    clippingParameter(clippingParameter),
    lambda(lambda),
    reset(false),
    deterministic(false),
    genWeights(0),
    discWeights(0)
{
  // Insert IdentityLayer for joining the Generator and Discriminator.
  // We need to do this carefully; we can't just insert into the network,
  // because that will mess up the internal state of the MultiLayer.
  std::vector<Layer<MatType>*> network =
      this->discriminator.network.Network();
  this->discriminator.network.Network().clear();

  // Reset the discriminator.
  this->discriminator.network = MultiLayer<MatType>();
  this->discriminator.network.Add(new Identity<MatType>());

  for (size_t i = 0; i < network.size(); ++i)
    this->discriminator.network.Add(network[i]);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::GAN(
    const GAN& network):
    predictors(network.predictors),
    responses(network.responses),
    generator(network.generator),
    discriminator(network.discriminator),
    initializeRule(network.initializeRule),
    noiseFunction(network.noiseFunction),
    noiseDim(network.noiseDim),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    multiplier(network.multiplier),
    clippingParameter(network.clippingParameter),
    lambda(network.lambda),
    reset(network.reset),
    currentBatch(network.currentBatch),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    noise(network.noise),
    deterministic(network.deterministic),
    genWeights(network.genWeights),
    discWeights(network.discWeights)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::GAN(
    GAN&& network):
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    generator(std::move(network.generator)),
    discriminator(std::move(network.discriminator)),
    initializeRule(std::move(network.initializeRule)),
    noiseFunction(std::move(network.noiseFunction)),
    noiseDim(network.noiseDim),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    multiplier(network.multiplier),
    clippingParameter(network.clippingParameter),
    lambda(network.lambda),
    reset(network.reset),
    currentBatch(network.currentBatch),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    noise(std::move(network.noise)),
    deterministic(network.deterministic),
    genWeights(network.genWeights),
    discWeights(network.discWeights)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>&
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::operator=(
    const GAN& other)
{
  if (this != &other)
  {
    predictors = other.predictors;
    responses = other.responses;
    generator = other.generator;
    discriminator = other.discriminator;
    initializeRule = other.initializeRule;
    noiseFunction = other.noiseFunction;
    noiseDim = other.noiseDim;
    batchSize = other.batchSize;
    generatorUpdateStep = other.generatorUpdateStep;
    preTrainSize = other.preTrainSize;
    multiplier = other.multiplier;
    clippingParameter = other.clippingParameter;
    lambda = other.lambda;
    reset = other.reset;
    currentBatch = other.currentBatch;
    parameter = other.parameter;
    numFunctions = other.numFunctions;
    noise = other.noise;
    deterministic = other.deterministic;
    genWeights = other.genWeights;
    discWeights = other.discWeights;
    gradientDiscriminator = other.gradientDiscriminator;
    noiseGradientDiscriminator = other.noiseGradientDiscriminator;
    gradientGenerator = other.gradientGenerator;

    if (!parameter.is_empty())
    {
      generator.Parameters() = MatType(parameter.memptr(), genWeights, 1, false,
          false);
      discriminator.Parameters() = MatType(parameter.memptr() + genWeights,
          discWeights, 1, false, false);

      generator.SetLayerMemory();
      discriminator.SetLayerMemory();
    }
  }

  return *this;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>&
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::operator=(
    GAN&& other)
{
  if (this != &other)
  {
    predictors = std::move(other.predictors);
    responses = std::move(other.responses);
    generator = std::move(other.generator);
    discriminator = std::move(other.discriminator);
    initializeRule = std::move(other.initializeRule);
    noiseFunction = std::move(other.noiseFunction);
    noiseDim = other.noiseDim;
    batchSize = other.batchSize;
    generatorUpdateStep = other.generatorUpdateStep;
    preTrainSize = other.preTrainSize;
    multiplier = other.multiplier;
    clippingParameter = other.clippingParameter;
    lambda = other.lambda;
    reset = other.reset;
    currentBatch = other.currentBatch;
    parameter = std::move(other.parameter);
    numFunctions = other.numFunctions;
    noise = std::move(other.noise);
    deterministic = other.deterministic;
    genWeights = other.genWeights;
    discWeights = other.discWeights;
    gradientDiscriminator = std::move(other.gradientDiscriminator);
    noiseGradientDiscriminator = std::move(other.noiseGradientDiscriminator);
    gradientGenerator = std::move(other.gradientGenerator);

    if (!parameter.is_empty())
    {
      generator.Parameters() = MatType(parameter.memptr(), genWeights, 1, false,
          false);
      discriminator.Parameters() = MatType(parameter.memptr() + genWeights,
          discWeights, 1, false, false);

      generator.SetLayerMemory();
      discriminator.SetLayerMemory();
    }
  }

  return *this;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::ResetData(
    MatType trainData)
{
  currentBatch = 0;

  numFunctions = trainData.n_cols;
  noise.set_size(noiseDim, batchSize);

  deterministic = true;
  ResetDeterministic();

  /**
   * These predictors are shared by the discriminator network. The additional
   * batch size predictors are taken from the generator network while training.
   * For more details please look in EvaluateWithGradient() function.
   */
  predictors.set_size(trainData.n_rows, numFunctions + batchSize);
  predictors.cols(0, numFunctions - 1) = std::move(trainData);
  discriminator.predictors = MatType(predictors.memptr(),
      predictors.n_rows, predictors.n_cols, false, false);

  responses.set_size(1, numFunctions + batchSize);
  responses.cols(0, numFunctions - 1).ones();
  responses.cols(numFunctions, numFunctions + batchSize - 1).zeros();
  discriminator.responses = MatType(responses.memptr(),
      responses.n_rows, responses.n_cols, false, false);

  generator.predictors.set_size(noiseDim, batchSize);
  generator.responses.set_size(predictors.n_rows, batchSize);

  generator.InputDimensions() = std::vector<size_t>({ noiseDim });
  discriminator.InputDimensions() = std::vector<size_t>({ predictors.n_rows });

  if (!reset)
  {
    Reset();
  }
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::Reset()
{
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);

  genWeights = generator.WeightSize();
  discWeights = discriminator.WeightSize();

  parameter.set_size(genWeights + discWeights, 1);
  generator.Parameters() = MatType(parameter.memptr(), genWeights, 1, false,
      false);
  discriminator.Parameters() = MatType(parameter.memptr() + genWeights,
      discWeights, 1, false, false);

  generator.SetLayerMemory();
  discriminator.SetLayerMemory();

  // Initialize the generator parameters.
  networkInit.Initialize(generator.network.Network(), parameter);
  // Initialize the discriminator parameters.
  networkInit.Initialize(discriminator.network.Network(), parameter, genWeights);

  reset = true;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::Train(
    MatType trainData,
    OptimizerType& Optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(trainData));

  return Optimizer.Optimize(*this, parameter, callbacks...);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
template<typename Policy>
std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                 std::is_same_v<Policy, DCGAN>, typename MatType::elem_type>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::Evaluate(
    const MatType& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  if (parameter.is_empty())
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
  typename MatType::elem_type res =
      discriminator.outputLayer.Forward(discriminator.networkOutput,
      realTarget);

  noise.imbue([&]() { return noiseFunction(); });
  generator.Forward(noise, generator.networkOutput);

  MatType fakeInput, fakeTarget;
  MakeAlias(fakeInput, predictors, predictors.n_rows, batchSize,
      numFunctions * predictors.n_rows);
  fakeInput = generator.networkOutput;

  MakeAlias(fakeTarget, responses, responses.n_rows, batchSize,
      numFunctions * responses.n_rows);
  fakeTarget.zeros();

  discriminator.Forward(fakeInput, discriminator.networkOutput);
  res += discriminator.outputLayer.Forward(discriminator.networkOutput,
      fakeTarget);

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
std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                 std::is_same_v<Policy, DCGAN>, typename MatType::elem_type>
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

  // Get the gradients of the Discriminator on real data.
  ElemType res = discriminator.EvaluateWithGradient(discriminator.Parameters(),
      i, gradientDiscriminator, batchSize);

  // Generate new fake data.
  noise.imbue([&]() { return noiseFunction(); });
  generator.Forward(noise, generator.networkOutput);

  MatType fakeInput, fakeTargets;
  MakeAlias(fakeInput, predictors, predictors.n_rows, batchSize,
      numFunctions * predictors.n_rows);
  fakeInput = generator.networkOutput;

  MakeAlias(fakeTargets, responses, responses.n_rows, batchSize,
      numFunctions * responses.n_rows);
  fakeTargets.zeros();

  // Gradients of the discriminator on generated (fake) data.
  res += discriminator.EvaluateWithGradient(discriminator.Parameters(),
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(noise))).
    fakeTargets.ones();

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
std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                 std::is_same_v<Policy, DCGAN>, void>
GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::
Gradient(const MatType& parameters,
         const size_t i,
         MatType& gradient,
         const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::Shuffle()
{
  const arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      numFunctions - 1, numFunctions));
  predictors.cols(0, numFunctions - 1) = predictors.cols(ordering);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::Forward(
    const MatType& input)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  generator.Forward(input, generator.networkOutput);
  discriminator.Forward(generator.networkOutput, discriminator.networkOutput);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::
Predict(MatType input, MatType& output)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  Forward(input);

  output = discriminator.networkOutput;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::
ResetDeterministic()
{
  discriminator.SetNetworkMode(!deterministic);
  generator.SetNetworkMode(!deterministic);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename MatType
>
template<typename Archive>
void GAN<Model, InitializationRuleType, Noise, PolicyType, MatType>::
serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(parameter));
  ar(CEREAL_NVP(generator));
  ar(CEREAL_NVP(discriminator));
  ar(CEREAL_NVP(reset));
  ar(CEREAL_NVP(genWeights));
  ar(CEREAL_NVP(discWeights));

  if (cereal::is_loading<Archive>())
  {
    genWeights = generator.Parameters().n_elem;
    discWeights = discriminator.Parameters().n_elem;

    parameter.set_size(genWeights + discWeights, 1);
    if (genWeights > 0)
    {
      parameter.submat(0, 0, genWeights - 1, 0) = generator.Parameters();
    }
    if (discWeights > 0)
    {
      parameter.submat(genWeights, 0,
          genWeights + discWeights - 1, 0) = discriminator.Parameters();
    }

    generator.Parameters() = MatType(parameter.memptr(), genWeights, 1, false,
        false);
    discriminator.Parameters() = MatType(parameter.memptr() + genWeights,
        discWeights, 1, false, false);

    generator.SetLayerMemory();
    discriminator.SetLayerMemory();

    deterministic = true;
    ResetDeterministic();
  }
}

} // namespace mlpack
# endif
